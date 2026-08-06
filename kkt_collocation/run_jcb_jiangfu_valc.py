"""VALC on the parameterized Jiang--Fu JCB benchmark.

Training controls are produced by the authors' MATLAB implementation of
Jiang--Fu Algorithm 1.  The benchmark extension is explicit: x1(0) varies
in [-0.1,0.1], while x2(0)=-1 remains the published value.  A supervised
policy and a dual-guided reduced-NLP KKT-refined copy are selected using an
independent 60-point validation cohort, then evaluated with HDS correction
against cold-start Jiang--Fu solutions on a disjoint 400-point cohort.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import nnls
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from offline_safe_control.kkt_regularization import augmented_lagrangian_kkt_residual


@dataclass(frozen=True)
class Config:
    horizon: float = 1.0
    zoh_steps: int = 20
    u_min: float = -15.0
    u_max: float = 15.0
    train_points: int = 400
    validation_points: int = 60
    test_points: int = 400
    collocation_substeps: int = 10
    epochs: int = 1500
    kkt_epochs: int = 30
    value_weight: float = 0.1
    kkt_weight: float = 1e-3
    anchor_weight: float = 0.5
    # The manuscript's original gate uses raw physical constraint units.
    # No additional engineering-normalization scale is introduced for JCB.
    gate_severe_magnitude: float = 0.01
    gate_allowed_severe_rate: float = 0.05
    margin: float = 1e-6
    lambda_grid: int = 101

    @property
    def dt(self) -> float:
        return self.horizon / self.zoh_steps


def initial(p: float) -> np.ndarray:
    return np.array([p, -1.0, 0.0, 0.0], dtype=float)


def ode(_t: float, z: np.ndarray, u: float) -> np.ndarray:
    return np.array([z[1], -z[1] + u, 1.0, z[0] ** 2 + z[1] ** 2 + .005 * u ** 2])


def g(z: np.ndarray) -> float:
    return float(z[1] - 8.0 * (z[2] - .5) ** 2 + .5)


def gdot(z: np.ndarray, u: float) -> float:
    return float(-z[1] + u - 16.0 * (z[2] - .5))


def rollout_cost(p: float, controls: np.ndarray, cfg: Config) -> float:
    """Exact state propagation and five-point quadrature of the JCB cost."""
    x1, x2, cost = float(p), -1.0, 0.0
    xi, wi = np.polynomial.legendre.leggauss(5)
    for u in np.asarray(controls, dtype=float):
        u = float(u)
        q = .5 * cfg.dt * (xi + 1.0)
        e = np.exp(-q)
        qx2 = u + (x2 - u) * e
        qx1 = x1 + u * q + (x2 - u) * (1.0 - e)
        cost += .5 * cfg.dt * np.sum(wi * (qx1 * qx1 + qx2 * qx2 + .005 * u * u))
        e1 = np.exp(-cfg.dt)
        x1, x2 = x1 + u * cfg.dt + (x2 - u) * (1.0 - e1), u + (x2 - u) * e1
    return float(cost)


def torch_rollout(p: torch.Tensor, controls: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable reduced-space 201-node transcription for KKT refinement."""
    batch = p.shape[0]
    h = cfg.dt / cfg.collocation_substeps
    x1, x2 = p, torch.full_like(p, -1.0)
    t = torch.zeros_like(p)
    running = torch.zeros_like(p)
    gs = [x2 - 8.0 * (t - .5).square() + .5]
    for j in range(cfg.zoh_steps * cfg.collocation_substeps):
        u = controls[:, j // cfg.collocation_substeps]
        l0 = x1.square() + x2.square() + .005 * u.square()
        e = torch.exp(torch.as_tensor(-h, dtype=p.dtype, device=p.device))
        nx1 = x1 + u * h + (x2 - u) * (1.0 - e)
        nx2 = u + (x2 - u) * e
        l1 = nx1.square() + nx2.square() + .005 * u.square()
        running = running + .5 * h * (l0 + l1)
        x1, x2, t = nx1, nx2, t + h
        gs.append(x2 - 8.0 * (t - .5).square() + .5)
    return running, torch.stack(gs, dim=1)


class PolicyValue(nn.Module):
    def __init__(self, controls: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh()
        )
        self.value = nn.Linear(64, 1)
        self.control = nn.Linear(64, controls)

    def forward(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(p)
        return self.value(h), 15.0 * torch.tanh(self.control(h))


def load_jiang_labels(path: Path) -> dict[str, np.ndarray]:
    raw = loadmat(path)
    needed = ("pStored", "controls", "objectives", "solveSeconds", "exitFlags", "completed")
    missing = [key for key in needed if key not in raw]
    if missing:
        raise ValueError(f"{path} does not contain {missing}")
    if not np.all(np.asarray(raw["completed"]).reshape(-1)):
        raise ValueError(f"{path} contains unfinished Jiang--Fu solves")
    if np.any(np.asarray(raw["exitFlags"]).reshape(-1) <= 0):
        raise ValueError(f"{path} contains failed Jiang--Fu solves")
    p = np.asarray(raw["pStored"], dtype=float).reshape(-1)
    controls = np.asarray(raw["controls"], dtype=float)
    objective = np.asarray(raw["objectives"], dtype=float).reshape(-1)
    seconds = np.asarray(raw["solveSeconds"], dtype=float).reshape(-1)
    if controls.shape != (len(p), 20):
        raise ValueError(f"Unexpected Jiang--Fu control shape {controls.shape}")
    return {"p": p, "controls": controls, "objective": objective, "seconds": seconds}


def make_corrector(cfg: Config) -> HDSLambdaCorrector:
    return HDSLambdaCorrector(
        ode, g, gdot, (cfg.u_min, cfg.u_max),
        HDSLambdaConfig(grid_size=cfg.lambda_grid, safety_margin=cfg.margin, max_step_fraction=200.0),
    )


def reconstruct_duals(p: np.ndarray, controls: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Recover nonnegative reduced-NLP multipliers on a fixed 201-node grid.

    These are numerical dual labels for the optional KKT refinement only.  The
    control labels themselves remain the unmodified Jiang--Fu Algorithm-1
    solutions, and no multiplier is interpreted as a continuous-time measure.
    """
    duals = np.zeros((len(p), cfg.zoh_steps * cfg.collocation_substeps + 1), dtype=np.float32)
    residuals = np.zeros(len(p), dtype=float)
    for i, (pi, ui) in enumerate(zip(p, controls)):
        pt = torch.tensor([pi], dtype=torch.float64)
        ut = torch.tensor(ui[None, :], dtype=torch.float64, requires_grad=True)
        objective, constraints = torch_rollout(pt, ut, cfg)
        grad_j = torch.autograd.grad(objective.sum(), ut, retain_graph=True)[0][0].detach().numpy()
        active = torch.where(constraints[0] > -1e-3)[0].tolist()
        if not active:
            residuals[i] = float(np.linalg.norm(grad_j))
            continue
        gradients = []
        for node in active:
            gradient = torch.autograd.grad(constraints[0, node], ut, retain_graph=True)[0][0]
            gradients.append(gradient.detach().numpy())
        design = np.column_stack(gradients)
        multipliers, _ = nnls(design, -grad_j)
        duals[i, active] = multipliers.astype(np.float32)
        residuals[i] = float(np.linalg.norm(grad_j + design @ multipliers))
    return duals, residuals


def train_policy(data: dict[str, np.ndarray], cfg: Config, seed: int, *, base: PolicyValue | None = None,
                 kkt: bool = False) -> tuple[PolicyValue, float, float]:
    torch.manual_seed(seed)
    p = torch.tensor(data["p"][:, None], dtype=torch.float32)
    uref = torch.tensor(data["controls"], dtype=torch.float32)
    jref = torch.tensor(data["objective"][:, None], dtype=torch.float32)
    dual = torch.tensor(data["duals"], dtype=torch.float32)
    pm, ps = p.mean(), p.std(unbiased=False).clamp_min(1e-6)
    jm, js = jref.mean(), jref.std(unbiased=False).clamp_min(1e-6)
    model = PolicyValue(cfg.zoh_steps) if base is None else base
    model.train()
    with torch.no_grad():
        anchor = model((p - pm) / ps)[1].detach().clone() if kkt else None
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 if not kkt else 1e-5)
    steps = cfg.kkt_epochs if kkt else cfg.epochs
    for _ in range(steps):
        predicted_value, predicted_controls = model((p - pm) / ps)
        loss = nn.functional.mse_loss(predicted_controls, uref)
        loss = loss + cfg.value_weight * nn.functional.mse_loss(predicted_value, (jref - jm) / js)
        if kkt:
            objective, constraints = torch_rollout(p[:, 0], predicted_controls, cfg)
            residual = augmented_lagrangian_kkt_residual(objective, predicted_controls, constraints, dual, 10.0).total
            # Scaling by a detached number makes the KKT contribution stable
            # across the 400 Jiang--Fu labels without changing its gradient.
            loss = loss + cfg.kkt_weight * residual / residual.detach().clamp_min(1.0)
            loss = loss + cfg.anchor_weight * nn.functional.mse_loss(predicted_controls, anchor)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model.eval(), float(pm), float(ps)


def predict(model: PolicyValue, mean: float, std: float, p: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    with torch.no_grad():
        controls = model(torch.tensor(((p - mean) / std)[:, None], dtype=torch.float32))[1].numpy()
    return controls, (time.perf_counter() - start) / len(p)


def evaluate(name: str, model: PolicyValue, mean: float, std: float, reference: dict[str, np.ndarray], cfg: Config) -> list[dict]:
    controls, inference_seconds = predict(model, mean, std, reference["p"])
    corrector = make_corrector(cfg)
    rows: list[dict] = []
    for index, (p, nominal, ref_obj) in enumerate(zip(reference["p"], controls, reference["objective"])):
        z0 = initial(float(p))
        nominal_peak = corrector.audit(z0, nominal, cfg.dt)
        start = time.perf_counter()
        outcome = corrector.correct(z0, nominal, cfg.dt)
        correction_seconds = time.perf_counter() - start
        if outcome.accepted:
            applied = outcome.controls
            applied_peak = corrector.audit(z0, applied, cfg.dt)
            applied_objective = rollout_cost(float(p), applied, cfg)
            relative = 100.0 * abs(applied_objective - ref_obj) / max(abs(ref_obj), 1e-12)
        else:
            applied_peak = np.nan
            applied_objective = np.nan
            relative = np.nan
        rows.append({
            "method": name, "index": index, "p": float(p), "reference_objective": float(ref_obj),
            "nominal_peak": float(nominal_peak), "accepted": bool(outcome.accepted),
            "fallback": bool(not outcome.accepted), "applied_peak": float(applied_peak),
            "applied_objective": float(applied_objective), "relative_objective_difference_percent": float(relative),
            "corrected_segments": int(sum(segment.corrected for segment in outcome.segments)),
            "inference_seconds": float(inference_seconds), "hds_seconds": float(correction_seconds),
        })
    return rows


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1))}


def summarise(rows: list[dict]) -> dict:
    accepted = np.asarray([row["accepted"] for row in rows], dtype=bool)
    nominal = np.asarray([row["nominal_peak"] for row in rows], dtype=float)
    applied = np.asarray([row["applied_peak"] for row in rows], dtype=float)
    finite_relative = np.asarray([row["relative_objective_difference_percent"] for row in rows], dtype=float)
    return {
        "points": len(rows), "acceptance_rate_percent": float(100.0 * accepted.mean()),
        "fallback_rate_percent": float(100.0 * (~accepted).mean()),
        "nominal_violation_rate_percent": float(100.0 * np.mean(nominal > 0.0)),
        "nominal_peak_max": float(nominal.max()), "accepted_peak_max": float(np.nanmax(applied)),
        "relative_objective_difference_percent": stats(finite_relative[np.isfinite(finite_relative)]),
        "applied_objective": stats(np.asarray([row["applied_objective"] for row in rows], dtype=float)[accepted]),
        "corrected_segments_mean": float(np.mean([row["corrected_segments"] for row in rows])),
        "inference_seconds": float(np.mean([row["inference_seconds"] for row in rows])),
        "hds_seconds": stats(np.asarray([row["hds_seconds"] for row in rows], dtype=float)),
    }


def choose_branch(supervised_rows: list[dict], cfg: Config) -> tuple[str, dict]:
    """Apply the direct-unit R_sev/M_sev gate in the manuscript.

    Only uncorrected supervised-policy HDS peaks enter this one-time offline
    decision.  The KKT candidate's validation objective or acceptance is not
    used to choose the branch; those are optional ablation diagnostics.
    """
    peaks = np.maximum(np.asarray([row["nominal_peak"] for row in supervised_rows], dtype=float), 0.0)
    severe_rate = float(np.mean(peaks > cfg.gate_severe_magnitude))
    maximum = float(peaks.max())
    triggered_by_rate = severe_rate > cfg.gate_allowed_severe_rate
    triggered_by_peak = maximum > cfg.gate_severe_magnitude
    choice = "KKT-refined" if triggered_by_rate or triggered_by_peak else "supervised"
    return choice, {
        "source": "uncorrected supervised-policy event-located HDS audit only",
        "severe_magnitude_threshold": cfg.gate_severe_magnitude,
        "allowed_severe_rate": cfg.gate_allowed_severe_rate,
        "R_sev": severe_rate,
        "M_sev": maximum,
        "triggered_by_rate": triggered_by_rate,
        "triggered_by_peak": triggered_by_peak,
        "selected_branch": choice,
    }


def write_cohorts(output: Path, cfg: Config) -> None:
    rng = np.random.default_rng(20260772)
    validation = rng.uniform(-.1, .1, cfg.validation_points)
    test = rng.uniform(-.1, .1, cfg.test_points)
    np.savetxt(output / "validation_initial_conditions.csv", validation, fmt="%.17g")
    np.savetxt(output / "test_initial_conditions.csv", test, fmt="%.17g")


def evaluate_only(output: Path, test_reference: Path, validation_reference: Path, cfg: Config) -> None:
    """Evaluate already selected frozen policies without retraining or re-gating."""
    summary_path = output / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Run validation training before --evaluate-only.")
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    test = load_jiang_labels(test_reference)
    validation = load_jiang_labels(validation_reference)
    if len(test["p"]) != cfg.test_points:
        raise ValueError("The supplied test cohort does not contain 400 points.")
    all_rows: list[dict] = []
    for seed_report in report["seeds"]:
        seed = int(seed_report["seed"])
        checkpoint = torch.load(output / f"models_seed{seed}.pth", map_location="cpu", weights_only=False)
        supervised = PolicyValue(cfg.zoh_steps)
        supervised.load_state_dict(checkpoint["supervised"])
        mean_s, std_s = checkpoint["normalization"]["supervised"]
        supervised_validation = evaluate("supervised", supervised, float(mean_s), float(std_s), validation, cfg)
        chosen, gate = choose_branch(supervised_validation, cfg)
        model = PolicyValue(cfg.zoh_steps)
        model.load_state_dict(checkpoint[chosen])
        mean, std = checkpoint["normalization"][chosen]
        rows = evaluate(chosen, model, float(mean), float(std), test, cfg)
        for row in rows:
            row["seed"] = seed
        all_rows.extend(rows)
        seed_report["gate"] = gate
        seed_report["validation"]["supervised"] = summarise(supervised_validation)
        seed_report["test"] = summarise(rows)
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    report["reference_test"] = {"objective": stats(test["objective"]), "coldstart_seconds": stats(test["seconds"])}
    report["pooled_test"] = summarise(all_rows)
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def recompute_gate_only(output: Path, validation_reference: Path, cfg: Config) -> None:
    """Recompute the gate without needlessly repeating an unchanged test run."""
    summary_path = output / "summary.json"
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = load_jiang_labels(validation_reference)
    for seed_report in report["seeds"]:
        seed = int(seed_report["seed"])
        checkpoint = torch.load(output / f"models_seed{seed}.pth", map_location="cpu", weights_only=False)
        model = PolicyValue(cfg.zoh_steps)
        model.load_state_dict(checkpoint["supervised"])
        mean, std = checkpoint["normalization"]["supervised"]
        supervised_validation = evaluate("supervised", model, float(mean), float(std), validation, cfg)
        selected, gate = choose_branch(supervised_validation, cfg)
        previous = seed_report.get("test", None)
        if previous is not None and selected != seed_report["gate"].get("selected_branch"):
            raise RuntimeError("The corrected gate selected a different branch; rerun --evaluate-only for final testing.")
        seed_report["gate"] = gate
        seed_report["validation"]["supervised"] = summarise(supervised_validation)
    report["config"] = asdict(cfg)
    report["validation_and_test_protocol"] = (
        "A disjoint 60-point cohort applies the manuscript direct-unit R_sev/M_sev gate "
        "to the uncorrected supervised policy only. A disjoint 400-point cohort is used only for final testing."
    )
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"seeds": [{"seed": item["seed"], "gate": item["gate"]} for item in report["seeds"]]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-labels", type=Path)
    parser.add_argument("--validation-reference", type=Path)
    parser.add_argument("--test-reference", type=Path)
    parser.add_argument("--make-cohorts", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--recompute-gate-only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=[20260771, 20260772, 20260773])
    args = parser.parse_args()
    cfg = Config()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.make_cohorts:
        write_cohorts(args.output, cfg)
        return
    if args.evaluate_only:
        if args.test_reference is None or args.validation_reference is None:
            raise ValueError("--evaluate-only requires validation and test reference files.")
        evaluate_only(args.output, args.test_reference, args.validation_reference, cfg)
        return
    if args.recompute_gate_only:
        if args.validation_reference is None:
            raise ValueError("--recompute-gate-only requires --validation-reference.")
        recompute_gate_only(args.output, args.validation_reference, cfg)
        return
    if not (args.training_labels and args.validation_reference):
        raise ValueError("Training and validation MATLAB label files are required for training.")

    train = load_jiang_labels(args.training_labels)
    validation = load_jiang_labels(args.validation_reference)
    test = load_jiang_labels(args.test_reference) if args.test_reference else None
    if len(train["p"]) != cfg.train_points or len(validation["p"]) != cfg.validation_points:
        raise ValueError("The supplied train/validation cohorts do not match the declared 400/60 protocol.")
    if test is not None and len(test["p"]) != cfg.test_points:
        raise ValueError("The supplied test cohort does not match the declared 400-point protocol.")
    corrector = make_corrector(cfg)
    teacher_peaks = np.asarray([corrector.audit(initial(float(p)), u, cfg.dt) for p, u in zip(train["p"], train["controls"])])
    if np.max(teacher_peaks) > 1e-8:
        raise RuntimeError(f"A Jiang--Fu training label failed the independent HDS audit: max={teacher_peaks.max():.3e}")
    train["duals"], residuals = reconstruct_duals(train["p"], train["controls"], cfg)
    np.savez_compressed(args.output / "jiangfu_training_data.npz", **train, teacher_hds_peak=teacher_peaks,
                        reconstructed_kkt_residual=residuals)

    all_rows: list[dict] = []
    seed_reports: list[dict] = []
    for seed in args.seeds:
        supervised, mean_s, std_s = train_policy(train, cfg, seed)
        refined, mean_k, std_k = train_policy(train, cfg, seed, base=copy.deepcopy(supervised), kkt=True)
        supervised_validation = evaluate("supervised", supervised, mean_s, std_s, validation, cfg)
        kkt_validation = evaluate("KKT-refined", refined, mean_k, std_k, validation, cfg)
        chosen, gate = choose_branch(supervised_validation, cfg)
        model, mean, std = (supervised, mean_s, std_s) if chosen == "supervised" else (refined, mean_k, std_k)
        test_rows = [] if test is None else evaluate(chosen, model, mean, std, test, cfg)
        for row in test_rows:
            row["seed"] = seed
        all_rows.extend(test_rows)
        seed_report = {"seed": seed, "gate": gate, "validation": {"supervised": summarise(supervised_validation),
            "KKT-refined": summarise(kkt_validation)}}
        if test is not None:
            seed_report["test"] = summarise(test_rows)
        seed_reports.append(seed_report)
        torch.save({"supervised": supervised.state_dict(), "KKT-refined": refined.state_dict(),
                    "normalization": {"supervised": [mean_s, std_s], "KKT-refined": [mean_k, std_k]},
                    "config": asdict(cfg)}, args.output / f"models_seed{seed}.pth")

    if all_rows:
        with (args.output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader(); writer.writerows(all_rows)
    report = {
        "benchmark": "Jiang--Fu (2026) Example 2 (JCB), with x1(0) parameterized in [-0.1,0.1] and x2(0)=-1.",
        "teacher_and_baseline": "The authors' unmodified Algorithm 1 is used for all offline labels and every cold-start reference solve.",
        "config": asdict(cfg), "training_teacher_hds_peak_max": float(teacher_peaks.max()),
        "reconstructed_reduced_nlp_kkt_residual": stats(residuals), "validation_and_test_protocol":
            "A disjoint 60-point cohort applies the manuscript direct-unit R_sev/M_sev gate to the uncorrected supervised policy only. A disjoint 400-point cohort is used only for final testing.",
        "reference_test": None if test is None else {"objective": stats(test["objective"]), "coldstart_seconds": stats(test["seconds"])},
        "seeds": seed_reports, "pooled_test": None if test is None else summarise(all_rows),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
