"""Paired S-u versus Jiang-compatible KKT continuation for Penicillin CVP20.

The continuation uses the *same 20 upper-bound constraints* as the saved
Jiang--Fu Algorithm-1 NLP: on every ZOH interval it integrates
``g(x(t_k)) + integral phi(gdot(x(t)), 1e-3) dt``, where ``phi`` is the
paper's smooth upper approximation to ``max(gdot, 0)``.  Thus the 20
``fmincon`` nonlinear-constraint multipliers and bound multipliers are used
against their corresponding finite-dimensional constraint functions.

This is exploratory: the differentiable RK4 evaluation is regression-checked
against a high-accuracy integration of that same upper-bound ODE, but it is
not claimed to reproduce MATLAB ode45 bit-for-bit.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.adaptive_event_hds import AdaptiveEventHDSConfig, AdaptiveEventHDSCorrector  # noqa: E402
from kkt_collocation.run_penicillin_ablation import (  # noqa: E402
    DT as OLD_DT, UMAX, g, gdot, ode, terminal_product,
)

# 16 RK4 substeps (h=0.125) is outside the stable regime of this PFBF
# dynamics.  32 steps is the smallest tested value that reproduces the
# Jiang--Fu teacher terminal objective to the requested tolerance.
N, HORIZON, SUBSTEPS, SMOOTHING = 20, 40.0, 32, 1e-3
ALPHA, RHO, ANCHOR = 1e-2, 10.0, 1.0
SEED = 20260771


@dataclass(frozen=True)
class Config:
    supervised_epochs: int = 200
    continuation_epochs: int = 10
    supervised_lr: float = 1e-3
    continuation_lr: float = 1e-5
    kkt_weight: float = ALPHA
    augmented_penalty: float = RHO
    anchor_weight: float = ANCHOR
    rk4_substeps_per_zoh: int = SUBSTEPS
    lambda_candidates: int = 31


class Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.control = nn.Linear(128, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return UMAX * torch.sigmoid(self.control(self.body(x)))


def _mat_value(handle: h5py.File, value) -> np.ndarray:
    return np.asarray(handle[value]) if isinstance(value, h5py.Reference) else np.asarray(value)


def _load_rows(directory: Path, prefix: str) -> dict[str, np.ndarray]:
    rows: list[dict] = []
    for path in sorted(map(Path, glob.glob(str(directory / f"{prefix}_shard*.mat")))):
        with h5py.File(path, "r") as handle:
            record = handle["records"]
            count = record["index"].size
            fields = {name: record[name][...].ravel() for name in ("index", "success", "x2_0", "objective", "controls", "solve_seconds", "path_multipliers", "lower_bound_multipliers", "upper_bound_multipliers")}
            for i in range(count):
                get = lambda name: _mat_value(handle, fields[name][i]).squeeze()
                rows.append({
                    "index": int(get("index")), "success": bool(get("success")), "x2": float(get("x2_0")),
                    "objective": float(get("objective")), "controls": np.asarray(get("controls"), float).reshape(N),
                    "solve_seconds": float(get("solve_seconds")), "mu": np.asarray(get("path_multipliers"), float).reshape(N),
                    "lower": np.asarray(get("lower_bound_multipliers"), float).reshape(N),
                    "upper": np.asarray(get("upper_bound_multipliers"), float).reshape(N),
                })
    rows.sort(key=lambda row: row["index"])
    if len(rows) != 400 or [row["index"] for row in rows] != list(range(400)) or not all(row["success"] for row in rows):
        raise RuntimeError(f"Expected 400 successful ordered {prefix} Jiang--Fu rows")
    return {key: np.asarray([row[key] for row in rows]) for key in ("x2", "objective", "controls", "solve_seconds", "mu", "lower", "upper")}


def _rhs(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    x1, x2, x3, x4 = x.unbind(1)
    d1, d2 = .006 * x1 + x2, x2 + .0001 + x2.square() / .1
    return torch.stack((
        .11 * x1 * x2 / d1 - u * x1 / x4,
        -.11 * x1 * x2 / (.47 * d1) - .004 * x1 * x2 / (1.2 * d2) - .029 * x1 + u * (400. - x2) / x4,
        .004 * x1 * x2 / d2 - .01 * x3 - u * x3 / x4,
        u,
    ), dim=1)


def jiang_upper_bound_rollout(x2: torch.Tensor, controls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Objective and 20 Jiang--Fu smooth upper-bound constraints, g<=0."""
    batch = len(x2)
    x = torch.stack((torch.ones(batch, device=controls.device), x2, torch.full((batch,), .001, device=controls.device), torch.full((batch,), 250., device=controls.device)), dim=1)
    h = HORIZON / (N * SUBSTEPS)
    constraints = []
    for segment in range(N):
        # The authors reset this accumulator at every current upper-bound
        # interval: g(x(t_k)) + integral phi(dot g) dt.
        q = x[:, 1] - .5
        u = controls[:, segment]
        for _ in range(SUBSTEPS):
            def rhs_q(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                state_rhs = _rhs(state, u)
                q_rhs = .5 * (state_rhs[:, 1] + torch.sqrt(state_rhs[:, 1].square() + SMOOTHING ** 2))
                return state_rhs, q_rhs
            k1, l1 = rhs_q(x)
            k2, l2 = rhs_q(x + .5 * h * k1)
            k3, l3 = rhs_q(x + .5 * h * k2)
            k4, l4 = rhs_q(x + h * k3)
            x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            q = q + h * (l1 + 2 * l2 + 2 * l3 + l4) / 6
        constraints.append(q)
    return -x[:, 2], torch.stack(constraints, dim=1)


def jiang_kkt_loss(x2: torch.Tensor, controls: torch.Tensor, mu: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, rho: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    objective, constraints = jiang_upper_bound_rollout(x2, controls)
    # fmincon lower/upper multipliers correspond to -u<=0 and u-umax<=0.
    lagrangian = objective + (mu * constraints).sum(1) + (lower * (-controls)).sum(1) + (upper * (controls - UMAX)).sum(1)
    lagrangian = lagrangian + .5 * rho * torch.relu(constraints).square().sum(1)
    stationarity = torch.autograd.grad(lagrangian.sum(), controls, create_graph=True, retain_graph=True)[0]
    terms = {
        "stationarity": stationarity.square().mean(),
        "primal": torch.relu(constraints).square().mean(),
        "complementarity": (mu * constraints).square().mean(),
        "bound_complementarity": (lower * controls).square().mean() + (upper * (controls - UMAX)).square().mean(),
    }
    return sum(terms.values()), terms


def direct_terminal_objective(x2: float, controls: np.ndarray) -> float:
    """Terminal minimization objective for a network control, with no HDS audit.

    This is deliberately only state propagation: it does not inspect the path
    constraint, locate events, correct controls, or apply any fallback.
    """
    state = np.array([1.0, x2, .001, 250.0])
    for control in controls:
        solution = solve_ivp(ode, (0.0, HORIZON / N), state, args=(float(control),),
                             method="DOP853", rtol=1e-10, atol=1e-12)
        if not solution.success:
            raise RuntimeError(solution.message)
        state = solution.y[:, -1]
    return -float(state[2])


def set_seed() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train(train: dict[str, np.ndarray], device: torch.device, cfg: Config) -> tuple[dict[str, Policy], dict[str, dict], tuple[float, float]]:
    set_seed()
    x2 = torch.tensor(train["x2"], dtype=torch.float32, device=device)
    target = torch.tensor(train["controls"], dtype=torch.float32, device=device)
    mu = torch.tensor(train["mu"], dtype=torch.float32, device=device)
    lower = torch.tensor(train["lower"], dtype=torch.float32, device=device)
    upper = torch.tensor(train["upper"], dtype=torch.float32, device=device)
    mean, std = float(x2.mean()), float(x2.std().clamp_min(1e-8))
    feature = ((x2 - mean) / std)[:, None]
    prototype = Policy().to(device)
    initial_state = {key: value.detach().clone() for key, value in prototype.state_dict().items()}
    models, report = {}, {}
    for method in ("S-u", "S-u+Jiang-KKT"):
        model = Policy().to(device); model.load_state_dict(initial_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.supervised_lr)
        for epoch in range(cfg.supervised_epochs):
            prediction = model(feature)
            loss = nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        anchor = model(feature).detach()
        if method == "S-u":
            # Same total epoch budget: another ten pure control-supervised epochs.
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.continuation_lr)
            for _ in range(cfg.continuation_epochs):
                loss = nn.functional.mse_loss(model(feature), target)
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.continuation_lr)
            for epoch in range(cfg.continuation_epochs):
                controls = model(feature)
                base = nn.functional.mse_loss(controls, target) + cfg.anchor_weight * nn.functional.mse_loss(controls, anchor)
                raw_kkt, terms = jiang_kkt_loss(x2, controls, mu, lower, upper, cfg.augmented_penalty)
                # Scale only the magnitude, not the direction, so the fixed
                # alpha has a stable interpretation across continuation steps.
                loss = base + cfg.kkt_weight * raw_kkt / raw_kkt.detach().clamp_min(1.)
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                if epoch in (0, cfg.continuation_epochs - 1):
                    report.setdefault(method, {})[f"continuation_epoch_{epoch + 1}"] = {k: float(v.detach()) for k, v in terms.items()} | {"raw_total": float(raw_kkt.detach())}
        with torch.enable_grad():
            controls = model(feature)
            raw, terms = jiang_kkt_loss(x2, controls, mu, lower, upper, cfg.augmented_penalty)
        models[method] = model.eval()
        report.setdefault(method, {}).update({"control_mse": float(nn.functional.mse_loss(controls, target).detach()), "jiang_kkt_raw": float(raw.detach()), "jiang_kkt_terms": {k: float(v.detach()) for k, v in terms.items()}})
    # Regression diagnostic: teacher controls evaluated by the compatible
    # differentiable upper-bound constraints and their own fmincon multipliers.
    teacher_controls = target.detach().clone().requires_grad_(True)
    teacher_raw, teacher_terms = jiang_kkt_loss(x2, teacher_controls, mu, lower, upper, cfg.augmented_penalty)
    report["teacher_compatible_kkt_check"] = {"raw_total": float(teacher_raw.detach()), **{k: float(v.detach()) for k, v in teacher_terms.items()}}
    return models, report, (mean, std)


def evaluate_direct(models: dict[str, Policy], test: dict[str, np.ndarray], mean: float, std: float, device: torch.device) -> tuple[list[dict], dict]:
    """Compare direct network rollouts with matching Jiang cold-start NLP objectives."""
    features = torch.tensor(((test["x2"] - mean) / std)[:, None], dtype=torch.float32, device=device)
    rows: list[dict] = []
    for name, model in models.items():
        started = time.perf_counter()
        with torch.no_grad(): controls = model(features).cpu().numpy()
        inference = (time.perf_counter() - started) / len(controls)
        for index, (x2, nominal, reference, reference_seconds) in enumerate(zip(test["x2"], controls, test["objective"], test["solve_seconds"])):
            started = time.perf_counter(); network_objective = direct_terminal_objective(float(x2), nominal); rollout_seconds = time.perf_counter() - started
            denom = max(abs(float(reference)), 1e-12)
            rows.append({"method": name, "index": index, "x2_0": float(x2),
                         "jiang_cvp20_reference_objective": float(reference), "jiang_cvp20_reference_solve_seconds": float(reference_seconds),
                         "network_objective": network_objective,
                         "relative_objective_gap_percent": 100 * (network_objective - reference) / denom,
                         "inference_seconds": inference, "rollout_seconds": rollout_seconds,
                         "total_deployment_seconds": inference + rollout_seconds})
    summary: dict[str, dict] = {}
    for name in models:
        group = [row for row in rows if row["method"] == name]
        val = lambda key: np.asarray([row[key] for row in group], float)
        summary[name] = {"samples": len(group), "direct_rollout_gap_percent": float(np.mean(val("relative_objective_gap_percent"))),
                         "mean_inference_seconds": float(np.mean(val("inference_seconds"))), "mean_rollout_seconds": float(np.mean(val("rollout_seconds"))),
                         "mean_total_seconds": float(np.mean(val("total_deployment_seconds"))),
                         "mean_cold_reference_seconds": float(np.mean(val("jiang_cvp20_reference_solve_seconds"))),
                         "mean_speedup": float(np.mean(val("jiang_cvp20_reference_solve_seconds") / val("total_deployment_seconds")))}
    return rows, summary


def evaluate_hds(models: dict[str, Policy], test: dict[str, np.ndarray], mean: float, std: float, device: torch.device, cfg: Config) -> tuple[list[dict], dict]:
    """Audit and, if needed, lambda-correct network controls; references stay untouched."""
    corrector = AdaptiveEventHDSCorrector(
        ode, g, gdot, (0., UMAX), AdaptiveEventHDSConfig(grid_size=cfg.lambda_candidates)
    )
    features = torch.tensor(((test["x2"] - mean) / std)[:, None], dtype=torch.float32, device=device)
    rows: list[dict] = []
    for name, model in models.items():
        started = time.perf_counter()
        with torch.no_grad():
            controls = model(features).cpu().numpy()
        inference = (time.perf_counter() - started) / len(controls)
        for index, (x2, nominal, reference, reference_seconds) in enumerate(zip(test["x2"], controls, test["objective"], test["solve_seconds"])):
            initial = np.array([1., x2, .001, 250.])
            nominal_g = float(corrector.audit(initial, nominal, HORIZON / N))
            started = time.perf_counter()
            result = corrector.correct(initial, nominal, HORIZON / N)
            hds_seconds = time.perf_counter() - started
            accepted = bool(result.accepted)
            applied = result.controls if accepted else nominal
            applied_g = float(corrector.audit(initial, applied, HORIZON / N)) if accepted else np.nan
            applied_objective = -terminal_product(float(x2), applied, corrector) if accepted else np.nan
            denom = max(abs(float(reference)), 1e-12)
            rows.append({
                "method": name, "index": index, "x2_0": float(x2),
                "accepted": accepted, "fallback": not accepted,
                "nominal_hds_max_g": nominal_g, "applied_hds_max_g": applied_g,
                "jiang_cvp20_reference_objective": float(reference),
                "jiang_cvp20_reference_solve_seconds": float(reference_seconds),
                "applied_objective": applied_objective,
                "hds_relative_objective_gap_percent": 100 * (applied_objective - reference) / denom if accepted else np.nan,
                "corrected_segments": int(sum(item.corrected for item in result.segments)),
                "inference_seconds": inference, "hds_correction_seconds": hds_seconds,
                "total_deployment_seconds": inference + hds_seconds,
            })
    summary: dict[str, dict] = {}
    for name in models:
        group = [row for row in rows if row["method"] == name]
        val = lambda key: np.asarray([row[key] for row in group], float)
        summary[name] = {
            "samples": len(group), "hds_gap_percent": float(np.nanmean(val("hds_relative_objective_gap_percent"))),
            "nominal_violation_rate": float(np.mean(val("nominal_hds_max_g") > 1e-8)),
            "acceptance_rate": float(np.mean(val("accepted"))), "fallback_rate": float(np.mean(val("fallback"))),
            "final_max_g": float(np.nanmax(val("applied_hds_max_g"))),
            "mean_corrected_segments": float(np.mean(val("corrected_segments"))),
            "mean_inference_seconds": float(np.mean(val("inference_seconds"))),
            "mean_hds_seconds": float(np.mean(val("hds_correction_seconds"))),
            "mean_total_seconds": float(np.mean(val("total_deployment_seconds"))),
            "mean_cold_reference_seconds": float(np.mean(val("jiang_cvp20_reference_solve_seconds"))),
            "mean_speedup": float(np.mean(val("jiang_cvp20_reference_solve_seconds") / val("total_deployment_seconds"))),
        }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "kkt_collocation/results/exploratory_penicillin_cvp20_jiang_seed20260771_v1/raw_jiang_fu_cvp20")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hds", action="store_true", help="Audit and correct network controls before objective comparison.")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    cfg, device = Config(), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, test_data = _load_rows(args.data_dir, "train"), _load_rows(args.data_dir, "test")
    models, training, norm = train(train_data, device, cfg)
    rows, deployment = (evaluate_hds(models, test_data, *norm, device, cfg) if args.hds
                        else evaluate_direct(models, test_data, *norm, device))
    for name, model in models.items(): torch.save({"model": model.state_dict(), "normalization": {"mean": norm[0], "std": norm[1]}, "config": asdict(cfg), "method": name}, args.output / f"{name}.pth")
    with (args.output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = {"formal_protocol": False, "benchmark": "Penicillin CVP20 Jiang--Fu", "seed": SEED, "config": asdict(cfg),
              "label_and_reference_source": str(args.data_dir), "reference_audit_performed": False,
              "network_evaluation": ("DOP853 stationary-event HDS with the frozen 31-point lambda correction grid; references are not audited."
                                     if args.hds else
                                     "Direct DOP853 terminal-state rollout only; no path-constraint audit, HDS correction, acceptance check, or fallback."),
              "kkt_definition": "fmincon finite-dimensional multipliers of the same 20 Jiang--Fu smooth upper-bound constraints and bound constraints; not continuous-time multipliers.",
              "training": training, "deployment": deployment,
              "hds_statement": ("Continuous-time numerical audit evidence under the declared model and numerical settings; not a real-system absolute safety guarantee."
                                if args.hds else None)}
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
