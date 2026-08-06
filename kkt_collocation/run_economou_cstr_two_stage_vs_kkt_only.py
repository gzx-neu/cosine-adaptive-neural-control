"""Formal single-seed Economou CSTR S / K-only / S+K study on RK10 labels.

This driver is deliberately independent from the older preliminary CSTR
screen.  Its loss is the *same reduced RK4 transcription* used to make the
labels: controls are the only primal variables, 100 post-step path nodes are
used, and finite-dimensional lower/upper box multipliers are retained.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing as mp
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kkt_collocation.economou_cstr_hds_fast import fast_correct, segment_event_audit
from kkt_collocation.generate_economou_cstr_reduced_kkt_data import ReducedEconomouCSTR
from kkt_collocation.run_economou_cstr_supervised_hds import PolicyValue, lhs_states, trajectory_objective
from kkt_collocation.screen_economou_cstr_30x30 import EconomouScreenConfig


@dataclass(frozen=True)
class Config:
    seed: int = 20260771
    supervised_epochs: int = 200
    continuation_epochs: int = 20
    supervised_lr: float = 1e-3
    continuation_lr: float = 1e-5
    # Frozen before running this RK10 experiment; same normalized KKT loss in
    # K-only and S+K, matching the established penicillin continuation scale.
    kkt_weight: float = 1e-2
    augmented_penalty: float = 10.0
    anchor_weight: float = 1.0
    rollout_consistency_weight: float = 0.0
    lambda_grid_size: int = 31
    validation_count: int = 60
    test_count: int = 100
    hds_workers: int = 4

    @property
    def total_epochs(self) -> int:
        return self.supervised_epochs + self.continuation_epochs


def json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def dump_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_labels(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["index"]))
    if len(rows) != 900 or not all(row.get("success") for row in rows):
        raise ValueError("Expected exactly 900 successful RK10 reduced-space labels")
    states = np.asarray([row["initial_state"] for row in rows], dtype=np.float64)
    controls = np.asarray([row["controls"] for row in rows], dtype=np.float64)
    objectives = np.asarray([row["objective"] for row in rows], dtype=np.float64)
    path = np.asarray([row["path_duals"] for row in rows], dtype=np.float64).reshape(len(rows), -1)
    bounds = np.asarray([row["bound_duals"] for row in rows], dtype=np.float64)
    self_check = np.asarray([row["kkt_stationarity_norm"] for row in rows], dtype=np.float64)
    if (controls.ndim != 3 or controls.shape[0] != 900 or controls.shape[2] != 2
            or path.shape != (900, controls.shape[1] * 2 * 10)
            or bounds.shape != (900, 2, controls.shape[1] * 2)):
        raise ValueError(f"Unexpected RK10 label shapes: {controls.shape}, {path.shape}, {bounds.shape}")
    return states, controls, objectives, path, bounds, self_check


def rollout_flat(initial: torch.Tensor, flat: torch.Tensor, cfg: EconomouScreenConfig):
    """Exact reduced RK10 transcription, including pre-update stage cost."""
    controls = flat.reshape(-1, cfg.zoh_steps, 2)
    state = initial
    cost = torch.zeros(len(initial), dtype=initial.dtype, device=initial.device)
    g = []
    h = cfg.dt
    for j in range(cfg.zoh_steps * cfg.substeps_per_zoh):
        u = controls[:, j // cfg.substeps_per_zoh, :]
        # This timing is identical to ReducedEconomouCSTR._build().
        cost = cost + h * (-u[:, 1] - 2.009 * state[:, 1] + (1.657e-3 * u[:, 0]) ** 2) / cfg.horizon_s

        def rhs(x):
            ca, cb, temp = x[:, 0], x[:, 1], x[:, 2]
            ti, flow = u[:, 0], u[:, 1]
            r1 = cfg.c1_s_inv * torch.exp(-cfg.e1_cal_mol / (cfg.gas_constant_cal_mol_K * temp))
            r2 = cfg.c2_s_inv * torch.exp(-cfg.e2_cal_mol / (cfg.gas_constant_cal_mol_K * temp))
            rate = r1 * ca - r2 * cb
            dilution = flow / cfg.residence_time_s
            heat = cfg.minus_delta_h_cal_mol / (cfg.density_kg_L * cfg.heat_capacity_cal_kg_K)
            return torch.stack((dilution * (cfg.ca_feed - ca) - rate,
                                dilution * (cfg.cb_feed - cb) + rate,
                                dilution * (ti - temp) + heat * rate), dim=1)

        k1 = rhs(state); k2 = rhs(state + .5 * h * k1)
        k3 = rhs(state + .5 * h * k2); k4 = rhs(state + h * k3)
        state = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        g.append(torch.stack((state[:, 0] - (cfg.ca_max - cfg.node_margin),
                              state[:, 2] - (cfg.temperature_max_K - cfg.node_margin)), dim=1))
    return cost, torch.cat(g, dim=1)


def kkt_terms(initial, flat, path_duals, bound_duals, cfg, rho: float):
    """KKT residual of the exact finite-dimensional reduced NLP.

    The multiplier labels are deliberately described as finite-dimensional
    quantities of the discretized transcription, never as continuous-time
    multipliers.
    """
    objective, g = rollout_flat(initial, flat, cfg)
    low = torch.tensor([cfg.ti_bounds_K[0], cfg.flow_bounds[0]], dtype=flat.dtype, device=flat.device).repeat(cfg.zoh_steps)
    high = torch.tensor([cfg.ti_bounds_K[1], cfg.flow_bounds[1]], dtype=flat.dtype, device=flat.device).repeat(cfg.zoh_steps)
    mu_lo, mu_hi = bound_duals[:, 0], bound_duals[:, 1]
    violation = torch.relu(g)
    lagrangian = (objective + (path_duals * g).sum(1)
                  + (mu_lo * (low - flat)).sum(1) + (mu_hi * (flat - high)).sum(1)
                  + .5 * rho * violation.square().sum(1))
    grad = torch.autograd.grad(lagrangian.sum(), flat, create_graph=True)[0]
    stationarity = grad.square().mean()
    primal = violation.square().mean()
    comp_path = (path_duals * g).square().mean()
    comp_bounds = ((mu_lo * (low - flat)).square().mean()
                   + (mu_hi * (flat - high)).square().mean())
    total = stationarity + primal + comp_path + comp_bounds
    return {"total": total, "stationarity": stationarity, "primal": primal,
            "complementarity_path": comp_path, "complementarity_bounds": comp_bounds,
            "objective": objective, "path_g": g}


def _mean(values):
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    return float(valid.mean()) if len(valid) else None


def _std(values):
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    return float(valid.std(ddof=0)) if len(valid) else None


def _reference_worker(payload):
    idx, state, config = payload
    cfg = EconomouScreenConfig(**config)
    started = time.perf_counter()
    try:
        solver = ReducedEconomouCSTR(cfg)
        try:
            result = solver.solve(np.asarray(state, dtype=float))
            attempt = "fixed_[400K,0.20]"
        except RuntimeError as primary_error:
            # Same independent, fixed fallback protocol as the RK10 label
            # generator; never a neural or neighbouring-label warm start.
            result = solver.solve(np.asarray(state, dtype=float),
                                  np.tile(np.array([420.0, 1.0]), cfg.zoh_steps))
            attempt = "fixed_[420K,1.0]_after_primary_failure"
        audit = fast_correct(np.asarray(state, dtype=float), np.asarray(result["controls"], dtype=float), cfg, grid=31)
        high_objective = trajectory_objective(np.asarray(state, dtype=float), np.asarray(result["controls"], dtype=float), cfg)
        return {"index": idx, "initial_state": state, "solver_success": True,
                "solve_seconds": result["solve_seconds"], "worker_seconds": time.perf_counter() - started,
                "discrete_objective": result["objective"], "high_fidelity_objective": high_objective,
                "audit_accepted": bool(audit["accepted"]), "audit_max_g": audit["final_peak"],
                "kkt_stationarity_norm": result["kkt_stationarity_norm"], "cold_start_attempt": attempt}
    except Exception as exc:
        return {"index": idx, "initial_state": state, "solver_success": False,
                "solve_seconds": time.perf_counter() - started, "worker_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}", "audit_accepted": False}


def _audit_worker(payload):
    idx, state, controls, config, grid = payload
    cfg = EconomouScreenConfig(**config)
    started = time.perf_counter()
    try:
        state0, raw_controls = np.asarray(state, float), np.asarray(controls, float)
        nominal = trajectory_objective(state0, raw_controls, cfg)
        # Nominal metrics are for the unmodified network sequence, not the
        # sequentially corrected intermediate state used inside the corrector.
        nominal_state, nominal_peak = state0.copy(), -np.inf
        for control in raw_controls:
            peak, nominal_state, _ = segment_event_audit(nominal_state, control, cfg)
            nominal_peak = max(nominal_peak, float(np.max(peak)))
        corrected = fast_correct(state0, raw_controls, cfg, grid=grid)
        applied = trajectory_objective(np.asarray(state, float), corrected["controls"], cfg) if corrected["accepted"] else np.nan
        return {"index": idx, "initial_state": state, "accepted": bool(corrected["accepted"]),
                "fallback": not bool(corrected["accepted"]), "nominal_max_g": nominal_peak,
                "final_max_g": corrected["final_peak"], "nominal_objective": nominal,
                "hds_objective": applied, "corrected_segments": int(corrected["corrected_segments"]),
                "hds_seconds": time.perf_counter() - started}
    except Exception as exc:
        return {"index": idx, "initial_state": state, "accepted": False, "fallback": True,
                "error": f"{type(exc).__name__}: {exc}", "hds_seconds": time.perf_counter() - started}


def evaluate_method(name, model, mean, std, test, cstr, cfg, output, references):
    device = next(model.parameters()).device
    inputs = torch.tensor((test[:, [0, 2]] - mean) / std, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(inputs)
        started = time.perf_counter()
        _, predicted = model(inputs)
        inference_per_sample = (time.perf_counter() - started) / len(test)
    controls = predicted.cpu().numpy()
    context = mp.get_context("spawn")
    payload = [(i, test[i].tolist(), controls[i].tolist(), asdict(cstr), cfg.lambda_grid_size)
               for i in range(len(test))]
    with context.Pool(cfg.hds_workers) as pool:
        rows = list(pool.imap_unordered(_audit_worker, payload, chunksize=1))
    rows.sort(key=lambda row: row["index"])
    reference_by_index = {int(row["index"]): row for row in references}
    for row in rows:
        ref = reference_by_index[row["index"]]
        row["reference_available"] = bool(ref.get("solver_success") and ref.get("audit_accepted"))
        row["reference_objective"] = ref.get("high_fidelity_objective")
        row["relative_nominal_gap_percent"] = (100 * (row["nominal_objective"] - ref["high_fidelity_objective"])
                                               / max(abs(ref["high_fidelity_objective"]), 1e-12)
                                               if row["reference_available"] else np.nan)
        row["relative_hds_gap_percent"] = (100 * (row["hds_objective"] - ref["high_fidelity_objective"])
                                           / max(abs(ref["high_fidelity_objective"]), 1e-12)
                                           if row["reference_available"] and row["accepted"] else np.nan)
        row["inference_seconds"] = inference_per_sample
        row["total_predeployment_seconds"] = inference_per_sample + row["hds_seconds"]
    fields = sorted({key for row in rows for key in row})
    with (output / f"{name}_test_sample_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    accepted = np.asarray([row["accepted"] for row in rows], bool)
    hds_gap = [row.get("relative_hds_gap_percent", np.nan) for row in rows]
    summary = {"accepted_network_samples": int(accepted.sum()), "fallback_samples": int((~accepted).sum()),
               "nominal_violation_rate_percent": float(100 * np.mean(np.asarray([r.get("nominal_max_g", np.nan) for r in rows]) > 1e-8)),
               "nominal_max_g": float(np.nanmax([r.get("nominal_max_g", np.nan) for r in rows])),
               "final_max_g": _mean([r.get("final_max_g", np.nan) for r in rows]),
               "mean_nominal_relative_gap_percent": _mean([r.get("relative_nominal_gap_percent", np.nan) for r in rows]),
               "mean_hds_relative_gap_percent": _mean(hds_gap), "std_hds_relative_gap_percent": _std(hds_gap),
               "median_hds_relative_gap_percent": float(np.nanmedian(hds_gap)) if np.isfinite(hds_gap).any() else None,
               "p95_hds_relative_gap_percent": float(np.nanpercentile(hds_gap, 95)) if np.isfinite(hds_gap).any() else None,
               "max_hds_relative_gap_percent": float(np.nanmax(hds_gap)) if np.isfinite(hds_gap).any() else None,
               "mean_corrected_segments": _mean([r.get("corrected_segments", np.nan) for r in rows]),
               "mean_hds_objective_change": _mean([r.get("hds_objective", np.nan) - r.get("nominal_objective", np.nan) for r in rows]),
               "mean_inference_seconds": inference_per_sample, "mean_hds_seconds": _mean([r["hds_seconds"] for r in rows]),
               "mean_total_predeployment_seconds": _mean([r["total_predeployment_seconds"] for r in rows])}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "kkt_collocation/results/economou_cstr_two_stage_vs_kkt_only_rk10_single")
    parser.add_argument("--smoke", action="store_true", help="Only run label self-check and 2 epochs per branch.")
    parser.add_argument("--labels", type=Path,
                        default=ROOT / "kkt_collocation/results/economou_cstr_reduced_kkt10_30x30/records.jsonl")
    parser.add_argument("--methods", nargs="+", choices=("S", "K-only", "S+K"),
                        default=("S", "K-only", "S+K"),
                        help="Training branches to execute; selected branches retain their standard total-update budget.")
    parser.add_argument("--label-grid-holdout", action="store_true",
                        help="Use uniformly sampled held-out label-grid states and objectives instead of cold-start NLP references.")
    parser.add_argument("--supervised-epochs", type=int, default=200)
    parser.add_argument("--continuation-epochs", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing independent result directory: {args.output}")
    args.output.mkdir(parents=True)
    if args.smoke:
        cfg = Config(supervised_epochs=2, continuation_epochs=1, validation_count=4, test_count=4, hds_workers=2)
    else:
        if args.supervised_epochs < 1 or args.continuation_epochs < 1:
            raise ValueError("Both training-stage lengths must be positive")
        cfg = Config(supervised_epochs=args.supervised_epochs, continuation_epochs=args.continuation_epochs)
    set_seed(cfg.seed)
    labels = args.labels.resolve()
    label_summary = json.loads((labels.parent / "summary.json").read_text(encoding="utf-8"))
    label_config = dict(label_summary["config"])
    # JSON turns declared tuple fields into lists; retain their exact values.
    for field in ("ti_bounds_K", "flow_bounds", "ca_initial_range", "temperature_initial_range_K"):
        label_config[field] = tuple(label_config[field])
    cstr = EconomouScreenConfig(**label_config)
    if cstr.substeps_per_zoh != 10:
        raise ValueError("This exact RK10 training driver requires RK10 labels")
    states_all, u_ref_all, j_ref_all, path_duals_all, bound_duals_all, recorded_self_check_all = load_labels(labels)
    label_references = None
    if args.label_grid_holdout:
        # Ten equally spaced coordinates on each axis give a fixed 10x10 test
        # lattice. The validation holdout is evenly spaced through the remaining
        # label ordering; neither holdout participates in training.
        grid = np.rint(np.linspace(0, 29, 10)).astype(int)
        test_indices = np.asarray([30 * i + j for i in grid for j in grid], dtype=int)
        remaining = np.setdiff1d(np.arange(900), test_indices, assume_unique=True)
        validation_indices = remaining[np.rint(np.linspace(0, len(remaining) - 1, cfg.validation_count)).astype(int)]
        train_indices = np.setdiff1d(remaining, validation_indices, assume_unique=True)
        validation, test = states_all[validation_indices], states_all[test_indices]
        label_references = [{"index": int(i), "initial_state": test[i].tolist(), "solver_success": True,
                             "audit_accepted": True, "high_fidelity_objective": float(j_ref_all[test_indices[i]]),
                             "discrete_label_objective": float(j_ref_all[test_indices[i]]),
                             "source": "uniform_heldout_discrete_label"}
                            for i in range(len(test))]
        states, u_ref, j_ref = states_all[train_indices], u_ref_all[train_indices], j_ref_all[train_indices]
        path_duals, bound_duals, recorded_self_check = (path_duals_all[train_indices], bound_duals_all[train_indices],
                                                         recorded_self_check_all[train_indices])
        split_meta = {"mode": "uniform_heldout_label_grid", "train_count": int(len(train_indices)),
                      "validation_count": int(len(validation_indices)), "test_count": int(len(test_indices)),
                      "test_label_indices": test_indices.tolist(), "validation_label_indices": validation_indices.tolist()}
    else:
        states, u_ref, j_ref = states_all, u_ref_all, j_ref_all
        path_duals, bound_duals, recorded_self_check = path_duals_all, bound_duals_all, recorded_self_check_all
        validation, test = lhs_states(cstr, cfg.validation_count, cfg.seed + 1), lhs_states(cstr, cfg.test_count, cfg.seed + 2)
        split_meta = {"mode": "independent_lhs_cold_reference", "train_count": int(len(states))}
    np.save(args.output / "validation_initial_conditions.npy", validation)
    np.save(args.output / "test_initial_conditions.npy", test)

    # Exact teacher self-check in float64.  A material mismatch is a hard stop.
    x64 = torch.tensor(states, dtype=torch.float64)
    flat64 = torch.tensor(u_ref.reshape(len(u_ref), -1), dtype=torch.float64, requires_grad=True)
    d64, b64 = torch.tensor(path_duals, dtype=torch.float64), torch.tensor(bound_duals, dtype=torch.float64)
    teacher = kkt_terms(x64, flat64, d64, b64, cstr, cfg.augmented_penalty)
    torch_norm_estimate = float(torch.sqrt(teacher["stationarity"]).detach())
    if not np.isfinite(torch_norm_estimate) or torch_norm_estimate > 1e-3:
        raise RuntimeError(f"Exact RK10 teacher KKT self-check failed: RMS stationarity={torch_norm_estimate:.3e}")
    teacher_summary = {"recorded_stationarity_norm_mean": float(recorded_self_check.mean()),
                       "recorded_stationarity_norm_max": float(recorded_self_check.max()),
                       "torch_rms_stationarity": torch_norm_estimate,
                       "torch_total_kkt_residual": float(teacher["total"].detach()),
                       "formulation": ("exact reduced RK4 transcription; pre-update stage cost; "
                                       f"{2 * cstr.zoh_steps * cstr.substeps_per_zoh} path and "
                                       f"{4 * cstr.zoh_steps} bound multipliers")}
    dump_json(args.output / "teacher_kkt_self_check.json", teacher_summary)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(states, dtype=torch.float32, device=device)
    ur = torch.tensor(u_ref, dtype=torch.float32, device=device)
    jr = torch.tensor(j_ref[:, None], dtype=torch.float32, device=device)
    dr = torch.tensor(path_duals, dtype=torch.float32, device=device)
    br = torch.tensor(bound_duals, dtype=torch.float32, device=device)
    mean, std = states[:, [0, 2]].mean(0), states[:, [0, 2]].std(0).clip(1e-6)
    jmean, jstd = jr.mean(), jr.std().clamp_min(1e-6)
    normalized_x = torch.tensor((states[:, [0, 2]] - mean) / std, dtype=torch.float32, device=device)
    low = torch.tensor([cstr.ti_bounds_K[0], cstr.flow_bounds[0]], dtype=torch.float32, device=device)
    span = torch.tensor([70.0, 1.0], dtype=torch.float32, device=device)

    def terms(model, need_kkt, anchor=None):
        predicted_j, predicted_u = model(normalized_x)
        control_mse = nn.functional.mse_loss((predicted_u - low) / span, (ur - low) / span)
        objective_mse = nn.functional.mse_loss(predicted_j, (jr - jmean) / jstd)
        result = {"control_mse": control_mse, "objective_mse": objective_mse,
                  "supervised": control_mse + .1 * objective_mse}
        if need_kkt:
            flat = predicted_u.reshape(len(predicted_u), -1)
            result.update({f"kkt_{key}": value for key, value in kkt_terms(x, flat, dr, br, cstr, cfg.augmented_penalty).items()
                           if key not in ("objective", "path_g")})
        if anchor is not None:
            result["anchor"] = nn.functional.mse_loss((predicted_u - low) / span, (anchor - low) / span)
        return result

    def train(method):
        set_seed(cfg.seed)
        model = PolicyValue(cstr).to(device)
        initial_state = copy.deepcopy(model.state_dict())
        # Same seed and explicit state capture prove each branch starts identically.
        model.load_state_dict(initial_state)
        history, failure = [], None
        started = time.perf_counter()
        stages = [("supervised", cfg.total_epochs, False, None, cfg.supervised_lr)] if method == "S" else []
        if method == "K-only":
            stages.append(("kkt_only", cfg.total_epochs, True, None, cfg.supervised_lr))
        if method == "S+K":
            stages.append(("supervised", cfg.supervised_epochs, False, None, cfg.supervised_lr))
        for stage, epochs, need_kkt, anchor, lr in stages:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            for epoch in range(1, epochs + 1):
                try:
                    record = terms(model, need_kkt, anchor)
                    if method == "K-only":
                        loss = record["kkt_total"] / record["kkt_total"].detach().clamp_min(1.0)
                    else:
                        loss = record["supervised"]
                        if need_kkt:
                            loss = loss + cfg.kkt_weight * record["kkt_total"] / record["kkt_total"].detach().clamp_min(1.0)
                        if anchor is not None:
                            loss = loss + cfg.anchor_weight * record["anchor"]
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite loss")
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                    history.append({"stage": stage, "epoch": epoch, "loss": float(loss.detach()),
                                    **{key: float(value.detach()) for key, value in record.items()}})
                    if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                        print(f"{method} {stage} {epoch}/{epochs} loss={float(loss.detach()):.3e}", flush=True)
                except (RuntimeError, FloatingPointError) as exc:
                    failure = f"{type(exc).__name__}: {exc}"; break
            if failure:
                break
            if method == "S+K" and stage == "supervised":
                with torch.no_grad():
                    _, frozen_anchor = model(normalized_x)
                stages.append(("continuation", cfg.continuation_epochs, True, frozen_anchor.detach(), cfg.continuation_lr))
        with torch.enable_grad():
            final = terms(model, True)
        training = {"completed": failure is None, "failure": failure, "seconds": time.perf_counter() - started,
                    "control_mse_normalized": float(final["control_mse"].detach()),
                    "objective_mse_normalized": float(final["objective_mse"].detach()),
                    "kkt_residual": float(final["kkt_total"].detach()),
                    "kkt_stationarity": float(final["kkt_stationarity"].detach()),
                    "kkt_primal": float(final["kkt_primal"].detach()),
                    "kkt_complementarity_path": float(final["kkt_complementarity_path"].detach()),
                    "kkt_complementarity_bounds": float(final["kkt_complementarity_bounds"].detach())}
        torch.save({"model": model.state_dict(), "state_mean": mean, "state_std": std, "config": asdict(cfg), "training": training}, args.output / f"{method}.pth")
        dump_json(args.output / f"{method}_training_log.json", {"training": training, "history": history})
        return model, training

    # Either reuse the fixed, held-out teacher-label objective for a fast
    # screening ablation or generate the formal independent cold-start set.
    if label_references is not None:
        references = label_references
    else:
        context = mp.get_context("spawn")
        ref_payload = [(i, test[i].tolist(), asdict(cstr)) for i in range(len(test))]
        with context.Pool(cfg.hds_workers) as pool:
            references = list(pool.imap_unordered(_reference_worker, ref_payload, chunksize=1))
        references.sort(key=lambda row: row["index"])
    reference_filename = "heldout_label_references.json" if label_references is not None else "cold_start_references.json"
    with (args.output / reference_filename).open("w", encoding="utf-8") as handle:
        json.dump(references, handle, indent=2, default=json_default)

    methods = {}
    for method in args.methods:
        print(f"Training {method}", flush=True)
        model, training = train(method)
        methods[method] = {"training": training, "deployment": evaluate_method(method, model, mean, std, test, cstr, cfg, args.output, references)}
        dump_json(args.output / "summary.json", {"status": "running", "config": asdict(cfg), "methods": methods})

    ref_ok = [row for row in references if row.get("solver_success") and row.get("audit_accepted")]
    ref_summary = ({"count": len(references), "source": "uniform held-out discrete teacher labels",
                    "mean_cold_start_seconds": None,
                    "note": "Objective gaps are screening comparisons to held-out discrete-label objectives, not cold-start NLP gaps."}
                   if label_references is not None else
                   {"count": len(references), "successful_and_audited": len(ref_ok),
                    "mean_cold_start_seconds": _mean([row.get("solve_seconds", np.nan) for row in ref_ok]),
                    "cold_start": "fixed [T_i,F]=[400 K,0.20] per test initial condition; no warm start"})
    table = ["| Method | Nominal gap (%) | HDS-corrected gap (%) | Nominal violation | Corrected segments | Accepted / fallback | KKT residual | Training stable |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for method, result in methods.items():
        d, t = result["deployment"], result["training"]
        table.append(f"| {method} | {d['mean_nominal_relative_gap_percent']!s} | {d['mean_hds_relative_gap_percent']!s} | {d['nominal_violation_rate_percent']:.1f}% | {d['mean_corrected_segments']!s} | {d['accepted_network_samples']} / {d['fallback_samples']} | {t['kkt_residual']:.3e} | {t['completed']} |")
    (args.output / "summary_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    final = {"status": "completed", "config": asdict(cfg), "label_source": str(labels),
             "label_protocol": "900 independent cold-start RK10 reduced-space labels; finite-dimensional discretized-NLP multipliers only",
             "teacher_kkt_self_check": teacher_summary, "split": split_meta, "reference": ref_summary, "methods": methods,
             "hds_statement": "Continuous-time numerical audit evidence under the declared model and numerical settings; not a real-system absolute safety guarantee."}
    dump_json(args.output / "summary.json", final)
    print(json.dumps(final, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
