"""Evaluate trained policies just outside the declared operating domain.

This is a robustness diagnostic, not a deployment path.  Production use first
applies :func:`assess_initial_state`; all points created here are deliberately
outside the declared domain and would therefore be sent directly to the
offline optimizer.  The script additionally bypasses that guard to quantify
how often an extrapolated policy can still be repaired and HDS-audited.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control import BoxOperatingDomain, assess_initial_state  # noqa: E402
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector  # noqa: E402
from run_penicillin_ablation import (  # noqa: E402
    DT as PEN_DT, UMAX as PEN_UMAX, Policy as PenicillinPolicy, g as pen_g,
    gdot as pen_gdot, ode as pen_ode, terminal_product,
)
from run_vdp_ablation import constraint as vdp_g, constraint_derivative as vdp_gdot, terminal_cost, vdp_ode  # noqa: E402
from kkt_collocation.train_vdp_kkt_policy import KKTPolicyValueNetwork, TrainConfig  # noqa: E402


VDP_DOMAIN = BoxOperatingDomain([-0.1, 0.9], [0.1, 1.1], state_indices=(0, 1), name="VDP operating domain")
PEN_DOMAIN = BoxOperatingDomain([0.1], [0.3], state_indices=(1,), name="penicillin operating domain")


def _load_checkpoint(path: Path) -> dict:
    # ``weights_only=False`` is required because project checkpoints contain
    # small metadata objects as well as state dictionaries.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(path, map_location="cpu")


def _as_float(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _vdp_ring(samples: int, expansion: float, seed: int) -> np.ndarray:
    """Uniformly draw an extended-box ring, excluding the operating box."""
    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(VDP_DOMAIN.lower), np.asarray(VDP_DOMAIN.upper)
    width = upper - lower
    extended_lower, extended_upper = lower - expansion * width, upper + expansion * width
    accepted: list[np.ndarray] = []
    while sum(len(chunk) for chunk in accepted) < samples:
        candidate = rng.uniform(extended_lower, extended_upper, size=(max(4 * samples, 32), 2))
        outside = np.any((candidate < lower) | (candidate > upper), axis=1)
        accepted.append(candidate[outside])
    points = np.vstack(accepted)[:samples]
    return np.column_stack((points, np.zeros(samples)))


def _penicillin_shell(samples: int, expansion: float, seed: int) -> np.ndarray:
    """Draw equally from the two one-dimensional shells around [0.1, 0.3]."""
    rng = np.random.default_rng(seed)
    lower, upper = 0.1, 0.3
    width = upper - lower
    extension = expansion * width
    left_count = samples // 2
    values = np.concatenate((
        rng.uniform(lower - extension, lower, size=left_count),
        rng.uniform(upper, upper + extension, size=samples - left_count),
    ))
    rng.shuffle(values)
    return values


def _load_vdp_policy(checkpoint: dict, key: str):
    model = KKTPolicyValueNetwork(TrainConfig())
    model.load_state_dict(checkpoint[key])
    model.eval()
    mean, std = checkpoint["normalization"][key]

    def predict(states: np.ndarray) -> tuple[np.ndarray, float]:
        x = torch.tensor((states[:, :2] - np.asarray(mean)) / np.asarray(std), dtype=torch.float32)
        start = time.perf_counter()
        with torch.no_grad():
            _, controls = model(x)
        return controls.numpy(), (time.perf_counter() - start) / len(states)

    return predict


def _load_penicillin_policy(checkpoint: dict, key: str):
    model = PenicillinPolicy()
    model.load_state_dict(checkpoint[key])
    model.eval()
    mean, std = checkpoint["normalization"][key]

    def predict(x2: np.ndarray) -> tuple[np.ndarray, float]:
        x = torch.tensor(((x2 - _as_float(mean)) / _as_float(std))[:, None], dtype=torch.float32)
        start = time.perf_counter()
        with torch.no_grad():
            _, controls = model(x)
        return controls.numpy(), (time.perf_counter() - start) / len(x2)

    return predict


def _summarise(rows: list[dict]) -> dict:
    raw = np.asarray([row["raw_hds_max_g"] for row in rows], dtype=float)
    accepted = np.asarray([row["accepted_after_hds_lambda"] for row in rows], dtype=bool)
    applied = np.asarray([row["applied_hds_max_g"] for row in rows], dtype=float)
    return {
        "samples": len(rows),
        "raw_violation_rate": float(np.mean(raw > 0.0)),
        "maximum_raw_hds_g": float(raw.max()),
        "bypass_guard_accepted_rate": float(accepted.mean()),
        "bypass_guard_fallback_rate": float(1.0 - accepted.mean()),
        "maximum_applied_hds_g_on_accepted": float(np.nanmax(applied)) if accepted.any() else None,
        "mean_corrected_segments": float(np.mean([row["corrected_segments"] for row in rows])),
        "mean_abs_lambda_minus_one": float(np.nanmean([row["mean_abs_lambda_minus_one"] for row in rows])),
        "mean_objective_change_from_nominal": float(np.nanmean([row["objective_change_from_nominal"] for row in rows])),
        "default_deployment_offline_fallback_rate": float(np.mean([not row["domain_guard_use_policy"] for row in rows])),
    }


def _run_vdp_layer(states: np.ndarray, predict, grid_size: int, safety_margin: float) -> list[dict]:
    corrector = HDSLambdaCorrector(vdp_ode, vdp_g, vdp_gdot, (-0.3, 1.0), HDSLambdaConfig(grid_size=grid_size, safety_margin=safety_margin, max_step_fraction=100.0))
    controls, inference_seconds = predict(states)
    rows = []
    for index, (state, nominal) in enumerate(zip(states, controls)):
        assessment = assess_initial_state(state, VDP_DOMAIN, [lambda x: x[0] + 0.4])
        raw_peak = corrector.audit(state, nominal, 0.5)
        raw_cost = terminal_cost(state, nominal, corrector, 0.5)
        correction = corrector.correct(state, nominal, 0.5)
        applied_peak = corrector.audit(state, correction.controls, 0.5) if correction.accepted else np.nan
        applied_cost = terminal_cost(state, correction.controls, corrector, 0.5) if correction.accepted else np.nan
        lambdas = [segment.lambda_value for segment in correction.segments if segment.lambda_value is not None]
        rows.append({
            "sample_index": index, "y1_0": state[0], "y2_0": state[1],
            "domain_guard_use_policy": assessment.use_policy, "domain_guard_action": assessment.action,
            "raw_hds_max_g": raw_peak, "accepted_after_hds_lambda": correction.accepted,
            "applied_hds_max_g": applied_peak, "corrected_segments": sum(segment.corrected for segment in correction.segments),
            "mean_abs_lambda_minus_one": float(np.mean(np.abs(np.asarray(lambdas) - 1.0))) if lambdas else np.nan,
            "objective_change_from_nominal": applied_cost - raw_cost if correction.accepted else np.nan,
            "inference_seconds": inference_seconds,
        })
    return rows


def _run_penicillin_layer(x2_values: np.ndarray, predict, grid_size: int, safety_margin: float) -> list[dict]:
    corrector = HDSLambdaCorrector(pen_ode, pen_g, pen_gdot, (0.0, PEN_UMAX), HDSLambdaConfig(grid_size=grid_size, safety_margin=safety_margin, max_step_fraction=100.0))
    controls, inference_seconds = predict(x2_values)
    rows = []
    for index, (x2, nominal) in enumerate(zip(x2_values, controls)):
        state = np.array([1.0, x2, 0.001, 250.0])
        assessment = assess_initial_state(state, PEN_DOMAIN, [lambda x: 0.5 - x[1]])
        raw_peak = corrector.audit(state, nominal, PEN_DT)
        raw_product = terminal_product(float(x2), nominal, corrector)
        correction = corrector.correct(state, nominal, PEN_DT)
        applied_peak = corrector.audit(state, correction.controls, PEN_DT) if correction.accepted else np.nan
        applied_product = terminal_product(float(x2), correction.controls, corrector) if correction.accepted else np.nan
        lambdas = [segment.lambda_value for segment in correction.segments if segment.lambda_value is not None]
        rows.append({
            "sample_index": index, "x2_0": x2,
            "domain_guard_use_policy": assessment.use_policy, "domain_guard_action": assessment.action,
            "raw_hds_max_g": raw_peak, "accepted_after_hds_lambda": correction.accepted,
            "applied_hds_max_g": applied_peak, "corrected_segments": sum(segment.corrected for segment in correction.segments),
            "mean_abs_lambda_minus_one": float(np.mean(np.abs(np.asarray(lambdas) - 1.0))) if lambdas else np.nan,
            # Product is maximized, so a negative difference is degradation.
            "objective_change_from_nominal": applied_product - raw_product if correction.accepted else np.nan,
            "inference_seconds": inference_seconds,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("vdp", "penicillin"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="models.pth from a frozen in-domain run")
    parser.add_argument("--policy-key", default=None, help="Checkpoint policy key; defaults to S for VDP and true_KKT for penicillin.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-layer", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument("--safety-margin", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260750)
    args = parser.parse_args()
    if args.samples_per_layer < 1:
        raise ValueError("samples-per-layer must be positive")

    checkpoint = _load_checkpoint(args.checkpoint)
    if args.problem == "vdp":
        key = args.policy_key or "S"
        predictor = _load_vdp_policy(checkpoint, key)
        layers = {
            "near_10_percent": _vdp_ring(args.samples_per_layer, 0.10, args.seed),
            "far_20_percent": _vdp_ring(args.samples_per_layer, 0.20, args.seed + 1),
        }
        evaluate = lambda points: _run_vdp_layer(points, predictor, args.grid_size, args.safety_margin)
    else:
        key = args.policy_key or "true_KKT"
        predictor = _load_penicillin_policy(checkpoint, key)
        layers = {
            "near_10_percent": _penicillin_shell(args.samples_per_layer, 0.10, args.seed),
            "far_20_percent": _penicillin_shell(args.samples_per_layer, 0.20, args.seed + 1),
        }
        evaluate = lambda points: _run_penicillin_layer(points, predictor, args.grid_size, args.safety_margin)

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "problem": args.problem,
        "checkpoint": str(args.checkpoint),
        "policy_key": key,
        "protocol": "Out-of-domain robustness stress test; no distribution-level safety or optimality guarantee is claimed.",
        "deployment_rule": "All out-of-domain initial states are routed directly to the offline optimizer; HDS-lambda results below intentionally bypass that guard for diagnosis only.",
        "safety_rule": f"event-located HDS peak <= -{args.safety_margin:g}",
        "layers": {},
    }
    for name, points in layers.items():
        rows = evaluate(points)
        with (args.output / f"{name}_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report["layers"][name] = _summarise(rows)
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
