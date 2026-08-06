"""All-400 adaptive-event HDS evaluation for the VDP Jiang--Fu CVP20 policies.

Cold-reference objectives are optional by design.  If no CVP20 test-reference
file is supplied, the evaluator reports deployment/audit quantities and leaves
objective-gap fields empty rather than borrowing an incompatible CVP10 value.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from offline_safe_control.adaptive_event_hds import AdaptiveEventHDSConfig, AdaptiveEventHDSCorrector  # noqa: E402
from kkt_collocation.run_vdp_ablation import constraint, constraint_derivative, vdp_ode  # noqa: E402
from kkt_collocation.run_vdp_cvp20_jiang_native_grid_ablation import Config, Policy  # noqa: E402

_CORRECTOR: AdaptiveEventHDSCorrector | None = None


def _init_worker() -> None:
    global _CORRECTOR
    _CORRECTOR = AdaptiveEventHDSCorrector(
        vdp_ode, constraint, constraint_derivative, (-0.3, 1.0), AdaptiveEventHDSConfig(grid_size=31)
    )


def _terminal_cost(initial: np.ndarray, controls: np.ndarray) -> float:
    if _CORRECTOR is None:
        raise RuntimeError("worker not initialized")
    state = np.asarray(initial, dtype=float).copy()
    for control in np.asarray(controls, dtype=float):
        _, state = _CORRECTOR.segment_peak(state, float(control), 0.25)
    return float(state[2])


def _evaluate_one(task: tuple[int, np.ndarray, np.ndarray, float, float]) -> dict:
    if _CORRECTOR is None:
        raise RuntimeError("worker not initialized")
    index, initial, nominal, infer_seconds, reference = task
    nominal_peak = float(_CORRECTOR.audit(initial, nominal, 0.25))
    nominal_cost = _terminal_cost(initial, nominal)
    started = time.perf_counter()
    outcome = _CORRECTOR.correct(initial, nominal, 0.25)
    hds_seconds = time.perf_counter() - started
    accepted = bool(outcome.accepted)
    applied = outcome.controls if accepted else None
    applied_peak = float(_CORRECTOR.audit(initial, applied, 0.25)) if accepted else np.nan
    applied_cost = _terminal_cost(initial, applied) if accepted else np.nan
    denom = abs(reference) if np.isfinite(reference) and abs(reference) > 1e-12 else np.nan
    corrected = int(sum(segment.corrected for segment in outcome.segments))
    return {
        "sample_index": index,
        "nominal_hds_max_g": nominal_peak,
        "nominal_objective": nominal_cost,
        "accepted": accepted,
        "fallback": not accepted,
        "applied_hds_max_g": applied_peak,
        "applied_objective": applied_cost,
        "reference_objective": reference,
        "nominal_relative_objective_gap": (nominal_cost - reference) / denom if np.isfinite(denom) else np.nan,
        "hds_relative_objective_gap": (applied_cost - reference) / denom if accepted and np.isfinite(denom) else np.nan,
        "hds_objective_change": applied_cost - nominal_cost if accepted else np.nan,
        "corrected_segments": corrected,
        "corrected_control_segment_fraction": corrected / 20.0,
        "inference_seconds": infer_seconds,
        "hds_audit_correction_seconds": hds_seconds,
        "total_deployment_seconds": infer_seconds + hds_seconds,
    }


def _mean(rows: list[dict], key: str) -> float | None:
    value = np.asarray([row[key] for row in rows], dtype=float)
    return float(np.nanmean(value)) if np.isfinite(value).any() else None


def summarize(rows: list[dict]) -> dict:
    nominal = np.asarray([row["nominal_hds_max_g"] for row in rows], dtype=float)
    applied = np.asarray([row["applied_hds_max_g"] for row in rows], dtype=float)
    accepted = np.asarray([row["accepted"] for row in rows], dtype=bool)
    corrected = np.asarray([row["corrected_segments"] for row in rows], dtype=float)
    return {
        "samples": len(rows),
        "nominal_continuous_time_violation_rate": float(np.mean(nominal > 0.0)),
        "nominal_max_g": float(nominal.max()),
        "hds_acceptance_rate": float(accepted.mean()),
        "hds_fallback_rate": float(1.0 - accepted.mean()),
        "hds_final_max_g": float(np.nanmax(applied)),
        "corrected_trajectory_fraction": float(np.mean(corrected > 0)),
        "corrected_control_segment_fraction": float(corrected.sum() / (20 * len(rows))),
        "mean_corrected_segments": float(corrected.mean()),
        "mean_hds_objective_change": _mean(rows, "hds_objective_change"),
        "mean_inference_seconds": _mean(rows, "inference_seconds"),
        "mean_hds_audit_correction_seconds": _mean(rows, "hds_audit_correction_seconds"),
        "mean_total_deployment_seconds": _mean(rows, "total_deployment_seconds"),
        "mean_nominal_relative_objective_gap": _mean(rows, "nominal_relative_objective_gap"),
        "mean_hds_relative_objective_gap": _mean(rows, "hds_relative_objective_gap"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-states", type=Path, default=ROOT / "kkt_collocation/results/final_multiseed_vdp900_penalty_seed20260751/test_states.npy")
    parser.add_argument("--reference-objectives", type=Path, default=None, help="Optional .npy, length-400 CVP20 cold objectives in matching point order.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    states = np.load(args.test_states)
    if states.shape != (400, 3):
        raise ValueError(f"expected frozen 400 VDP states, got {states.shape}")
    reference = np.full(400, np.nan) if args.reference_objectives is None else np.asarray(np.load(args.reference_objectives), dtype=float)
    if reference.shape != (400,):
        raise ValueError("reference objectives must have shape (400,)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    models: dict[str, np.ndarray] = {}
    infer: dict[str, float] = {}
    for path in sorted(args.training.glob("*.pth")):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = Policy(cfg).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        x = torch.tensor((states[:, :2] - checkpoint["input_mean"].numpy()) / checkpoint["input_std"].numpy(), dtype=torch.float32, device=device)
        started = time.perf_counter()
        with torch.no_grad():
            controls = model(x).cpu().numpy()
        models[str(checkpoint["method"])] = controls
        infer[str(checkpoint["method"])] = (time.perf_counter() - started) / 400.0
    if not models:
        raise ValueError("no policy checkpoints found")
    args.output.mkdir(parents=True)
    metadata = {
        "formal_protocol": False,
        "training_directory": str(args.training),
        "test_states": str(args.test_states),
        "test_samples": 400,
        "reference_objectives": str(args.reference_objectives) if args.reference_objectives else None,
        "reference_gap_status": "unavailable: no CVP20 cold-reference objective file supplied" if args.reference_objectives is None else "available",
        "hds_settings": {"integrator": "DOP853", "adaptive_steps": True, "rtol": 1e-10, "atol": 1e-12, "lambda_candidates": 31, "safety_margin": 1e-6, "duration_per_zoh": 0.25},
        "hds_statement": "continuous-time numerical audit evidence under the declared model and numerical settings.",
    }
    summary = {"metadata": metadata, "methods": {}}
    for method, controls in models.items():
        tasks = [(index, states[index], controls[index], infer[method], float(reference[index])) for index in range(400)]
        if args.workers == 1:
            _init_worker(); rows = [_evaluate_one(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
                rows = list(pool.map(_evaluate_one, tasks, chunksize=1))
        with (args.output / f"per_sample_{method}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        summary["methods"][method] = summarize(rows)
        print(f"evaluated {method}", flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
