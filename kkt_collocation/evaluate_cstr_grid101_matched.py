"""Frozen 400-point CSTR comparison using the validation-selected 101-point grid."""
from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from kkt_collocation.run_cstr_full_simulation import (
    CSTRConfig, Policy, cstr_ode, lhs_states, objective, path_constraint,
    path_derivative, predict,
)

GRID_SIZE = 101
TRAINING_SEED = 20260718
TEST_SEED = 20260724


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def mean_sd(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1))}


def main() -> None:
    with (ROOT / "kkt_collocation" / "results" / "cstr_full_simulation_900" / "cstr_labels.pkl").open("rb") as handle:
        labels = pickle.load(handle)
    cfg = replace(CSTRConfig(**labels["config"]), hds_tolerance=1e-6)
    checkpoint = load(ROOT / "kkt_collocation" / "results" / "cstr_kkt_ablation1200_900" / f"cstr_seed{TRAINING_SEED}_kkt_refined.pth")
    policy = Policy(cfg)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    states = lhs_states(cfg, 400, TEST_SEED)
    nominal, inference_seconds = predict(policy, np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"]), states)
    corrector = HDSLambdaCorrector(
        lambda t, x, u: cstr_ode(t, x, u, cfg),
        lambda x: path_constraint(x, cfg),
        lambda x, u: path_derivative(x, u, cfg),
        (cfg.cooling_min, cfg.cooling_max),
        HDSLambdaConfig(grid_size=GRID_SIZE, safety_margin=cfg.hds_tolerance, max_step_fraction=75.0),
    )
    reference_path = ROOT / "kkt_collocation" / "results" / "cstr_matched400_coldstart_900" / "cstr_coldstart_nlp_comparison.csv"
    with reference_path.open(encoding="utf-8", newline="") as handle:
        references = list(csv.DictReader(handle))
    rows = []
    for index, (state, control, reference) in enumerate(zip(states, nominal, references)):
        raw = corrector.audit(state, control, cfg.zoh_duration)
        raw_objective = objective(state, control, cfg)
        started = time.perf_counter()
        result = corrector.correct(state, control, cfg.zoh_duration)
        elapsed = time.perf_counter() - started
        if not result.accepted:
            raise RuntimeError(f"Unexpected fallback at frozen test point {index}.")
        applied = result.controls
        applied_peak = corrector.audit(state, applied, cfg.zoh_duration)
        applied_objective = objective(state, applied, cfg)
        ref = float(reference["reference_objective"])
        rows.append({
            "sample_index": index, "CA0": float(state[0]), "T0": float(state[1]),
            "nominal_hds_max_g": float(raw), "applied_hds_max_g": float(applied_peak),
            "nominal_objective": float(raw_objective), "applied_objective": float(applied_objective),
            "reference_objective": ref,
            "relative_objective_difference_percent": abs(applied_objective - ref) / max(abs(ref), 1e-12) * 100.0,
            "objective_change": float(applied_objective - raw_objective),
            "corrected_segments": int(sum(segment.corrected for segment in result.segments)),
            "hds_seconds": float(elapsed),
        })
        if (index + 1) % 50 == 0:
            print(f"{index + 1}/400", flush=True)
    output = ROOT / "kkt_collocation" / "results" / "eai_extension" / "cstr_grid101_matched400"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    values = lambda key: np.asarray([float(row[key]) for row in rows])
    report = {
        "purpose": "Frozen matched 400-point cold-start comparison using the 101-point lambda grid selected on separate validation cohorts.",
        "config": asdict(cfg), "grid_size": GRID_SIZE, "training_seed": TRAINING_SEED, "test_seed": TEST_SEED,
        "points": len(rows),
        "relative_objective_difference_percent": mean_sd(values("relative_objective_difference_percent")),
        "valc_time_seconds": mean_sd(values("hds_seconds") + inference_seconds),
        "valc_max_hds_g": float(values("applied_hds_max_g").max()),
        "valc_objective": mean_sd(values("applied_objective")),
        "reference_time_seconds": mean_sd(np.asarray([float(row["coldstart_nlp_seconds"]) for row in references])),
        "reference_objective": mean_sd(values("reference_objective")),
        "mean_objective_change": float(values("objective_change").mean()),
        "mean_corrected_segments": float(values("corrected_segments").mean()),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
