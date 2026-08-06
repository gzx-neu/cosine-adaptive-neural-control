"""Deploy CSTR KKT-refined policies selected by independent validation references.

The selected branch for each seed is determined in
``cstr_validation_objective_selection.json`` before this frozen 3x400 test
evaluation is read.  HDS acceptance uses the final conservative margin.
"""
from __future__ import annotations

import csv
import argparse
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

SEEDS = (20260718, 20260725, 20260726)
TEST_SEEDS = (20260724, 20260825, 20260826)


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def mean_sd(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sample_std": float(a.std(ddof=1))}


def make_grid_corrector(cfg: CSTRConfig, grid_size: int) -> HDSLambdaCorrector:
    return HDSLambdaCorrector(
        lambda t, x, u: cstr_ode(t, x, u, cfg),
        lambda x: path_constraint(x, cfg),
        lambda x, u: path_derivative(x, u, cfg),
        (cfg.cooling_min, cfg.cooling_max),
        HDSLambdaConfig(grid_size=grid_size, safety_margin=cfg.hds_tolerance, max_step_fraction=75.0),
    )


def summarize(rows: list[dict]) -> dict:
    raw = np.asarray([row["nominal_hds_max_g"] for row in rows])
    applied = np.asarray([row["applied_hds_max_g"] for row in rows])
    return {
        "samples": len(rows),
        "nominal_violation_rate_percent": float(100 * np.mean(raw > 1e-8)),
        "nominal_dispatch_rate_percent": float(100 * np.mean(raw > -1e-6)),
        "accepted_rate_percent": float(100 * np.mean([row["accepted"] for row in rows])),
        "fallback_rate_percent": float(100 * np.mean([row["fallback"] for row in rows])),
        "mean_corrected_segments": float(np.mean([row["corrected_segments"] for row in rows])),
        "mean_objective_change": float(np.mean([row["objective_change"] for row in rows])),
        "mean_hds_seconds": float(np.mean([row["hds_seconds"] for row in rows])),
        "max_independent_hds_peak": float(applied.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    labels_path = ROOT / "kkt_collocation" / "results" / "cstr_full_simulation_900" / "cstr_labels.pkl"
    with labels_path.open("rb") as handle: labels = pickle.load(handle)
    cfg = CSTRConfig(**labels["config"])
    selection_path = ROOT / "kkt_collocation" / "results" / "eai_extension" / "cstr_validation_objective_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if any(selection["seeds"][str(seed)]["selected_branch"] != "KKT-refined" for seed in SEEDS):
        raise RuntimeError("This evaluator is intentionally for the KKT-refined validation-selected branch.")
    checkpoint_dir = ROOT / "kkt_collocation" / "results" / "cstr_kkt_ablation1200_900"
    output = args.output or (ROOT / "kkt_collocation" / "results" / "eai_extension" / "cstr_validation_selected")
    output.mkdir(parents=True, exist_ok=True)
    cfg = replace(cfg, hds_tolerance=1e-6)
    corrector = make_grid_corrector(cfg, args.grid_size)
    rows = []
    for seed, test_seed in zip(SEEDS, TEST_SEEDS):
        item = load(checkpoint_dir / f"cstr_seed{seed}_kkt_refined.pth")
        model = Policy(cfg); model.load_state_dict(item["model"]); model.eval()
        states = lhs_states(cfg, 400, test_seed)
        nominal, inference = predict(model, np.asarray(item["mean"]), np.asarray(item["std"]), states)
        for index, (state, control) in enumerate(zip(states, nominal)):
            raw = corrector.audit(state, control, cfg.zoh_duration)
            raw_obj = objective(state, control, cfg)
            start = time.perf_counter(); outcome = corrector.correct(state, control, cfg.zoh_duration); elapsed = time.perf_counter() - start
            if not outcome.accepted:
                raise RuntimeError(f"Unexpected fallback for seed {seed}, test point {index}")
            applied = outcome.controls
            peak = corrector.audit(state, applied, cfg.zoh_duration)
            applied_obj = objective(state, applied, cfg)
            rows.append({"training_seed": seed, "sample_index": index, "CA0": float(state[0]), "T0": float(state[1]),
                         "nominal_hds_max_g": float(raw), "applied_hds_max_g": float(peak), "accepted": True, "fallback": False,
                         "nominal_objective": raw_obj, "applied_objective": applied_obj, "objective_change": applied_obj - raw_obj,
                         "corrected_segments": int(sum(segment.corrected for segment in outcome.segments)),
                         "inference_seconds": float(inference), "hds_seconds": float(elapsed)})
            if (index + 1) % 50 == 0: print(f"seed {seed}: {index + 1}/400", flush=True)
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    per_seed = {str(seed): summarize([row for row in rows if row["training_seed"] == seed]) for seed in SEEDS}
    aggregate = {key: mean_sd([per_seed[str(seed)][key] for seed in SEEDS]) for key in (
        "nominal_violation_rate_percent", "nominal_dispatch_rate_percent", "accepted_rate_percent", "fallback_rate_percent",
        "mean_corrected_segments", "mean_objective_change", "mean_hds_seconds", "max_independent_hds_peak")}

    # The existing seed-20260718 cold-start records supply an independent
    # reference without using a test-time solver result to select the policy.
    ref_path = ROOT / "kkt_collocation" / "results" / "cstr_matched400_coldstart_900" / "cstr_coldstart_nlp_comparison.csv"
    with ref_path.open(newline="", encoding="utf-8") as handle: references = list(csv.DictReader(handle))
    first = [row for row in rows if row["training_seed"] == SEEDS[0]]
    gaps = [abs(row["applied_objective"] - float(ref["reference_objective"])) / max(abs(float(ref["reference_objective"])), 1e-12) * 100
            for row, ref in zip(first, references)]
    comparison = {"points": len(first), "relative_objective_difference_percent": mean_sd(gaps),
                  "valc_time_seconds": mean_sd([row["hds_seconds"] + row["inference_seconds"] for row in first]),
                  "valc_max_hds_g": float(max(row["applied_hds_max_g"] for row in first)),
                  "valc_objective": mean_sd([row["applied_objective"] for row in first]),
                  "reference_time_seconds": mean_sd([float(row["coldstart_nlp_seconds"]) for row in references]),
                  "reference_max_hds_g": float(max(float(row["reference_hds_max_g_K"]) for row in references)),
                  "reference_objective": mean_sd([float(row["reference_objective"]) for row in references])}
    report = {"config": asdict(cfg), "grid_size": args.grid_size, "selection_source": str(selection_path), "per_seed": per_seed,
              "aggregate": aggregate, "matched_coldstart_seed_20260718": comparison}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
