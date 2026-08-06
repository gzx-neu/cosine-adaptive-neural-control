"""Evaluate frozen 900-label CSTR policies with a 31-point lambda grid.

This diagnostic leaves the published 21-point results untouched and evaluates
the same three frozen checkpoints on the same 3 x 400 LHS cohorts.
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from kkt_collocation.run_cstr_full_simulation import (
    CSTRConfig, Policy, cstr_ode, lhs_states, objective, path_constraint,
    path_derivative, predict, summarise,
)


TRAIN_SEEDS = (20260718, 20260725, 20260726)
TEST_SEEDS = (20260724, 20260825, 20260826)
SOURCE = ROOT / "kkt_collocation" / "results" / "cstr_multiseed_test1200_900_clean"
OUTPUT = ROOT / "kkt_collocation" / "results" / "cstr_grid31_comparison_900"
FIELDS = (
    "method", "training_seed", "sample_index", "CA0", "T0", "nominal_hds_max_g",
    "applied_hds_max_g", "nominal_objective", "applied_objective", "objective_change",
    "accepted", "fallback", "corrected_segments", "mean_abs_lambda_minus_one",
    "candidate_evaluations", "inference_seconds", "filter_seconds",
)


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def make_corrector(cfg: CSTRConfig) -> HDSLambdaCorrector:
    return HDSLambdaCorrector(
        lambda t, x, u: cstr_ode(t, x, u, cfg),
        lambda x: path_constraint(x, cfg),
        lambda x, u: path_derivative(x, u, cfg),
        (cfg.cooling_min, cfg.cooling_max),
        HDSLambdaConfig(grid_size=31, safety_margin=cfg.hds_tolerance, max_step_fraction=75.0),
    )


def read_21_rows(path: Path) -> dict[tuple[int, int], dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            (int(row["training_seed"]), int(row["sample_index"])): row
            for row in csv.DictReader(handle)
        }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "kkt_collocation" / "results" / "cstr_full_simulation_900" / "cstr_labels.pkl").open("rb") as handle:
        labels = pickle.load(handle)
    cfg = CSTRConfig(**labels["config"])
    baseline = read_21_rows(SOURCE / "per_sample.csv")
    rows: list[dict] = []

    for training_seed, test_seed in zip(TRAIN_SEEDS, TEST_SEEDS):
        checkpoint = load_checkpoint(SOURCE / f"cstr_seed{training_seed}.pth")
        model = Policy(cfg)
        model.load_state_dict(checkpoint["model"])
        states = lhs_states(cfg, 400, test_seed)
        controls, inference = predict(model, np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"]), states)
        corrector = make_corrector(cfg)
        for index, (state, nominal) in enumerate(zip(states, controls)):
            raw_peak = corrector.audit(state, nominal, cfg.zoh_duration)
            raw_objective = objective(state, nominal, cfg)
            started = time.perf_counter()
            outcome = corrector.correct(state, nominal, cfg.zoh_duration)
            elapsed = time.perf_counter() - started
            accepted = outcome.accepted
            applied = outcome.controls if accepted else None
            applied_peak = corrector.audit(state, applied, cfg.zoh_duration) if accepted else np.nan
            applied_objective = objective(state, applied, cfg) if accepted else np.nan
            lambdas = np.asarray([segment.lambda_value for segment in outcome.segments if segment.lambda_value is not None])
            row = {
                "method": "Adaptive+HDS (31 candidates)", "training_seed": training_seed,
                "sample_index": index, "CA0": state[0], "T0": state[1],
                "nominal_hds_max_g": raw_peak, "applied_hds_max_g": applied_peak,
                "nominal_objective": raw_objective, "applied_objective": applied_objective,
                "objective_change": applied_objective - raw_objective if accepted else np.nan,
                "accepted": accepted, "fallback": not accepted,
                "corrected_segments": sum(segment.corrected for segment in outcome.segments),
                "mean_abs_lambda_minus_one": float(np.mean(np.abs(lambdas - 1.0))) if lambdas.size else 0.0,
                "candidate_evaluations": sum(segment.candidate_evaluations for segment in outcome.segments),
                "inference_seconds": inference, "filter_seconds": elapsed,
            }
            rows.append(row)
            if (index + 1) % 50 == 0:
                print(f"seed {training_seed}: {index + 1}/400", flush=True)

    with (OUTPUT / "per_sample_grid31.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)

    per_seed = {str(seed): summarise([row for row in rows if row["training_seed"] == seed]) for seed in TRAIN_SEEDS}
    differences = []
    for row in rows:
        old = baseline[(row["training_seed"], row["sample_index"])]
        outcome_changed = (
            str(row["accepted"]).lower() != old["accepted"].lower()
            or row["corrected_segments"] != int(old["corrected_segments"])
            or abs(row["applied_hds_max_g"] - float(old["applied_hds_max_g"])) > 1e-9
            or abs(row["objective_change"] - float(old["objective_change"])) > 1e-9
        )
        if outcome_changed:
            differences.append({
                "training_seed": row["training_seed"], "sample_index": row["sample_index"],
                "corrected_segments_21": int(old["corrected_segments"]),
                "corrected_segments_31": row["corrected_segments"],
                "peak_21": float(old["applied_hds_max_g"]), "peak_31": row["applied_hds_max_g"],
                "objective_change_21": float(old["objective_change"]),
                "objective_change_31": row["objective_change"],
                "mean_abs_lambda_minus_one_21": float(old["mean_abs_lambda_minus_one"]),
                "mean_abs_lambda_minus_one_31": row["mean_abs_lambda_minus_one"],
            })
    metrics = ("raw_violation_rate_percent", "accepted_rate_percent", "fallback_rate_percent", "corrected_segments_mean", "accepted_peak_max_K", "mean_objective_change", "mean_hds_ms")
    aggregate = {
        metric: {"mean": float(np.mean([per_seed[str(seed)][metric] for seed in TRAIN_SEEDS])),
                 "sample_std": float(np.std([per_seed[str(seed)][metric] for seed in TRAIN_SEEDS], ddof=1))}
        for metric in metrics
    }
    report = {
        "purpose": "Frozen-checkpoint comparison of 21- and 31-point lambda candidate grids.",
        "config": asdict(cfg), "grid_size": 31, "samples": len(rows),
        "per_seed": per_seed, "aggregate": aggregate,
        "comparison_with_grid21": {"changed_rows": len(differences), "rows": differences},
    }
    (OUTPUT / "summary_grid31.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
