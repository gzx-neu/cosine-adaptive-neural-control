"""Select the CSTR HDS--lambda candidate-grid density on independent data.

The policy checkpoints and the branch-selection protocol are frozen.  This
script varies only the bounded candidate grid used by the pre-execution
corrector, scores it against independent cold-start NLP references, and saves
the validation-only choice for a subsequent frozen-test evaluation.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from kkt_collocation.run_cstr_full_simulation import (
    CSTRConfig, CSTRTranscription, Policy, cstr_ode, lhs_states, objective,
    path_constraint, path_derivative, predict,
)

SEEDS = (20260718, 20260725, 20260726)
GRID_SIZES = (31, 61, 101)


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def make_corrector(cfg: CSTRConfig, grid_size: int) -> HDSLambdaCorrector:
    return HDSLambdaCorrector(
        lambda t, x, u: cstr_ode(t, x, u, cfg),
        lambda x: path_constraint(x, cfg),
        lambda x, u: path_derivative(x, u, cfg),
        (cfg.cooling_min, cfg.cooling_max),
        HDSLambdaConfig(grid_size=grid_size, safety_margin=1e-6, max_step_fraction=75.0),
    )


def score(checkpoint: Path, cfg: CSTRConfig, states: np.ndarray, references: np.ndarray, grid_size: int) -> dict:
    item = load(checkpoint)
    model = Policy(cfg)
    model.load_state_dict(item["model"])
    model.eval()
    controls, inference_seconds = predict(model, np.asarray(item["mean"]), np.asarray(item["std"]), states)
    corrector = make_corrector(cfg, grid_size)
    gaps, times, changes, corrected = [], [], [], []
    for state, nominal, reference in zip(states, controls, references):
        raw_objective = objective(state, nominal, cfg)
        started = time.perf_counter()
        outcome = corrector.correct(state, nominal, cfg.zoh_duration)
        times.append(time.perf_counter() - started)
        if not outcome.accepted:
            return {"accepted": False}
        applied_objective = objective(state, outcome.controls, cfg)
        gaps.append(abs(applied_objective - reference) / max(abs(reference), 1e-12) * 100.0)
        changes.append(applied_objective - raw_objective)
        corrected.append(sum(segment.corrected for segment in outcome.segments))
    return {
        "accepted": True,
        "objective_gap_percent": float(np.mean(gaps)),
        "objective_gap_sample_std_percent": float(np.std(gaps, ddof=1)),
        "mean_objective_change": float(np.mean(changes)),
        "mean_corrected_segments": float(np.mean(corrected)),
        "mean_hds_seconds": float(np.mean(times)),
        "mean_total_seconds": float(np.mean(times) + inference_seconds),
    }


def main() -> None:
    labels_file = ROOT / "kkt_collocation" / "results" / "cstr_full_simulation_900" / "cstr_labels.pkl"
    with labels_file.open("rb") as handle:
        labels = pickle.load(handle)
    cfg = CSTRConfig(**labels["config"])
    checkpoints = ROOT / "kkt_collocation" / "results" / "cstr_kkt_ablation1200_900"
    report = {
        "criterion": "For each frozen KKT-refined policy, select the accepted candidate-grid density with the smallest mean relative objective gap on an independent 60-point cold-start-NLP validation cohort.",
        "grid_sizes": GRID_SIZES,
        "seeds": {},
    }
    for seed in SEEDS:
        states = lhs_states(cfg, 60, seed + 1000)
        nlp = CSTRTranscription(cfg)
        references = np.asarray([nlp.solve(state)["objective"] for state in states], dtype=float)
        results = {
            str(grid): score(checkpoints / f"cstr_seed{seed}_kkt_refined.pth", cfg, states, references, grid)
            for grid in GRID_SIZES
        }
        eligible = [(result["objective_gap_percent"], grid) for grid, result in results.items() if result["accepted"]]
        selected = min(eligible)[1] if eligible else None
        report["seeds"][str(seed)] = {
            "validation_seed": seed + 1000,
            "reference_points": 60,
            "grid_results": results,
            "selected_grid_size": selected,
        }
        print(seed, report["seeds"][str(seed)], flush=True)
    output = ROOT / "kkt_collocation" / "results" / "eai_extension" / "cstr_lambda_grid_validation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
