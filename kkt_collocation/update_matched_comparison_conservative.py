"""Rebuild the matched 400-point comparison for the conservative HDS rule."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from kkt_collocation.run_cstr_full_simulation import CSTRConfig, cstr_ode, path_constraint, path_derivative
from kkt_collocation.run_penicillin_ablation import DT as PEN_DT, UMAX as PEN_UMAX, g as pen_g, gdot as pen_gdot, ode as pen_ode
from kkt_collocation.run_vdp_ablation import constraint as vdp_g, constraint_derivative as vdp_gdot, vdp_ode


SEED_DIRS = {
    "VDP": ROOT / "kkt_collocation" / "results" / "conservative_margin_1e6" / "vdp_seed20260751",
    "Penicillin": ROOT / "kkt_collocation" / "results" / "conservative_margin_1e6" / "penicillin_seed20260761",
    "CSTR": ROOT / "kkt_collocation" / "results" / "conservative_margin_1e6" / "cstr_seed20260718",
}


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def corrector(problem: str, margin: float) -> tuple[HDSLambdaCorrector, float]:
    if problem == "VDP":
        return HDSLambdaCorrector(vdp_ode, vdp_g, vdp_gdot, (-0.3, 1.0),
                                  HDSLambdaConfig(grid_size=31, safety_margin=margin, max_step_fraction=100.0)), 0.5
    if problem == "Penicillin":
        return HDSLambdaCorrector(pen_ode, pen_g, pen_gdot, (0.0, PEN_UMAX),
                                  HDSLambdaConfig(grid_size=31, safety_margin=margin, max_step_fraction=100.0)), PEN_DT
    checkpoint = np.load(SEED_DIRS[problem] / "population_controls.npz")
    del checkpoint
    cfg = CSTRConfig()
    funcs = (lambda t, x, u: cstr_ode(t, x, u, cfg), lambda x: path_constraint(x, cfg),
             lambda x, u: path_derivative(x, u, cfg))
    return HDSLambdaCorrector(*funcs, (cfg.cooling_min, cfg.cooling_max),
                              HDSLambdaConfig(grid_size=31, safety_margin=margin, max_step_fraction=100.0)), cfg.zoh_duration


def reference_data(problem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if problem in ("VDP", "Penicillin"):
        source = rows(ROOT / "kkt_collocation" / "results" / "jiang_fu_algorithm1_matched400" / "per_point.csv")
        selected = [row for row in source if row["problem"].lower() == problem.lower()]
        return (np.asarray([float(row["independent_objective"]) for row in selected]),
                np.asarray([float(row["solve_seconds"]) for row in selected]),
                np.asarray([float(row["hds_gmax"]) for row in selected]))
    source = rows(ROOT / "kkt_collocation" / "results" / "cstr_matched400_coldstart_900" / "cstr_coldstart_nlp_comparison.csv")
    return (np.asarray([float(row["reference_objective"]) for row in source]),
            np.asarray([float(row["coldstart_nlp_seconds"]) for row in source]),
            np.asarray([float(row["reference_hds_max_g_K"]) for row in source]))


def stats(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("VDP", "Penicillin", "CSTR"), required=True)
    parser.add_argument("--margin", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=ROOT / "kkt_collocation" / "results" / "conservative_margin_1e6" / "matched400")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    population = np.load(SEED_DIRS[args.problem] / "population_controls.npz")
    states = np.asarray(population["initial_states"]); nominal = np.asarray(population["nominal_controls"])
    applied_expected = np.asarray(population["applied_controls"])
    deployment = rows(SEED_DIRS[args.problem] / "per_sample.csv")
    applied_objective = np.asarray([float(row["applied_objective"]) for row in deployment])
    applied_peak = np.asarray([float(row["independent_hds_max_g"]) for row in deployment])
    inference = np.asarray([float(row["inference_seconds"]) for row in deployment])
    auditor, duration = corrector(args.problem, args.margin)
    correction_times = []
    for index, (state, control) in enumerate(zip(states, nominal)):
        started = time.perf_counter(); outcome = auditor.correct(state, control, duration); correction_times.append(time.perf_counter() - started)
        if not outcome.accepted or not np.allclose(outcome.controls, applied_expected[index], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{args.problem} conservative correction mismatch at sample {index}")
        if (index + 1) % 100 == 0:
            print(f"{args.problem}: timed {index + 1}/400", flush=True)
    valc_time = np.asarray(correction_times) + inference
    reference_objective, reference_time, reference_peak = reference_data(args.problem)
    if len(reference_objective) != 400:
        raise RuntimeError(f"{args.problem} reference has {len(reference_objective)} points")
    relative = np.abs(applied_objective - reference_objective) / np.maximum(np.abs(reference_objective), 1e-12) * 100.0
    report = {
        "problem": args.problem, "points": 400, "safety_margin": args.margin,
        "reference": {"time_seconds": stats(reference_time), "objective": stats(reference_objective),
                      "accepted": int(np.sum(reference_peak <= -args.margin)), "max_hds_peak": float(reference_peak.max())},
        "VALC": {"time_seconds": stats(valc_time), "objective": stats(applied_objective),
                 "relative_objective_difference_percent": stats(relative),
                 "accepted": int(np.sum(applied_peak <= 0.0)), "max_independent_hds_peak": float(applied_peak.max())},
    }
    (args.output / f"{args.problem.lower()}_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
