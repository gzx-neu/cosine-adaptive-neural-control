"""Evaluate the frozen CSTR supervised checkpoint on an extended 400-point cohort.

This script does not retrain or alter the 144 offline labels.  It draws one
new independent in-domain Latin-hypercube test cohort, applies the frozen
selected checkpoint, and runs the mandatory event-located HDS--lambda stage.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kkt_collocation.run_cstr_full_simulation import (  # noqa: E402
    CSTRConfig, Policy, lhs_states, make_corrector, objective, plot_controls, plot_results,
    predict, summarise,
)


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "cstr_full_simulation_900")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "cstr_extended_test400")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")

    checkpoint = load_checkpoint(args.experiment / "cstr_supervised.pth")
    cfg = CSTRConfig(**checkpoint["config"])
    model = Policy(cfg)
    model.load_state_dict(checkpoint["model"])
    args.output.mkdir(parents=True, exist_ok=True)
    states = lhs_states(cfg, args.samples, args.seed)
    controls, inference = predict(model, np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"]), states)
    csv_path = args.output / "per_sample.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            existing = list(csv.DictReader(handle))
    else:
        existing = []
    rows: list[dict] = existing
    corrector = make_corrector(cfg)
    fieldnames = ["method", "sample_index", "CA0", "T0", "nominal_hds_max_g", "applied_hds_max_g",
                  "nominal_objective", "applied_objective", "objective_change", "accepted", "fallback",
                  "corrected_segments", "mean_abs_lambda_minus_one", "inference_seconds", "filter_seconds"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for index in range(len(existing), args.samples):
            state, nominal = states[index], controls[index]
            raw_peak = corrector.audit(state, nominal, cfg.zoh_duration)
            raw_objective = objective(state, nominal, cfg)
            start = time.perf_counter()
            outcome = corrector.correct(state, nominal, cfg.zoh_duration)
            elapsed = time.perf_counter() - start
            if not outcome.accepted:
                raise RuntimeError(f"CSTR extended test fallback at sample {index}")
            applied_peak = corrector.audit(state, outcome.controls, cfg.zoh_duration)
            applied_objective = objective(state, outcome.controls, cfg)
            lambdas = np.asarray([s.lambda_value for s in outcome.segments if s.lambda_value is not None], dtype=float)
            row = {"method": "Adaptive (Supervised) + HDS", "sample_index": index, "CA0": state[0], "T0": state[1],
                   "nominal_hds_max_g": raw_peak, "applied_hds_max_g": applied_peak,
                   "nominal_objective": raw_objective, "applied_objective": applied_objective,
                   "objective_change": applied_objective - raw_objective, "accepted": True, "fallback": False,
                   "corrected_segments": int(sum(s.corrected for s in outcome.segments)),
                   "mean_abs_lambda_minus_one": float(np.mean(np.abs(lambdas - 1.0))) if len(lambdas) else 0.0,
                   "inference_seconds": inference, "filter_seconds": elapsed}
            writer.writerow(row); handle.flush(); rows.append(row)
            if (index + 1) % 20 == 0:
                print(f"completed {index + 1}/{args.samples}", flush=True)
    # Convert CSV strings from a resumed run to numeric records before summary/plots.
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("nominal_hds_max_g", "applied_hds_max_g", "nominal_objective", "applied_objective", "objective_change", "mean_abs_lambda_minus_one", "inference_seconds", "filter_seconds"):
            row[key] = float(row[key])
        row["corrected_segments"] = int(row["corrected_segments"])
        row["accepted"] = row["accepted"].strip().lower() == "true"
        row["fallback"] = row["fallback"].strip().lower() == "true"
    original = json.loads((args.experiment / "summary.json").read_text(encoding="utf-8"))
    report = {
        "model": original["model"], "config": original["config"], "labels": original["labels"],
        "gate": original["gate"], "selected_branch": "Supervised",
        "test_protocol": "Independent 400-point in-domain LHS cohort evaluated with a frozen checkpoint; no retraining.",
        "test_seed": args.seed, "test": summarise(rows),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_results(rows, states, controls, cfg, args.output / "cstr_temperature_population")
    plot_controls(rows, states, controls, cfg, args.output / "cstr_controls_population")
    figures = ROOT / "论文写作" / "figures"
    plot_results(rows, states, controls, cfg, figures / "cstr_temperature_population")
    plot_controls(rows, states, controls, cfg, figures / "cstr_controls_population")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
