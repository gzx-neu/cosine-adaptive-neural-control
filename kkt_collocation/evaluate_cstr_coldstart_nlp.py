"""Matched cold-start CSTR NLP comparison for a frozen Adaptive+HDS policy.

The initial conditions are a frozen Latin-hypercube cohort in the declared
operating domain. Every deterministic reference solve starts from
the transcription's physics-based conservative guess, never from a stored
label or neural-policy sequence.  The same event-located HDS audit is applied
to the learned and reference sequences before reporting a relative objective
difference.  This is an empirical same-grid comparison, not a global-optimality
claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kkt_collocation.run_cstr_full_simulation import (  # noqa: E402
    CSTRConfig,
    CSTRTranscription,
    Policy,
    lhs_states,
    make_corrector,
    objective,
    predict,
)


def mean_sd(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "cstr_full_simulation_900")
    parser.add_argument("--points", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", type=Path, default=None,
                        help="Separate checkpoint directory; defaults to --experiment.")
    args = parser.parse_args()
    output_dir = args.output or args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.experiment / "cstr_supervised.pth", map_location="cpu", weights_only=False)
    summary_path = args.experiment / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cfg = CSTRConfig(**summary["config"])
    else:
        cfg = CSTRConfig(**checkpoint["config"])
    policy = Policy(cfg); policy.load_state_dict(checkpoint["model"])
    states = lhs_states(cfg, args.points, args.seed)
    nominal, inference_per_sample = predict(policy, np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"]), states)
    corrector, nlp = make_corrector(cfg), CSTRTranscription(cfg)
    output_csv = output_dir / "cstr_coldstart_nlp_comparison.csv"
    fields = ("sample_index", "CA0", "T0", "raw_hds_max_g_K", "adaptive_hds_max_g_K",
              "reference_hds_max_g_K", "adaptive_objective", "reference_objective",
              "relative_objective_difference_percent", "corrected_segments",
              "adaptive_hds_seconds", "coldstart_nlp_seconds")
    rows: list[dict[str, float | int | bool]] = []
    if output_csv.exists():
        with output_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) > args.points:
            raise RuntimeError("Existing checkpoint has more rows than requested points.")
    start_index = len(rows)
    if start_index and int(rows[-1]["sample_index"]) != start_index - 1:
        raise RuntimeError("CSTR checkpoint sample indices are not contiguous.")

    mode = "a" if start_index else "w"
    with output_csv.open(mode, newline="", encoding="utf-8") as handle:
      writer = csv.DictWriter(handle, fieldnames=fields)
      if not start_index:
          writer.writeheader()
      for index in range(start_index, len(states)):
        state, raw_control = states[index], nominal[index]
        raw_peak = corrector.audit(state, raw_control, cfg.zoh_duration)
        adaptive_start = time.perf_counter()
        corrected = corrector.correct(state, raw_control, cfg.zoh_duration)
        adaptive_seconds = time.perf_counter() - adaptive_start + inference_per_sample
        if not corrected.accepted:
            raise RuntimeError(f"Adaptive+HDS fallback at frozen comparison point {index}")
        applied = corrected.controls
        applied_peak = corrector.audit(state, applied, cfg.zoh_duration)
        adaptive_objective = objective(state, applied, cfg)

        # No argument is passed as controls: this is the declared cold start.
        reference = nlp.solve(state, controls=None)
        reference_objective = float(reference["objective"])
        relative_gap = abs(adaptive_objective-reference_objective) / max(abs(reference_objective), 1e-12) * 100.0
        row = {
            "sample_index": index,
            "CA0": float(state[0]), "T0": float(state[1]),
            "raw_hds_max_g_K": float(raw_peak), "adaptive_hds_max_g_K": float(applied_peak),
            "reference_hds_max_g_K": float(reference["hds_max_g"]),
            "adaptive_objective": adaptive_objective, "reference_objective": reference_objective,
            "relative_objective_difference_percent": relative_gap,
            "corrected_segments": int(sum(segment.corrected for segment in corrected.segments)),
            "adaptive_hds_seconds": adaptive_seconds, "coldstart_nlp_seconds": float(reference["solve_seconds"]),
        }
        writer.writerow(row); handle.flush()
        rows.append(row)
        print(f"{index + 1}/{args.points}: gap={relative_gap:.3f}% adaptive={adaptive_seconds:.3f}s NLP={reference['solve_seconds']:.3f}s")
    if len(rows) != args.points:
        raise RuntimeError("CSTR checkpoint is incomplete.")
    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=float)
    aggregate = {
        "comparison": f"{args.points}-point frozen LHS, matched cold-start direct-RK4 NLP reference",
        "cold_start_definition": "No neural control or stored label is supplied to the deterministic NLP.",
        "points": args.points, "seed": args.seed, "config": asdict(cfg),
        "relative_objective_difference_percent": mean_sd(values("relative_objective_difference_percent")),
        "adaptive_hds_seconds": mean_sd(values("adaptive_hds_seconds")),
        "coldstart_nlp_seconds": mean_sd(values("coldstart_nlp_seconds")),
        "adaptive_accepted": int(sum(values("adaptive_hds_max_g_K") <= cfg.hds_tolerance)),
        "reference_accepted": int(sum(values("reference_hds_max_g_K") <= cfg.hds_tolerance)),
        "mean_corrected_segments": float(np.mean(values("corrected_segments"))),
    }
    (output_dir / "cstr_coldstart_nlp_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
