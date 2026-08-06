"""Validation-only adaptive-HDS comparison for exploratory VDP/penicillin policies.

No fixed test state or cold-start test objective is read.  Candidate selection
uses only the established validation initial conditions and the two policies'
own adaptive-DOP853 corrected physical objectives.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_unified_su_suj_sk_konly_ablation import Problem, UnifiedConfig


def _mean(values: list[float]) -> float | None:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    return float(valid.mean()) if len(valid) else None


def _evaluate(problem: Problem, name: str, checkpoint_path: Path, states: np.ndarray) -> tuple[list[dict], dict]:
    checkpoint = torch.load(checkpoint_path, map_location=problem.device, weights_only=False)
    model = problem.make_model()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    controls, inference_seconds = problem.predict(model, states)
    corrector = problem.corrector()
    rows: list[dict] = []
    for index, (state, nominal) in enumerate(zip(states, controls)):
        initial = problem.initial_state(state)
        nominal_peak = corrector.audit(initial, nominal, problem.duration)
        nominal_objective = problem.cost(state, nominal, corrector)
        started = time.perf_counter()
        outcome = corrector.correct(initial, nominal, problem.duration)
        correction_seconds = time.perf_counter() - started
        if outcome.accepted:
            applied_peak = corrector.audit(initial, outcome.controls, problem.duration)
            applied_objective = problem.cost(state, outcome.controls, corrector)
        else:
            applied_peak = np.nan
            applied_objective = np.nan
        rows.append({
            "method": name,
            "index": index,
            "accepted": bool(outcome.accepted),
            "fallback": not bool(outcome.accepted),
            "nominal_max_g": nominal_peak,
            "final_max_g": applied_peak,
            "nominal_objective": nominal_objective,
            "hds_objective": applied_objective,
            "corrected_segments": int(sum(segment.corrected for segment in outcome.segments)),
            "inference_seconds": inference_seconds,
            "hds_seconds": correction_seconds,
            "total_predeployment_seconds": inference_seconds + correction_seconds,
        })
    accepted = np.asarray([row["accepted"] for row in rows], dtype=bool)
    return rows, {
        "checkpoint": str(checkpoint_path),
        "samples": len(rows),
        "accepted": int(accepted.sum()),
        "fallback": int((~accepted).sum()),
        "nominal_violation_rate_percent": float(100 * np.mean([row["nominal_max_g"] > 1e-8 for row in rows])),
        "nominal_max_g": float(np.max([row["nominal_max_g"] for row in rows])),
        "final_max_g_mean": _mean([row["final_max_g"] for row in rows]),
        "mean_corrected_segments": _mean([row["corrected_segments"] for row in rows]),
        "mean_hds_objective": _mean([row["hds_objective"] for row in rows]),
        "mean_hds_objective_change": _mean([row["hds_objective"] - row["nominal_objective"] for row in rows]),
        "mean_inference_seconds": inference_seconds,
        "mean_hds_seconds": _mean([row["hds_seconds"] for row in rows]),
        "mean_total_predeployment_seconds": _mean([row["total_predeployment_seconds"] for row in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("vdp", "penicillin"), required=True)
    parser.add_argument("--seed", type=int, default=20260771)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not args.baseline.is_file() or not args.candidate.is_file():
        raise FileNotFoundError("Both checkpoint paths must exist")
    args.output.mkdir(parents=True)
    problem = Problem(args.benchmark, torch.device("cpu"), args.seed, UnifiedConfig())
    states = np.asarray(problem.validation_states, dtype=float)
    np.save(args.output / "validation_states.npy", states)
    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for name, checkpoint in (("S-u", args.baseline), ("candidate", args.candidate)):
        rows, summary = _evaluate(problem, name, checkpoint, states)
        all_rows.extend(rows)
        summaries[name] = summary
    fields = sorted({key for row in all_rows for key in row})
    with (args.output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(all_rows)
    report = {
        "formal_protocol": False,
        "purpose": "validation-only exploratory selection; fixed test references were not read",
        "benchmark": args.benchmark,
        "seed": args.seed,
        "validation_samples": len(states),
        "objective_direction": "minimize terminal cost" if args.benchmark == "vdp" else "minimize J=-final_x3",
        "hds": {"integrator": "DOP853", "rtol": 1e-10, "atol": 1e-12, "peak_location": "constraint-derivative stationary events plus segment endpoints", "lambda_candidates": 31},
        "hds_statement": "continuous-time numerical audit evidence under the declared model and numerical settings; not a real-system absolute safety guarantee",
        "methods": summaries,
        "comparison": {
            "candidate_minus_S-u_mean_hds_objective": summaries["candidate"]["mean_hds_objective"] - summaries["S-u"]["mean_hds_objective"],
            "candidate_minus_S-u_corrected_segments": summaries["candidate"]["mean_corrected_segments"] - summaries["S-u"]["mean_corrected_segments"],
        },
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
