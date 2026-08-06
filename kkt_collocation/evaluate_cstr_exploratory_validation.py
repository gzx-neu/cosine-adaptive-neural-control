"""Validation-only CSTR deployment comparison for exploratory continuations.

This driver never reads the frozen all-400 test reference.  It creates and
saves one independent LHS validation cohort, evaluates named checkpoints with
adaptive DOP853/event HDS, and compares their corrected physical objectives
on exactly those common initial states.  There is intentionally no NLP gap in
this exploratory selector: cold-start test references remain untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from evaluate_unified_economou_cstr_n100_hds import _mean
from run_economou_cstr_supervised_hds import PolicyValue, lhs_states
from run_economou_cstr_two_stage_vs_kkt_only import ROOT, _audit_worker
from screen_economou_cstr_30x30 import EconomouScreenConfig


def _load_cstr() -> EconomouScreenConfig:
    labels = ROOT / "kkt_collocation/results/economou_cstr_reduced_kkt_n100_rk10_margin0_ca030_050_t410_420"
    raw = json.loads((labels / "summary.json").read_text(encoding="utf-8"))["config"]
    for key in ("ti_bounds_K", "flow_bounds", "ca_initial_range", "temperature_initial_range_K"):
        raw[key] = tuple(raw[key])
    return EconomouScreenConfig(**raw)


def _controls(checkpoint_path: Path, cstr: EconomouScreenConfig, states: np.ndarray) -> tuple[np.ndarray, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = PolicyValue(cstr).eval()
    model.load_state_dict(checkpoint["model"])
    mean = np.asarray(checkpoint["state_mean"], dtype=float)
    std = np.asarray(checkpoint["state_std"], dtype=float)
    features = torch.tensor((states[:, [0, 2]] - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(10):
            model(features)
        started = time.perf_counter()
        _, controls = model(features)
    return controls.numpy(), (time.perf_counter() - started) / len(states)


def _evaluate(name: str, checkpoint: Path, cstr: EconomouScreenConfig, states: np.ndarray, workers: int) -> tuple[list[dict], dict]:
    controls, inference_seconds = _controls(checkpoint, cstr, states)
    payload = [(i, states[i].tolist(), controls[i].tolist(), asdict(cstr), 31) for i in range(len(states))]
    context = mp.get_context("spawn")
    if workers == 1:
        rows = [_audit_worker(item) for item in payload]
    else:
        with context.Pool(workers) as pool:
            rows = list(pool.imap_unordered(_audit_worker, payload, chunksize=1))
    rows.sort(key=lambda row: row["index"])
    for row in rows:
        row["method"] = name
        row["inference_seconds"] = inference_seconds
        row["total_predeployment_seconds"] = inference_seconds + row["hds_seconds"]
    accepted = np.asarray([row["accepted"] for row in rows], dtype=bool)
    summary = {
        "checkpoint": str(checkpoint),
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
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.count < 1 or args.workers < 1:
        raise ValueError("--count and --workers must be positive")
    if not args.baseline.is_file() or not args.candidate.is_file():
        raise FileNotFoundError("Both checkpoint paths must exist")
    args.output.mkdir(parents=True)
    cstr = _load_cstr()
    states = lhs_states(cstr, args.count, args.seed)
    np.save(args.output / "validation_initial_states.npy", states)
    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for name, checkpoint in (("S-u", args.baseline), ("candidate", args.candidate)):
        rows, summary = _evaluate(name, checkpoint, cstr, states, args.workers)
        all_rows.extend(rows)
        summaries[name] = summary
    fields = sorted({key for row in all_rows for key in row})
    with (args.output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(all_rows)
    comparison = {
        "candidate_minus_S-u_mean_hds_objective": summaries["candidate"]["mean_hds_objective"] - summaries["S-u"]["mean_hds_objective"],
        "candidate_minus_S-u_corrected_segments": summaries["candidate"]["mean_corrected_segments"] - summaries["S-u"]["mean_corrected_segments"],
    }
    report = {
        "formal_protocol": False,
        "purpose": "validation-only exploratory selection; never a frozen all-400/286 test result",
        "validation": {"count": args.count, "lhs_seed": args.seed, "states_file": "validation_initial_states.npy"},
        "hds": {"integrator": "DOP853", "rtol": 1e-10, "atol": 1e-12, "peak_location": "constraint-derivative stationary events plus segment endpoints", "lambda_candidates": 31},
        "hds_statement": "continuous-time numerical audit evidence under the declared model and numerical settings; not a real-system absolute safety guarantee",
        "methods": summaries,
        "comparison": comparison,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
