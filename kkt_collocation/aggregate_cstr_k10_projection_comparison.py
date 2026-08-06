"""Aggregate the paired 10-seed CSTR 200+10 projection comparison."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "kkt_collocation/results/multiseed10_cstr_k10_cuda_20260803_v1"
METHODS = ("supervised", "unprocessed", "linear_cosine", "standard_pcgrad")
BRANCH = {"supervised": "S-u", **{name: "S-u+K" for name in METHODS[1:]}}
SEEDS = tuple(range(20260771, 20260781))


def load(output: Path, method: str, seed: int) -> dict:
    directory = output / method / "cstr" / f"seed{seed}"
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "hds_test400/summary.json").read_text(encoding="utf-8"))
    result = summary["methods"][BRANCH[method]]
    training = result["training"]
    return {
        "method": method,
        "seed": seed,
        "device": config["reproducibility"]["device"],
        "completed": training["completed"],
        "hds_gap_all400_percent": result["hds_gap_all_400_percent"]["mean"],
        "hds_gap_qualified286_percent": result[
            "hds_gap_continuous_audit_qualified_reference_percent"
        ]["mean"],
        "nominal_violation_rate_percent": result["nominal_violation_rate_percent"],
        "mean_corrected_segments": result["mean_corrected_segments"],
        "accepted": result["accepted_network_samples"],
        "fallback": result["fallback_samples"],
        "training_seconds": training["seconds"],
        "mean_inference_ms": 1000.0 * result["mean_inference_seconds"],
        "mean_hds_ms": 1000.0 * result["mean_hds_seconds"],
        "mean_total_ms": 1000.0 * result["mean_total_predeployment_seconds"],
    }


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, float)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--expected-device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    seeds = tuple(args.seeds)
    rows = [load(output, method, seed) for method in METHODS for seed in seeds]
    if not all(row["completed"] and row["device"] == args.expected_device for row in rows):
        raise RuntimeError(f"Not every run completed on {args.expected_device}")
    if not all(row["accepted"] == 400 and row["fallback"] == 0 for row in rows):
        raise RuntimeError("Acceptance/fallback invariant failed")
    metrics = tuple(key for key in rows[0] if key not in ("method", "seed", "device", "completed"))
    aggregate = {
        method: {
            metric: summarize([row[metric] for row in rows if row["method"] == method])
            for metric in metrics
        }
        for method in METHODS
    }

    comparisons = {}
    for metric in ("hds_gap_all400_percent", "hds_gap_qualified286_percent"):
        comparisons[metric] = {}
        for left, right in combinations(METHODS, 2):
            a = np.asarray([next(row[metric] for row in rows if row["method"] == left and row["seed"] == seed) for seed in seeds])
            b = np.asarray([next(row[metric] for row in rows if row["method"] == right and row["seed"] == seed) for seed in seeds])
            difference = b - a
            comparisons[metric][f"{right}_minus_{left}"] = {
                "mean_difference_percentage_points": float(difference.mean()),
                "right_wins_ties_losses": [int(np.sum(b < a)), int(np.sum(np.isclose(b, a, atol=1e-12, rtol=0.0))), int(np.sum(b > a))],
                "paired_t_test_two_sided_p": float(stats.ttest_rel(b, a).pvalue),
                "wilcoxon_two_sided_p": float(stats.wilcoxon(b, a).pvalue),
                "seed_differences": {str(seed): float(value) for seed, value in zip(seeds, difference)},
            }

    report = {
        "protocol": "CSTR N=100/RK10, 200 S-u + 10 continuation epochs, frozen 400 reference",
        "seeds": seeds,
        "device": args.expected_device,
        "methods": aggregate,
        "paired_comparisons": comparisons,
    }
    (output / f"aggregate_{len(seeds)}seeds_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "per_seed_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    lines = [
        f"# CSTR 200+10 KKT projection comparison ({len(seeds)} seeds)", "",
        "| Method | HDS gap all-400 (%) | Delta vs S-u (pp) | Wins vs S-u | Qualified-286 gap (%) | Nominal violation (%) | Corrected segments |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = aggregate[method]
        if method == "supervised":
            delta, wins = 0.0, "-"
        else:
            comp = comparisons["hds_gap_all400_percent"][f"{method}_minus_supervised"]
            delta = comp["mean_difference_percentage_points"]
            wtl = comp["right_wins_ties_losses"]
            wins = f"{wtl[0]}/{sum(wtl)}"
        lines.append(
            f"| {method} | {item['hds_gap_all400_percent']['mean']:.5f} +/- {item['hds_gap_all400_percent']['sample_sd']:.5f} "
            f"| {delta:+.5f} | {wins} "
            f"| {item['hds_gap_qualified286_percent']['mean']:.5f} "
            f"| {item['nominal_violation_rate_percent']['mean']:.2f} "
            f"| {item['mean_corrected_segments']['mean']:.3f} |"
        )
    lines.extend(["", f"All {len(rows)} runs used {args.expected_device.upper()} and evaluated all 400 points; the qualified-reference subset contains 286 points."])
    (output / f"aggregate_{len(seeds)}seeds_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
