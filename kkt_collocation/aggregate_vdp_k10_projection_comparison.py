"""Aggregate the paired 10-seed VDP 200+10 projection comparison."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "kkt_collocation/results/multiseed10_vdp_k10_cuda_20260803_v1"
METHODS = ("supervised", "unprocessed", "linear_cosine", "standard_pcgrad")
BRANCH = {
    "supervised": "S-u",
    "unprocessed": "S-u+K",
    "linear_cosine": "S-u+K",
    "standard_pcgrad": "S-u+K",
}
SEEDS = tuple(range(20260771, 20260781))


def _load(output: Path, benchmark: str, method: str, seed: int) -> dict:
    directory = output / method / benchmark / f"seed{seed}"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    branch = BRANCH[method]
    deployment = summary["deployment"][branch]
    training = summary["training"][branch]
    return {
        "method": method,
        "seed": seed,
        "device": config["device"],
        "test_sha256": config["test_sha256"],
        "completed": training["completed"],
        "training_seconds": training["train_seconds"],
        "hds_gap_percent": deployment["hds_gap_percent"],
        "nominal_gap_percent": deployment["nominal_gap_percent"],
        "nominal_violation_rate_percent": 100.0 * deployment["nominal_violation_rate"],
        "mean_corrected_segments": deployment["mean_corrected_segments"],
        "acceptance_rate_percent": 100.0 * deployment["continuous_time_audit_acceptance_rate"],
        "fallback_rate_percent": 100.0 * deployment["offline_optimizer_fallback_rate"],
        "final_max_g": deployment["final_max_g"],
        "mean_inference_ms": 1000.0 * deployment["mean_inference_seconds"],
        "mean_hds_ms": 1000.0 * deployment["mean_audit_seconds"],
        "mean_total_ms": 1000.0 * deployment["mean_total_seconds"],
    }


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("vdp", "penicillin"), default="vdp")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--additional-output", type=Path)
    parser.add_argument("--additional-seeds", nargs="+", type=int)
    parser.add_argument("--aggregate-output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    seeds = tuple(args.seeds)
    sources = [(output, seeds)]
    if args.additional_output is not None or args.additional_seeds is not None:
        if args.additional_output is None or args.additional_seeds is None:
            raise ValueError("--additional-output and --additional-seeds must be supplied together")
        sources.append((args.additional_output.resolve(), tuple(args.additional_seeds)))
    all_seeds = tuple(seed for _, source_seeds in sources for seed in source_seeds)
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Seed sets overlap")
    rows = [
        _load(source, args.benchmark, method, seed)
        for source, source_seeds in sources
        for method in METHODS
        for seed in source_seeds
    ]
    aggregate_output = (args.aggregate_output or output).resolve()
    aggregate_output.mkdir(parents=True, exist_ok=True)
    devices = {row["device"] for row in rows}
    if not all(row["completed"] for row in rows) or len(devices) != 1:
        raise RuntimeError("Not every run completed on the same device")
    if len({row["test_sha256"] for row in rows}) != 1:
        raise RuntimeError("Test cohorts are not identical")

    metrics = (
        "hds_gap_percent", "nominal_gap_percent", "nominal_violation_rate_percent",
        "mean_corrected_segments", "acceptance_rate_percent", "fallback_rate_percent",
        "final_max_g", "training_seconds", "mean_inference_ms", "mean_hds_ms",
        "mean_total_ms",
    )
    aggregate = {
        method: {
            metric: _summary([row[metric] for row in rows if row["method"] == method])
            for metric in metrics
        }
        for method in METHODS
    }

    comparisons: dict[str, dict] = {}
    for left, right in combinations(METHODS, 2):
        a = np.asarray([next(row["hds_gap_percent"] for row in rows
                             if row["method"] == left and row["seed"] == seed)
                        for seed in all_seeds])
        b = np.asarray([next(row["hds_gap_percent"] for row in rows
                             if row["method"] == right and row["seed"] == seed)
                        for seed in all_seeds])
        difference = b - a
        wilcoxon = stats.wilcoxon(b, a, alternative="two-sided", zero_method="wilcox")
        comparisons[f"{right}_minus_{left}"] = {
            "mean_gap_difference_percentage_points": float(difference.mean()),
            "sample_sd_difference": float(difference.std(ddof=1)),
            "right_wins_ties_losses": [
                int(np.sum(b < a)), int(np.sum(np.isclose(b, a, rtol=0.0, atol=1e-12))),
                int(np.sum(b > a)),
            ],
            "paired_t_test_two_sided_p": float(stats.ttest_rel(b, a).pvalue),
            "wilcoxon_two_sided_p": float(wilcoxon.pvalue),
            "seed_differences": {str(seed): float(value) for seed, value in zip(all_seeds, difference)},
        }

    report = {
        "protocol": f"{args.benchmark} 200 S-u epochs + 10 KKT continuation epochs; matched-400 HDS",
        "seeds": all_seeds,
        "device": next(iter(devices)),
        "all_same_test_sha256": rows[0]["test_sha256"],
        "methods": aggregate,
        "paired_hds_gap_comparisons": comparisons,
    }
    (aggregate_output / f"aggregate_{len(all_seeds)}seeds_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (aggregate_output / "per_seed_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {args.benchmark} 200+10 KKT projection comparison ({len(all_seeds)} seeds)", "",
        "| Method | HDS gap (%) | Delta vs S-u (pp) | Wins vs S-u | Nominal violation (%) | Corrected segments | Train (s) | HDS (ms/point) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = aggregate[method]
        if method == "supervised":
            difference, wins = 0.0, "-"
        else:
            comparison = comparisons[f"{method}_minus_supervised"]
            difference = comparison["mean_gap_difference_percentage_points"]
            wtl = comparison["right_wins_ties_losses"]
            wins = f"{wtl[0]}/{sum(wtl)}"
        lines.append(
            f"| {method} | {item['hds_gap_percent']['mean']:.5f} +/- {item['hds_gap_percent']['sample_sd']:.5f} "
            f"| {difference:+.5f} | {wins} "
            f"| {item['nominal_violation_rate_percent']['mean']:.2f} "
            f"| {item['mean_corrected_segments']['mean']:.4f} "
            f"| {item['training_seconds']['mean']:.2f} "
            f"| {item['mean_hds_ms']['mean']:.2f} |"
        )
    lines.extend(["", f"All {len(rows)} runs used {next(iter(devices)).upper()}, evaluated all 400 matched test points, and had 100% acceptance with zero fallback."])
    (aggregate_output / f"aggregate_{len(all_seeds)}seeds_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
