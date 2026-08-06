"""Aggregate VDP or penicillin K10 comparisons across multiple result roots."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

from aggregate_vdp_k10_projection_comparison import METHODS, _load, _summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("vdp", "penicillin"), required=True)
    parser.add_argument("--input-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--expected-device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()

    roots = tuple(path.resolve() for path in args.input_roots)
    seeds = tuple(args.seeds)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    sources = {}
    for method in METHODS:
        for seed in seeds:
            matches = [
                root for root in roots
                if (root / method / args.benchmark / f"seed{seed}" / "summary.json").exists()
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one source for {method} seed {seed}, found {matches}"
                )
            rows.append(_load(matches[0], args.benchmark, method, seed))
            sources[f"{method}:{seed}"] = str(matches[0])

    if not all(row["completed"] and row["device"] == args.expected_device for row in rows):
        raise RuntimeError("Completion/device invariant failed")
    if len({row["test_sha256"] for row in rows}) != 1:
        raise RuntimeError("Test cohorts differ")
    if not all(
        row["acceptance_rate_percent"] == 100.0 and row["fallback_rate_percent"] == 0.0
        for row in rows
    ):
        raise RuntimeError("Acceptance/fallback invariant failed")

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

    comparisons = {}
    for left, right in combinations(METHODS, 2):
        a = np.asarray([
            next(row["hds_gap_percent"] for row in rows
                 if row["method"] == left and row["seed"] == seed)
            for seed in seeds
        ])
        b = np.asarray([
            next(row["hds_gap_percent"] for row in rows
                 if row["method"] == right and row["seed"] == seed)
            for seed in seeds
        ])
        difference = b - a
        comparisons[f"{right}_minus_{left}"] = {
            "mean_gap_difference_percentage_points": float(difference.mean()),
            "sample_sd_difference": float(difference.std(ddof=1)),
            "right_wins_ties_losses": [
                int(np.sum(b < a)),
                int(np.sum(np.isclose(b, a, rtol=0.0, atol=1e-12))),
                int(np.sum(b > a)),
            ],
            "paired_t_test_two_sided_p": float(stats.ttest_rel(b, a).pvalue),
            "wilcoxon_two_sided_p": float(stats.wilcoxon(b, a).pvalue),
            "seed_differences": {str(seed): float(value) for seed, value in zip(seeds, difference)},
        }

    report = {
        "protocol": f"{args.benchmark} 200 S-u + 10 continuation; matched-400 HDS",
        "seeds": seeds,
        "device": args.expected_device,
        "test_sha256": rows[0]["test_sha256"],
        "input_roots": [str(root) for root in roots],
        "sources": sources,
        "methods": aggregate,
        "paired_hds_gap_comparisons": comparisons,
    }
    (output / f"aggregate_{len(seeds)}seeds_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (output / f"per_seed_{len(seeds)}seeds_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {args.benchmark} 200+10 comparison ({len(seeds)} seeds)", "",
        "| Method | HDS gap (%) | Delta vs S-u (pp) | Wins vs S-u | Nominal violation (%) | Corrected segments | HDS (ms/point) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = aggregate[method]
        if method == "supervised":
            delta, wins = 0.0, "-"
        else:
            comp = comparisons[f"{method}_minus_supervised"]
            delta = comp["mean_gap_difference_percentage_points"]
            wtl = comp["right_wins_ties_losses"]
            wins = f"{wtl[0]}/{sum(wtl)}"
        lines.append(
            f"| {method} | {item['hds_gap_percent']['mean']:.5f} +/- "
            f"{item['hds_gap_percent']['sample_sd']:.5f} | {delta:+.5f} | {wins} | "
            f"{item['nominal_violation_rate_percent']['mean']:.2f} | "
            f"{item['mean_corrected_segments']['mean']:.4f} | "
            f"{item['mean_hds_ms']['mean']:.2f} |"
        )
    lines.extend([
        "",
        f"All {len(rows)} runs used {args.expected_device.upper()}, evaluated the same 400 matched points, "
        "and had 100% acceptance with zero fallback.",
    ])
    (output / f"aggregate_{len(seeds)}seeds_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
