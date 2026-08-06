"""Aggregate the CSTR four-method comparison across multiple result roots."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

from aggregate_cstr_k10_projection_comparison import METHODS, load, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--expected-device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    roots = tuple(path.resolve() for path in args.input_roots)
    seeds = tuple(args.seeds)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    sources: dict[str, str] = {}
    for method in METHODS:
        for seed in seeds:
            matches = [
                root for root in roots
                if (root / method / "cstr" / f"seed{seed}" / "hds_test400" / "summary.json").exists()
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one source for {method} seed {seed}, found {matches}"
                )
            row = load(matches[0], method, seed)
            rows.append(row)
            sources[f"{method}:{seed}"] = str(matches[0])

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
            a = np.asarray([
                next(row[metric] for row in rows if row["method"] == left and row["seed"] == seed)
                for seed in seeds
            ])
            b = np.asarray([
                next(row[metric] for row in rows if row["method"] == right and row["seed"] == seed)
                for seed in seeds
            ])
            difference = b - a
            comparisons[metric][f"{right}_minus_{left}"] = {
                "mean_difference_percentage_points": float(difference.mean()),
                "right_wins_ties_losses": [
                    int(np.sum(b < a)),
                    int(np.sum(np.isclose(b, a, atol=1e-12, rtol=0.0))),
                    int(np.sum(b > a)),
                ],
                "paired_t_test_two_sided_p": float(stats.ttest_rel(b, a).pvalue),
                "wilcoxon_two_sided_p": float(stats.wilcoxon(b, a).pvalue),
                "seed_differences": {str(seed): float(value) for seed, value in zip(seeds, difference)},
            }

    report = {
        "protocol": "CSTR N=100/RK10, 200 S-u + 10 continuation epochs, frozen 400 reference",
        "seeds": seeds,
        "device": args.expected_device,
        "input_roots": [str(root) for root in roots],
        "sources": sources,
        "methods": aggregate,
        "paired_comparisons": comparisons,
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
        f"# CSTR 200+10 KKT projection comparison ({len(seeds)} seeds)", "",
        "| Method | HDS gap all-400 (%) | Delta vs S-u (pp) | Wins vs S-u | Qualified-286 gap (%) | Corrected segments |",
        "|---|---:|---:|---:|---:|---:|",
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
            f"| {method} | {item['hds_gap_all400_percent']['mean']:.5f} +/- "
            f"{item['hds_gap_all400_percent']['sample_sd']:.5f} | {delta:+.5f} | {wins} | "
            f"{item['hds_gap_qualified286_percent']['mean']:.5f} +/- "
            f"{item['hds_gap_qualified286_percent']['sample_sd']:.5f} | "
            f"{item['mean_corrected_segments']['mean']:.3f} |"
        )
    lines.extend([
        "",
        f"All {len(rows)} runs used {args.expected_device.upper()}, evaluated all 400 points, "
        "and had 400/400 accepted network policies with no fallback.",
    ])
    (output / f"aggregate_{len(seeds)}seeds_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
