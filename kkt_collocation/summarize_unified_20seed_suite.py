"""Summarize completed unified-ablation seeds without mixing protocols."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(20260771, 20260791))
BENCHMARKS = ("vdp", "penicillin", "cstr")
METHODS = ("S-u", "S-uJ", "S+K", "K-only")


def _finite(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _mean_std(values: list[float | None]) -> dict:
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)], float)
    if not len(array):
        return {"count": 0, "mean": None, "std": None, "median": None, "q25": None, "q75": None}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def _paired_bootstrap(rows: list[dict], baseline: str, candidate: str, key: str) -> dict:
    by_seed: dict[int, dict[str, float]] = {}
    for row in rows:
        value = _finite(row.get(key))
        if value is not None:
            by_seed.setdefault(int(row["Seed"]), {})[row["Method"]] = value
    pairs = [
        (seed, values[candidate] - values[baseline])
        for seed, values in sorted(by_seed.items())
        if baseline in values and candidate in values
    ]
    if not pairs:
        return {"paired_count": 0, "mean_candidate_minus_baseline": None, "ci95_percentile_bootstrap": [None, None], "seed_differences": []}
    differences = np.asarray([value for _, value in pairs], float)
    rng = np.random.default_rng(20260802)
    boot = differences[rng.integers(0, len(differences), size=(20000, len(differences)))].mean(axis=1)
    return {
        "paired_count": int(len(differences)),
        "mean_candidate_minus_baseline": float(differences.mean()),
        "ci95_percentile_bootstrap": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "candidate_better_count": int(np.sum(differences < 0)),
        "candidate_tied_count": int(np.sum(differences == 0)),
        "candidate_worse_count": int(np.sum(differences > 0)),
        "seed_differences": [{"seed": seed, "difference": value} for seed, value in pairs],
    }


def _load_vdp_or_pen(root: Path, benchmark: str, seed: int) -> list[dict]:
    path = root / benchmark / f"seed{seed}" / "summary.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for method in METHODS:
        training = data["training"][method]
        deployment = data["deployment"][method]
        samples = int(deployment.get("samples", 0))
        acceptance = _finite(deployment.get("continuous_time_audit_acceptance_rate"))
        accepted = None if acceptance is None else int(round(samples * acceptance))
        fallback = None if accepted is None else samples - accepted
        rows.append({
            "Benchmark": benchmark,
            "Method": method,
            "Seed": seed,
            "Training protocol": "210 supervised" if method in ("S-u", "S-uJ") else ("200 supervised + 10 KKT" if method == "S+K" else "210 direct KKT"),
            "Reference subset": "fixed-50 cold-start NLP subset of fixed-400 test",
            "Nominal gap (%)": _finite(deployment.get("nominal_gap_percent")),
            "HDS gap (%)": _finite(deployment.get("hds_gap_percent")),
            "Qualified-reference HDS gap (%)": None,
            "Nominal violation (%)": None if deployment.get("nominal_violation_rate") is None else 100.0 * float(deployment["nominal_violation_rate"]),
            "Accepted": accepted,
            "Fallback": fallback,
            "Corrected segments": _finite(deployment.get("mean_corrected_segments")),
            "HDS objective change": _finite(deployment.get("mean_hds_objective_change")),
            "HDS seconds": _finite(deployment.get("mean_audit_seconds")),
            "KKT residual": _finite(training.get("final_kkt_residual")),
            "Training stable": bool(training["completed"]),
            "Failure reason": training.get("failure_reason"),
        })
    return rows


def _load_cstr(root: Path, seed: int) -> list[dict]:
    path = root / "cstr" / f"seed{seed}" / "hds_test400" / "summary.json"
    training_path = root / "cstr" / f"seed{seed}" / "training_summary.json"
    if not training_path.exists():
        return []
    training_data = json.loads(training_path.read_text(encoding="utf-8"))["methods"]
    evaluation = json.loads(path.read_text(encoding="utf-8"))["methods"] if path.exists() else {}
    rows: list[dict] = []
    for method in METHODS:
        training = training_data[method]
        deployment = evaluation.get(method, {})
        all_gap = deployment.get("hds_gap_all_400_percent", {})
        qualified_gap = deployment.get("hds_gap_continuous_audit_qualified_reference_percent", {})
        nominal_gap = deployment.get("nominal_gap_all_400_percent", {})
        rows.append({
            "Benchmark": "cstr",
            "Method": method,
            "Seed": seed,
            "Training protocol": "210 supervised" if method in ("S-u", "S-uJ") else ("200 supervised + 10 KKT" if method == "S+K" else "210 direct KKT"),
            "Reference subset": "fixed-400 cold-start NLP; qualified column uses fixed 286-reference subset",
            "Nominal gap (%)": _finite(nominal_gap.get("mean")),
            "HDS gap (%)": _finite(all_gap.get("mean")),
            "Qualified-reference HDS gap (%)": _finite(qualified_gap.get("mean")),
            "Nominal violation (%)": _finite(deployment.get("nominal_violation_rate_percent")),
            "Accepted": deployment.get("accepted_network_samples"),
            "Fallback": deployment.get("fallback_samples"),
            "Corrected segments": _finite(deployment.get("mean_corrected_segments")),
            "HDS objective change": _finite(deployment.get("mean_hds_objective_change")),
            "HDS seconds": _finite(deployment.get("mean_hds_seconds")),
            "KKT residual": _finite(training.get("kkt_residual")),
            "Training stable": bool(training["completed"]),
            "Failure reason": training.get("failure_reason"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "kkt_collocation/results/unified_su_suj_sk_konly_20seeds_v1",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    input_root = args.input_root
    output = args.output or input_root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for seed in SEEDS:
        rows.extend(_load_vdp_or_pen(input_root, "vdp", seed))
        rows.extend(_load_vdp_or_pen(input_root, "penicillin", seed))
        rows.extend(_load_cstr(input_root, seed))
    fields = [
        "Benchmark", "Method", "Seed", "Training protocol", "Reference subset",
        "Nominal gap (%)", "HDS gap (%)", "Qualified-reference HDS gap (%)",
        "Nominal violation (%)", "Accepted", "Fallback", "Corrected segments",
        "HDS objective change", "HDS seconds", "KKT residual", "Training stable", "Failure reason",
    ]
    with (output / "per_seed_table.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows: list[dict] = []
    for benchmark in BENCHMARKS:
        for method in METHODS:
            subset = [row for row in rows if row["Benchmark"] == benchmark and row["Method"] == method]
            gap = _mean_std([row["HDS gap (%)"] for row in subset])
            qualified = _mean_std([row["Qualified-reference HDS gap (%)"] for row in subset])
            corrected = _mean_std([row["Corrected segments"] for row in subset])
            aggregate_rows.append({
                "Benchmark": benchmark,
                "Method": method,
                "Seeds completed": len(subset),
                "Stable trainings": sum(row["Training stable"] for row in subset),
                "HDS gap finite seeds": gap["count"],
                "HDS gap mean (%)": gap["mean"],
                "HDS gap std (%)": gap["std"],
                "HDS gap median (%)": gap["median"],
                "HDS gap IQR low (%)": gap["q25"],
                "HDS gap IQR high (%)": gap["q75"],
                "Qualified HDS gap mean (%)": qualified["mean"],
                "Corrected segments mean": corrected["mean"],
                "Accepted total": sum(int(row["Accepted"] or 0) for row in subset),
                "Fallback total": sum(int(row["Fallback"] or 0) for row in subset),
            })
    aggregate_fields = list(aggregate_rows[0])
    with (output / "aggregate_table.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    paired = {}
    for benchmark in BENCHMARKS:
        subset = [row for row in rows if row["Benchmark"] == benchmark]
        paired[benchmark] = {
            "S+K_vs_S-uJ_HDS_gap_percent": _paired_bootstrap(subset, "S-uJ", "S+K", "HDS gap (%)"),
            "S+K_vs_S-u_HDS_gap_percent": _paired_bootstrap(subset, "S-u", "S+K", "HDS gap (%)"),
            "S+K_vs_S-uJ_corrected_segments": _paired_bootstrap(subset, "S-uJ", "S+K", "Corrected segments"),
        }
        if benchmark == "cstr":
            paired[benchmark]["S+K_vs_S-uJ_qualified_HDS_gap_percent"] = _paired_bootstrap(
                subset, "S-uJ", "S+K", "Qualified-reference HDS gap (%)"
            )
    (output / "paired_comparisons.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")

    completed = {
        benchmark: sorted({int(row["Seed"]) for row in rows if row["Benchmark"] == benchmark})
        for benchmark in BENCHMARKS
    }
    summary = {
        "formal_seed_set": SEEDS,
        "completed_seeds": completed,
        "complete_20_seed_suite": all(tuple(completed[benchmark]) == SEEDS for benchmark in BENCHMARKS),
        "gap_aggregation_rule": "Only finite accepted-network-policy gaps are aggregated. Training failures and optimizer fallback are reported separately and never substituted into neural-policy means.",
        "cross_benchmark_rule": "VDP/penicillin fixed-50 reference gaps and CSTR fixed-400/286 gaps are never pooled into one mean.",
        "hds_statement": "continuous-time numerical audit evidence under declared model/numerics; not a real-system absolute safety guarantee",
        "tables": aggregate_rows,
        "paired_comparisons": paired,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = [
        "# Unified 20-seed ablation summary",
        "",
        f"Suite complete: **{summary['complete_20_seed_suite']}**.",
        "",
        "The three benchmarks retain separate reference protocols. VDP and penicillin use their fixed 50-point matched cold-start NLP subsets; CSTR reports both all 400 cold-start references and the fixed 286-reference continuous-audit-qualified subset.",
        "",
        "K-only numerical failures or optimizer fallback are not neural-policy objective gaps and are excluded from finite-gap means.",
    ]
    (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
