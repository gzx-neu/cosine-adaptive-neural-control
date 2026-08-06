"""Build EAAI-oriented ablation artifacts from frozen, per-sample results.

The script does not retrain policies or re-run the deterministic solver.  It
aggregates the same frozen test cohorts used in the manuscript to isolate the
contribution of the pre-execution HDS--lambda correction.  In particular,
``HDS audit only`` means that an unsafe nominal sequence is dispatched rather
than corrected, under the same conservative acceptance margin used by VALC.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "kkt_collocation" / "results"
WRITING = ROOT / "论文写作"
MARGIN = 1e-6


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def selected_rows(problem: str) -> list[tuple[str, list[dict[str, str]], list[dict[str, str]]]]:
    if problem == "VDP":
        aggregate = read_json(RESULTS / "final_multiseed_vdp900_penalty_aggregate" / "summary.json")
    elif problem == "Penicillin":
        aggregate = read_json(RESULTS / "final_multiseed_penicillin400_penalty_aggregate" / "summary.json")
    else:
        directory = RESULTS / "cstr_multiseed_test1200_900_clean"
        rows = read_rows(directory / "per_sample.csv")
        output = []
        for seed in ("20260718", "20260725", "20260726"):
            group = [row for row in rows if row["training_seed"] == seed]
            output.append((seed, group, group))
        return output

    output = []
    for source in aggregate["source_directories"]:
        directory = ROOT / Path(source)
        rows = read_rows(directory / "per_sample.csv")
        raw = [row for row in rows if row["method"].startswith("Adaptive (") and "+ HDS-lambda" not in row["method"]]
        corrected = [row for row in rows if row["method"].startswith("Adaptive (") and "+ HDS-lambda" in row["method"]]
        if not raw or not corrected:
            raise ValueError(f"Could not locate the selected policy rows in {directory}")
        output.append((directory.name.rsplit("seed", 1)[-1], raw, corrected))
    return output


def correction_ablation(problem: str) -> dict:
    per_seed = {}
    for seed, raw_rows, corrected_rows in selected_rows(problem):
        if len(raw_rows) != len(corrected_rows):
            raise ValueError(f"Mismatched raw and corrected cohort sizes for {problem}, seed {seed}")
        peak_key = "nominal_hds_max_g" if "nominal_hds_max_g" in raw_rows[0] else "raw_hds_max_g"
        raw_peaks = np.asarray([float(row[peak_key]) for row in raw_rows])
        audit_only_accept = raw_peaks <= -MARGIN
        full_accept = np.asarray([as_bool(row["accepted"]) for row in corrected_rows])
        corrected_segments = np.asarray([float(row["corrected_segments"]) for row in corrected_rows])
        per_seed[seed] = {
            "samples": int(len(raw_rows)),
            "nominal_violation_rate_percent": float(100.0 * np.mean(raw_peaks > 0.0)),
            "audit_only_acceptance_percent": float(100.0 * np.mean(audit_only_accept)),
            "audit_only_dispatch_percent": float(100.0 * np.mean(~audit_only_accept)),
            "valc_acceptance_percent": float(100.0 * np.mean(full_accept)),
            "valc_dispatch_percent": float(100.0 * np.mean(~full_accept)),
            "corrected_segments_mean": float(np.mean(corrected_segments)),
        }
    metric_names = tuple(next(iter(per_seed.values())).keys())
    aggregate = {
        name: mean_sd([summary[name] for summary in per_seed.values()])
        for name in metric_names if name != "samples"
    }
    return {"per_seed": per_seed, "aggregate": aggregate, "samples": int(sum(item["samples"] for item in per_seed.values()))}


def gate_summary() -> dict:
    reports = [
        read_json(RESULTS / "eai_extension" / "gate_threshold_sensitivity.json"),
        read_json(RESULTS / "eai_extension" / "gate_vdp_20260752.json"),
        read_json(RESULTS / "eai_extension" / "gate_vdp_20260753.json"),
    ]
    output = {}
    for problem in ("VDP", "Penicillin"):
        default_triggers = []
        trigger_counts = []
        normalized_peaks = []
        for report in reports:
            item = report[problem]
            default = next(row for row in item["rows"]
                           if row["severe_rate_threshold"] == 0.025 and row["peak_threshold"] == 0.025)
            default_triggers.append(bool(default["trigger"]))
            trigger_counts.append(int(sum(bool(row["trigger"]) for row in item["rows"])))
            normalized_peaks.append(float(item["maximum_normalized_violation"]))
        output[problem] = {
            "training_seeds": 3,
            "validation_samples_per_seed": reports[0][problem]["validation_samples"],
            "default_threshold_pair": {"severe_rate": 0.025, "peak": 0.025},
            "default_trigger_by_seed": default_triggers,
            "trigger_counts_over_25_threshold_pairs": trigger_counts,
            "maximum_normalized_violation_by_seed": normalized_peaks,
        }
    return output


def margin_summary() -> dict:
    reports = [
        read_json(RESULTS / "hds_margin_calibration" / "summary.json"),
        read_json(RESULTS / "hds_margin_calibration_cstr_critical" / "summary.json"),
    ]
    output = {}
    for report in reports:
        for problem, item in report["benchmarks"].items():
            output[problem] = {
                "boundary_critical_trajectories": item["selected_boundary_critical_trajectories"],
                "accepted_trajectories": item["accepted_after_margin_correction"],
                "maximum_absolute_cross_audit_difference": item["max_absolute_cross_audit_difference"],
                "maximum_one_sided_cross_audit_increase": item["max_one_sided_cross_audit_increase"],
                "all_cross_audits_below_zero": item["all_cross_audits_below_zero"],
            }
    return output


def candidate_grid_summary() -> dict:
    report = read_json(RESULTS / "cstr_grid31_comparison_900" / "summary_grid31.json")
    return {
        "problem": "CSTR",
        "samples": report["samples"],
        "comparison": "frozen 21- versus 31-candidate grid",
        "changed_rows": report["comparison_with_grid21"]["changed_rows"],
        "aggregate_grid31": report["aggregate"],
    }


def tex_table(correction: dict[str, dict]) -> str:
    lines = [
        "% Auto-generated from frozen per-sample VALC results.",
        r"\begin{table*}[t]", r"\centering", r"\scriptsize",
        r"\caption{Contribution of HDS--$\lambda$ correction on the gate-selected policy. ``HDS audit only'' dispatches a nominal sequence whenever it fails the same conservative acceptance rule used by VALC. Values are mean $\pm$ sample standard deviation across three independent 400-point test cohorts.}",
        r"\label{tab:correction-ablation}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Problem & nominal violation & audit-only dispatch & VALC dispatch & corrected segments \\", r"\midrule",
    ]
    for problem in ("VDP", "Penicillin", "CSTR"):
        item = correction[problem]["aggregate"]
        lines.append(
            f"{problem} & "
            f"{item['nominal_violation_rate_percent']['mean']:.2f} $\\pm$ {item['nominal_violation_rate_percent']['sample_std']:.2f}\\% & "
            f"{item['audit_only_dispatch_percent']['mean']:.2f} $\\pm$ {item['audit_only_dispatch_percent']['sample_std']:.2f}\\% & "
            f"{item['valc_dispatch_percent']['mean']:.2f} $\\pm$ {item['valc_dispatch_percent']['sample_std']:.2f}\\% & "
            f"{item['corrected_segments_mean']['mean']:.3f} $\\pm$ {item['corrected_segments_mean']['sample_std']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def main() -> None:
    output = RESULTS / "eai_extension"
    output.mkdir(parents=True, exist_ok=True)
    correction = {problem: correction_ablation(problem) for problem in ("VDP", "Penicillin", "CSTR")}
    report = {
        "acceptance_margin": MARGIN,
        "correction_ablation": correction,
        "gate_threshold_sensitivity": gate_summary(),
        "candidate_grid_sensitivity": candidate_grid_summary(),
        "margin_cross_audit": margin_summary(),
        "boundary": "The correction ablation is a counterfactual deployment accounting on frozen cohorts. It does not execute a rejected audit-only sequence; it dispatches that instance to the offline optimizer.",
    }
    (output / "eai_extension_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    table = tex_table(correction)
    (WRITING / "table_correction_ablation.tex").write_text(table, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
