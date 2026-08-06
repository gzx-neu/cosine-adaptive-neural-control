"""Aggregate fixed-test-set ablations over independent training seeds.

The test/validation designs must remain fixed across directories.  Thus the
reported standard deviation quantifies only neural-training randomness, not a
mixture of training and test-set variation.  Outputs are both machine-readable
JSON and a paper-ready LaTex table with mean +/- sample standard deviation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _fmt(values: list[float], percent: bool = False, digits: int = 2) -> str:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return "--"
    scale = 100.0 if percent else 1.0
    mean = array.mean() * scale
    std = (array.std(ddof=1) if len(array) > 1 else 0.0) * scale
    suffix = r"\%" if percent else ""
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True,
                        help="Exactly three or more result directories containing summary.json.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--problem", choices=("vdp", "penicillin"), required=True)
    args = parser.parse_args()
    if len(args.inputs) < 3:
        raise ValueError("At least three independent training seeds are required.")
    summaries = []
    for directory in args.inputs:
        with (directory / "summary.json").open(encoding="utf-8") as handle:
            summary = json.load(handle)
        summaries.append({"directory": str(directory), "summary": summary})
    tests = {(item["summary"]["data_split"]["test_seed"], item["summary"]["config"]["test_samples"])
             for item in summaries}
    validations = {(item["summary"]["data_split"]["validation_seed"], item["summary"]["config"]["validation_samples"])
                   for item in summaries}
    if len(tests) != 1 or len(validations) != 1:
        raise ValueError("All seeds must use identical validation and test designs.")
    method_names = sorted(set.intersection(*(set(item["summary"]["methods"]) for item in summaries)))
    report: dict = {
        "problem": args.problem,
        "training_seeds": [item["summary"]["config"]["seed"] for item in summaries],
        "fixed_validation_design": list(validations)[0],
        "fixed_test_design": list(tests)[0],
        "adaptive_branches": [item["summary"]["adaptive_gate"]["selected_branch"] for item in summaries],
        "source_directories": [item["directory"] for item in summaries],
        "methods": {},
    }
    rate_key = "nominal_violation_rate" if args.problem == "vdp" else "raw_violation_rate"
    severe_key = "nominal_severe_violation_rate" if args.problem == "vdp" else "raw_severe_violation_rate"
    mean_violation_key = "mean_positive_nominal_violation" if args.problem == "vdp" else "mean_positive_raw_violation"
    max_raw_key = "max_nominal_hds_g" if args.problem == "vdp" else "max_raw_hds_g"
    safe_key = "accepted_max_hds_g" if args.problem == "vdp" else "max_applied_hds_g"
    objective_key = "mean_objective_change" if args.problem == "vdp" else "mean_product_change"
    for method in method_names:
        values = {key: [item["summary"]["methods"][method].get(key, np.nan) for item in summaries]
                  for key in (rate_key, severe_key, mean_violation_key, max_raw_key,
                              "accepted_rate", "fallback_rate", safe_key,
                              "mean_corrected_segments", "mean_abs_lambda_minus_one",
                              objective_key, "mean_inference_seconds", "mean_filter_seconds")}
        report["methods"][method] = {}
        for key, series in values.items():
            finite = np.asarray(series, dtype=float)
            finite = finite[np.isfinite(finite)]
            report["methods"][method][key] = {
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "sample_std": float(finite.std(ddof=1)) if len(finite) > 1 else (0.0 if len(finite) else np.nan),
                "per_seed": series,
            }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Method & raw violation & accepted & corrected segments & objective change & HDS time (s) & fallback \\",
             r"\midrule"]
    for method in method_names:
        data = report["methods"][method]
        lines.append("{} & {} & {} & {} & {} & {} & {} \\\\".format(
            method.replace("_", r"\_"), _fmt(data[rate_key]["per_seed"], True),
            _fmt(data["accepted_rate"]["per_seed"], True),
            _fmt(data["mean_corrected_segments"]["per_seed"]),
            _fmt(data[objective_key]["per_seed"], digits=4),
            _fmt(data["mean_filter_seconds"]["per_seed"], digits=4),
            _fmt(data["fallback_rate"]["per_seed"], True)))
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (args.output / "table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
