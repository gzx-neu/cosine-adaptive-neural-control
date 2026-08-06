"""HDS evaluation of the completed all-label CSTR S and S+K checkpoints.

The 400 reference controls are independent cold-start N=100/RK10 NLP
solutions.  Objectives for both references and corrected neural controls are
recomputed with the declared high-accuracy DOP853 model.  Since the reference
transcription has node_margin=0, the report retains both all-reference gaps
and the subset whose supplied event-located audit is non-positive.
"""
from __future__ import annotations

import csv
import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from run_economou_cstr_supervised_hds import PolicyValue, trajectory_objective
from run_economou_cstr_two_stage_vs_kkt_only import ROOT, _audit_worker, _mean, _std
from screen_economou_cstr_30x30 import EconomouScreenConfig

TRAIN = ROOT / "kkt_collocation/results/economou_cstr_n100_all900_s_sk_konly_training_seed20260771"
REF = ROOT / "kkt_collocation/results/economou_cstr_reduced_kkt_n100_test400_lhs_margin0"
OUT = ROOT / "kkt_collocation/results/economou_cstr_n100_test400_hds_s_sk"
GRID = 31
WORKERS = 4


def reference_worker(payload):
    index, state, controls, config, supplied_event_peak = payload
    cfg = EconomouScreenConfig(**config)
    started = time.perf_counter()
    objective = trajectory_objective(np.asarray(state, float), np.asarray(controls, float), cfg)
    return {"index": index, "high_fidelity_objective": objective,
            "supplied_event_located_max_g": float(supplied_event_peak),
            "continuous_audit_qualified": bool(supplied_event_peak <= 1e-8),
            "reference_objective_seconds": time.perf_counter() - started}


def stats(values):
    data = np.asarray(values, float); data = data[np.isfinite(data)]
    if not len(data): return {"count": 0, "mean": None, "std": None, "median": None, "p95": None, "max": None}
    return {"count": int(len(data)), "mean": float(data.mean()), "std": float(data.std()),
            "median": float(np.median(data)), "p95": float(np.quantile(data, .95)), "max": float(data.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--methods", nargs="+", default=("S", "S+K"))
    parser.add_argument("--test-stop", type=int, default=None,
                        help="Exploratory leading subset size; omit for all 400.")
    args = parser.parse_args()
    train, output_dir = args.train, args.output
    if output_dir.exists(): raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    raw_config = json.loads((REF / "summary.json").read_text(encoding="utf-8"))["config"]
    for key in ("ti_bounds_K", "flow_bounds", "ca_initial_range", "temperature_initial_range_K"):
        raw_config[key] = tuple(raw_config[key])
    cstr = EconomouScreenConfig(**raw_config)
    label_rows = [json.loads(line) for line in (REF / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    label_rows.sort(key=lambda item: int(item["index"]))
    if len(label_rows) != 400 or not all(item.get("success") for item in label_rows):
        raise ValueError("Expected 400 successful frozen cold-start labels")
    if args.test_stop is not None:
        if not 1 <= args.test_stop <= 400:
            raise ValueError("--test-stop must lie in [1,400]")
        label_rows = label_rows[:args.test_stop]
    test = np.asarray([row["initial_state"] for row in label_rows], float)
    context = mp.get_context("spawn")
    ref_payload = [(i, row["initial_state"], row["controls"], asdict(cstr), row["event_located_max_g"])
                   for i, row in enumerate(label_rows)]
    with context.Pool(WORKERS) as pool:
        references = list(pool.imap_unordered(reference_worker, ref_payload, chunksize=1))
    references.sort(key=lambda item: item["index"])
    reference_map = {row["index"]: row for row in references}
    with (output_dir / "cold_reference_high_fidelity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=references[0].keys()); writer.writeheader(); writer.writerows(references)

    output = {"protocol": f"Frozen leading {len(label_rows)}-point subset of the independent 400-point LHS cold-start reference; exploratory screen only.",
              "reference": {"cold_start_solver_success_count": len(label_rows),
                            "high_fidelity_objective_recomputed": True,
                            "continuous_audit_qualified_count": int(sum(row["continuous_audit_qualified"] for row in references)),
                            "note": "The qualified subset follows the supplied event-located audit of the node_margin=0 discrete reference controls."},
              "methods": {},
              "hds_statement": "Continuous-time numerical audit evidence under the declared model and numerical settings; not a real-system absolute safety guarantee."}
    for name in args.methods:
        checkpoint = torch.load(train / f"{name}.pth", map_location="cpu", weights_only=False)
        model = PolicyValue(cstr).eval()
        model.load_state_dict(checkpoint["model"])
        mean = np.asarray(checkpoint["state_mean"], float); std = np.asarray(checkpoint["state_std"], float)
        inputs = torch.tensor((test[:, [0, 2]] - mean) / std, dtype=torch.float32)
        with torch.no_grad():
            for _ in range(10): model(inputs)
            started = time.perf_counter(); _, predicted = model(inputs)
            inference_per_sample = (time.perf_counter() - started) / len(test)
        controls = predicted.numpy()
        payload = [(i, test[i].tolist(), controls[i].tolist(), asdict(cstr), GRID) for i in range(len(test))]
        with context.Pool(WORKERS) as pool:
            rows = list(pool.imap_unordered(_audit_worker, payload, chunksize=1))
        rows.sort(key=lambda item: item["index"])
        for row in rows:
            ref = reference_map[row["index"]]
            denominator = max(abs(ref["high_fidelity_objective"]), 1e-12)
            row["reference_high_fidelity_objective"] = ref["high_fidelity_objective"]
            row["reference_continuous_audit_qualified"] = ref["continuous_audit_qualified"]
            row["relative_nominal_gap_percent"] = 100 * (row["nominal_objective"] - ref["high_fidelity_objective"]) / denominator
            row["relative_hds_gap_percent"] = (100 * (row["hds_objective"] - ref["high_fidelity_objective"]) / denominator
                                                if row["accepted"] else np.nan)
            row["relative_hds_gap_qualified_reference_percent"] = (row["relative_hds_gap_percent"]
                                                                       if row["accepted"] and ref["continuous_audit_qualified"] else np.nan)
            row["inference_seconds"] = inference_per_sample
            row["total_predeployment_seconds"] = inference_per_sample + row["hds_seconds"]
        with (output_dir / f"{name}_per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = sorted({key for row in rows for key in row}); writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        hds_all = [row["relative_hds_gap_percent"] for row in rows]
        hds_qualified = [row["relative_hds_gap_qualified_reference_percent"] for row in rows]
        output["methods"][name] = {
            "accepted_network_samples": int(sum(row["accepted"] for row in rows)),
            "fallback_samples": int(sum(row["fallback"] for row in rows)),
            "nominal_violation_rate_percent": float(100 * np.mean([row["nominal_max_g"] > 1e-8 for row in rows])),
            "nominal_max_g": float(np.max([row["nominal_max_g"] for row in rows])),
            "final_max_g_mean": _mean([row["final_max_g"] for row in rows]),
            "mean_corrected_segments": _mean([row["corrected_segments"] for row in rows]),
            "mean_hds_objective_change": _mean([row["hds_objective"] - row["nominal_objective"] for row in rows]),
            "hds_gap_all_400_percent": stats(hds_all),
            "hds_gap_continuous_audit_qualified_reference_percent": stats(hds_qualified),
            "mean_inference_seconds": inference_per_sample,
            "mean_hds_seconds": _mean([row["hds_seconds"] for row in rows]),
            "mean_total_predeployment_seconds": _mean([row["total_predeployment_seconds"] for row in rows]),
        }
    (output_dir / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    table = ["| Method | HDS gap, all 400 (%) | HDS gap, qualified reference subset (%) | Corrected segments | Accepted / fallback |",
             "|---|---:|---:|---:|---:|"]
    for method, result in output["methods"].items():
        table.append(f"| {method} | {result['hds_gap_all_400_percent']['mean']:.4f} | {result['hds_gap_continuous_audit_qualified_reference_percent']['mean']:.4f} | {result['mean_corrected_segments']:.2f} | {result['accepted_network_samples']} / {result['fallback_samples']} |")
    (output_dir / "summary_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))

if __name__ == "__main__": main()
