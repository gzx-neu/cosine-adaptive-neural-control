"""Evaluate one unified CSTR seed on the frozen 400-point HDS protocol.

The network controls are always audited afresh.  The cold-start NLP controls'
high-accuracy DOP853 objectives are immutable benchmark data, so this driver
reuses the previously computed 400-row reference cache after validating its
indices and the expected 286-row continuous-audit-qualified subset.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kkt_collocation.run_economou_cstr_supervised_hds import PolicyValue
from kkt_collocation.run_economou_cstr_two_stage_vs_kkt_only import ROOT, _audit_worker, _mean
from kkt_collocation.screen_economou_cstr_30x30 import EconomouScreenConfig


REF = ROOT / "kkt_collocation/results/economou_cstr_reduced_kkt_n100_test400_lhs_margin0"
REFERENCE_CACHE = ROOT / "kkt_collocation/results/economou_cstr_n100_test400_hds_s200_sk10_fair/cold_reference_high_fidelity.csv"
FORMAL_METHODS = ("S-u", "S-uJ", "S+K", "K-only")
EXPLORATORY_METHODS = FORMAL_METHODS + ("S-u+K",)
GRID = 31


def stats(values) -> dict:
    data = np.asarray(values, float)
    data = data[np.isfinite(data)]
    if not len(data):
        return {"count": 0, "mean": None, "std": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(data)),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "median": float(np.median(data)),
        "p95": float(np.quantile(data, 0.95)),
        "max": float(data.max()),
    }


def _load_reference_cache(label_rows: list[dict]) -> list[dict]:
    if not REFERENCE_CACHE.exists():
        raise FileNotFoundError(
            f"Missing immutable high-fidelity reference cache: {REFERENCE_CACHE}"
        )
    with REFERENCE_CACHE.open(encoding="utf-8") as handle:
        cached = list(csv.DictReader(handle))
    if len(cached) != 400:
        raise ValueError(f"Expected 400 cached reference rows, got {len(cached)}")
    rows: list[dict] = []
    for expected_index, (cache, label) in enumerate(zip(cached, label_rows)):
        index = int(cache["index"])
        if index != expected_index or int(label["index"]) != expected_index:
            raise ValueError("Reference cache indices do not match frozen records.jsonl")
        supplied_peak = float(cache["supplied_event_located_max_g"])
        if not np.isclose(supplied_peak, float(label["event_located_max_g"]), rtol=0.0, atol=1e-12):
            raise ValueError(f"Reference cache event peak mismatch at index {index}")
        rows.append({
            "index": index,
            "high_fidelity_objective": float(cache["high_fidelity_objective"]),
            "supplied_event_located_max_g": supplied_peak,
            "continuous_audit_qualified": supplied_peak <= 1e-8,
            "reference_objective_seconds": float(cache.get("reference_objective_seconds", "nan")),
        })
    qualified = sum(row["continuous_audit_qualified"] for row in rows)
    if qualified != 286:
        raise ValueError(f"Expected 286 continuous-audit-qualified references, got {qualified}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--methods", nargs="+", choices=EXPLORATORY_METHODS, default=FORMAL_METHODS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke-samples", type=int, default=None)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    train = args.train
    output_dir = args.output or train / "hds_test400"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation directory: {output_dir}")
    output_dir.mkdir(parents=True)

    raw_config = json.loads((REF / "summary.json").read_text(encoding="utf-8"))["config"]
    for key in ("ti_bounds_K", "flow_bounds", "ca_initial_range", "temperature_initial_range_K"):
        raw_config[key] = tuple(raw_config[key])
    cstr = EconomouScreenConfig(**raw_config)
    if cstr.node_margin != 0 or cstr.zoh_steps != 100 or cstr.substeps_per_zoh != 10:
        raise ValueError("Frozen reference is not the required N=100/RK10/node_margin=0 protocol")
    label_rows = [
        json.loads(line)
        for line in (REF / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    label_rows.sort(key=lambda item: int(item["index"]))
    if len(label_rows) != 400 or not all(item.get("success") for item in label_rows):
        raise ValueError("Expected 400 successful frozen cold-start NLP references")
    references = _load_reference_cache(label_rows)
    reference_map = {row["index"]: row for row in references}
    test = np.asarray([row["initial_state"] for row in label_rows], float)
    count = 400 if args.smoke_samples is None else args.smoke_samples
    if count < 1 or count > 400:
        raise ValueError("--smoke-samples must be between 1 and 400")

    training_summary = json.loads((train / "training_summary.json").read_text(encoding="utf-8"))
    is_formal_protocol = bool(training_summary.get("protocol", {}).get("formal_protocol", False)) and args.smoke_samples is None
    context = mp.get_context("spawn")
    output = {
        "formal_protocol": is_formal_protocol,
        "nonformal_smoke_warning": "DO NOT INCLUDE IN FORMAL TABLE" if not is_formal_protocol else None,
        "protocol": "Frozen independent 400-point LHS cold-start NLP reference evaluation; no reference used in training, tuning, seed selection, or model selection.",
        "reference": {
            "source": str(REF),
            "high_fidelity_objective_cache": str(REFERENCE_CACHE),
            "cache_reuse_note": "The immutable cold-reference DOP853 objectives are reused; every network policy is newly audited.",
            "cold_start_solver_success_count": 400,
            "continuous_audit_qualified_count": 286,
            "evaluated_samples": count,
        },
        "hds_statement": "Continuous-time numerical audit evidence under the declared model and numerical settings; not a real-system absolute safety guarantee.",
        "methods": {},
    }

    for method in args.methods:
        training = training_summary["methods"].get(method)
        if training is None:
            output["methods"][method] = {"not_evaluable": True, "reason": "method was not trained"}
            continue
        if not training["completed"]:
            output["methods"][method] = {
                "not_evaluable": True,
                "reason": "training numerical failure; no deployable neural policy",
                "training": training,
            }
            continue
        checkpoint = torch.load(train / f"{method}.pth", map_location="cpu", weights_only=False)
        model = PolicyValue(cstr).eval()
        model.load_state_dict(checkpoint["model"])
        mean = np.asarray(checkpoint["state_mean"], float)
        std = np.asarray(checkpoint["state_std"], float)
        inputs = torch.tensor((test[:count, [0, 2]] - mean) / std, dtype=torch.float32)
        with torch.no_grad():
            for _ in range(10):
                model(inputs)
            started = time.perf_counter()
            _, predicted = model(inputs)
            inference_per_sample = (time.perf_counter() - started) / count
        controls = predicted.numpy()
        payload = [
            (i, test[i].tolist(), controls[i].tolist(), asdict(cstr), GRID)
            for i in range(count)
        ]
        print(f"[CSTR {train.name}] HDS evaluation {method}: {count} samples", flush=True)
        if args.workers == 1:
            rows = [_audit_worker(item) for item in payload]
        else:
            with context.Pool(args.workers) as pool:
                rows = list(pool.imap_unordered(_audit_worker, payload, chunksize=1))
        rows.sort(key=lambda item: item["index"])
        for row in rows:
            ref = reference_map[row["index"]]
            denominator = max(abs(ref["high_fidelity_objective"]), 1e-12)
            row["method"] = method
            row["reference_high_fidelity_objective"] = ref["high_fidelity_objective"]
            row["reference_continuous_audit_qualified"] = ref["continuous_audit_qualified"]
            row["relative_nominal_gap_percent"] = 100.0 * (
                row["nominal_objective"] - ref["high_fidelity_objective"]
            ) / denominator
            row["relative_hds_gap_percent"] = (
                100.0 * (row["hds_objective"] - ref["high_fidelity_objective"]) / denominator
                if row["accepted"] else np.nan
            )
            row["relative_nominal_gap_qualified_reference_percent"] = (
                row["relative_nominal_gap_percent"] if ref["continuous_audit_qualified"] else np.nan
            )
            row["relative_hds_gap_qualified_reference_percent"] = (
                row["relative_hds_gap_percent"]
                if row["accepted"] and ref["continuous_audit_qualified"]
                else np.nan
            )
            row["inference_seconds"] = inference_per_sample
            row["total_predeployment_seconds"] = inference_per_sample + row["hds_seconds"]

        fields = sorted({key for row in rows for key in row})
        with (output_dir / f"test_per_sample_{method}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        output["methods"][method] = {
            "training": training,
            "accepted_network_samples": int(sum(row["accepted"] for row in rows)),
            "fallback_samples": int(sum(row["fallback"] for row in rows)),
            "optimizer_fallback_is_neural_policy": False,
            "nominal_violation_rate_percent": float(100 * np.mean([row["nominal_max_g"] > 1e-8 for row in rows])),
            "nominal_max_g": float(np.max([row["nominal_max_g"] for row in rows])),
            "final_max_g_mean": _mean([row["final_max_g"] for row in rows]),
            "mean_corrected_segments": _mean([row["corrected_segments"] for row in rows]),
            "mean_hds_objective_change": _mean([row["hds_objective"] - row["nominal_objective"] for row in rows]),
            "nominal_gap_all_400_percent": stats([row["relative_nominal_gap_percent"] for row in rows]),
            "hds_gap_all_400_percent": stats([row["relative_hds_gap_percent"] for row in rows]),
            "nominal_gap_continuous_audit_qualified_reference_percent": stats([row["relative_nominal_gap_qualified_reference_percent"] for row in rows]),
            "hds_gap_continuous_audit_qualified_reference_percent": stats([row["relative_hds_gap_qualified_reference_percent"] for row in rows]),
            "mean_inference_seconds": inference_per_sample,
            "mean_hds_seconds": _mean([row["hds_seconds"] for row in rows]),
            "mean_total_predeployment_seconds": _mean([row["total_predeployment_seconds"] for row in rows]),
        }

    (output_dir / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    table = [
        "| Method | HDS gap, all 400 (%) | HDS gap, qualified reference subset (%) | Corrected segments | Accepted / fallback |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, result in output["methods"].items():
        if result.get("not_evaluable"):
            table.append(f"| {method} | n/a | n/a | n/a | training failure |")
            continue
        all_gap = result["hds_gap_all_400_percent"]["mean"]
        qualified_gap = result["hds_gap_continuous_audit_qualified_reference_percent"]["mean"]
        all_text = "n/a" if all_gap is None else f"{all_gap:.4f}"
        qualified_text = "n/a" if qualified_gap is None else f"{qualified_gap:.4f}"
        table.append(
            f"| {method} | {all_text} | {qualified_text} | "
            f"{result['mean_corrected_segments']:.2f} | "
            f"{result['accepted_network_samples']} / {result['fallback_samples']} |"
        )
    (output_dir / "summary_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
