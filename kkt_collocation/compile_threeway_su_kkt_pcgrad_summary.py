"""Compile the fixed-seed, all-400 S-u/KKT/PCGrad exploratory audit table."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kkt_collocation/results/exploratory_su_kkt_pcgrad_matched400_v1"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if not (OUT / "vdp_seed20260771/summary.json").is_file() or not (OUT / "penicillin_seed20260771/summary.json").is_file():
        raise FileNotFoundError("Run evaluate_su_kkt_pcgrad_matched400.py for VDP and penicillin first")
    vdp = load(OUT / "vdp_seed20260771/summary.json")["methods"]
    pen = load(OUT / "penicillin_seed20260771/summary.json")["methods"]
    cstr_direct = load(ROOT / "kkt_collocation/results/exploratory_cstr_su_k10_hds_seed20260771_v2/summary.json")["methods"]
    cstr_pcgrad = load(ROOT / "kkt_collocation/results/exploratory_cstr_su_k10_pcgrad_validation_v1/hds_test400_pcgrad_selected/summary.json")["methods"]
    cstr_ref = load(ROOT / "kkt_collocation/results/economou_cstr_reduced_kkt_n100_test400_lhs_margin0/summary.json")
    cold_cstr = float(cstr_ref["mean_solve_seconds_successful"])

    rows: list[dict[str, object]] = []
    for benchmark, source in (("VDP", vdp), ("Penicillin", pen)):
        for method in ("S-u", "S-u+K", "S-u+K-PCGrad"):
            metric = source[method]["deployment"]
            rows.append({
                "benchmark": benchmark, "method": method, "seed": 20260771,
                "reference_points": 400, "hds_gap_percent": 100 * float(metric["hds_relative_objective_gap"]),
                "mean_corrected_segments": float(metric["mean_corrected_segments"]),
                "mean_inference_seconds": float(metric["mean_inference_seconds"]),
                "mean_hds_seconds": float(metric["mean_audit_seconds"]),
                "mean_total_deployment_seconds": float(metric["mean_total_seconds"]),
                "mean_cold_reference_seconds": float(metric["mean_cold_reference_solve_seconds"]),
                "mean_cold_over_deployment_speedup": float(metric["mean_cold_reference_speedup"]),
                "acceptance_rate": float(metric["continuous_time_audit_acceptance_rate"]),
                "fallback_rate": float(metric["offline_optimizer_fallback_rate"]),
                "cstr_qualified_286_hds_gap_percent": "",
            })

    for method, source_method in (("S-u", cstr_pcgrad["S-u"]), ("S-u+K", cstr_direct["S-u+K"]),
                                  ("S-u+K-PCGrad", cstr_pcgrad["S-u+K"])):
        all400 = source_method["hds_gap_all_400_percent"]
        q286 = source_method["hds_gap_continuous_audit_qualified_reference_percent"]
        total = float(source_method["mean_total_predeployment_seconds"])
        rows.append({
            "benchmark": "Economou CSTR", "method": method, "seed": 20260771,
            "reference_points": 400, "hds_gap_percent": float(all400["mean"]),
            "mean_corrected_segments": float(source_method["mean_corrected_segments"]),
            "mean_inference_seconds": float(source_method["mean_inference_seconds"]),
            "mean_hds_seconds": float(source_method["mean_hds_seconds"]),
            "mean_total_deployment_seconds": total,
            "mean_cold_reference_seconds": cold_cstr,
            "mean_cold_over_deployment_speedup": cold_cstr / total,
            "acceptance_rate": float(source_method["accepted_network_samples"]) / 400.0,
            "fallback_rate": float(source_method["fallback_samples"]) / 400.0,
            "cstr_qualified_286_hds_gap_percent": float(q286["mean"]),
        })

    with (OUT / "summary_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "formal_protocol": False,
        "scope": "single fixed seed, pre-existing policies, all-400 pointwise cold references; evaluation only",
        "main_metric": "HDS-corrected relative objective gap (%) against the same initial-condition cold reference",
        "cstr_note": "all 400 are the main CSTR result; the 286-point continuous-audit-qualified reference subset is reported only as a supplement",
        "hds_statement": "Continuous-time numerical audit evidence under the declared model and numerical settings; not a real-system absolute safety guarantee.",
        "rows": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
