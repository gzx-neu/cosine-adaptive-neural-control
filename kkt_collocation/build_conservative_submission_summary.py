"""Build the frozen submission summary for the conservative HDS protocol.

This script intentionally excludes historical 5/20-point comparisons and the
former ``g_max <= 1e-8`` acceptance rule.  It combines only the final
three-seed in-domain reevaluation, matched 400-point comparisons, guard-bypassed
OOD diagnostics, and cross-integrator margin-calibration reports used by the
active manuscript.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "kkt_collocation" / "results"
CONSERVATIVE = RESULTS / "conservative_margin_1e6"
CSTR_GRID31 = RESULTS / "conservative_margin_1e6_grid31"
WRITING = ROOT / "论文写作"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    in_domain = read_json(CONSERVATIVE / "summary.json")
    in_domain["benchmarks"]["CSTR"] = read_json(CSTR_GRID31 / "summary.json")["benchmarks"]["CSTR"]
    matched = {
        problem: read_json(CONSERVATIVE / "matched400" / f"{problem.lower()}_summary.json")
        for problem in ("VDP", "Penicillin", "CSTR")
    }
    matched["CSTR"] = read_json(CSTR_GRID31 / "matched400" / "cstr_summary.json")
    ood = {
        "VDP": read_json(CONSERVATIVE / "ood_vdp_seed20260751" / "summary.json"),
        "Penicillin": read_json(CONSERVATIVE / "ood_penicillin_seed20260761" / "summary.json"),
        "CSTR": read_json(CONSERVATIVE / "ood_cstr_seed20260722" / "summary.json"),
    }
    ood["CSTR"] = read_json(RESULTS / "domain_stress_cstr_900_seed20260722_grid31" / "summary.json")

    base_calibration = read_json(RESULTS / "hds_margin_calibration" / "summary.json")
    cstr_calibration = read_json(RESULTS / "hds_margin_calibration_cstr_critical" / "summary.json")
    calibration = {
        "proposed_safety_margin": 1e-6,
        "interpretation": "empirical cross-audit; not a formal global-error certificate",
        "benchmarks": {
            "VDP": base_calibration["benchmarks"]["VDP"],
            "Penicillin": base_calibration["benchmarks"]["Penicillin"],
            "CSTR": cstr_calibration["benchmarks"]["CSTR"],
        },
    }
    calibration["maximum_absolute_cross_audit_difference"] = max(
        item["max_absolute_cross_audit_difference"]
        for item in calibration["benchmarks"].values()
    )

    report = {
        "schema_version": "VALC-conservative-HDS-v1",
        "status": "frozen submission data",
        "safety_protocol": {
            "decision_rule": "HDS peak <= -eta_safe",
            "eta_safe": 1e-6,
            "independent_acceptance_rule": "tight-audit HDS peak <= 0",
            "scope": "instance-level numerical feasibility evidence under the stated model and audit settings",
        },
        "in_domain_three_seed_1200": in_domain,
        "matched_cold_start_400": matched,
        "guard_bypassed_ood_diagnostic": ood,
        "margin_calibration": calibration,
        "manuscript_tables": [
            "table_in_domain.tex",
            "table_strict_comparison.tex",
            "ood_stress_table.tex",
        ],
        "manuscript_figures": [
            "population_control_envelopes",
            "population_path_constraint_envelopes",
            "candidate_kkt_paired_migration",
            "hds_safety_performance_migration",
            "ood_stress_diagnostic",
        ],
    }

    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (WRITING / "final_experiment_data.json").write_text(payload, encoding="utf-8")
    (CONSERVATIVE / "final_submission_summary.json").write_text(payload, encoding="utf-8")
    print(WRITING / "final_experiment_data.json")
    print(CONSERVATIVE / "final_submission_summary.json")


if __name__ == "__main__":
    main()
