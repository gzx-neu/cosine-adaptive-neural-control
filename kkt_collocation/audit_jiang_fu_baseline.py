"""Independently audit the Jiang--Fu Algorithm-1 baseline outputs.

The MATLAB wrapper preserves the authors' solver and exports its controls.
This script re-integrates those controls with the same event-located HDS
auditor used for every learned-policy result in this project.  It therefore
keeps solver termination and continuous-time verification as separate facts.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from kkt_collocation.run_penicillin_ablation import g as pen_g
from kkt_collocation.run_penicillin_ablation import gdot as pen_gdot
from kkt_collocation.run_penicillin_ablation import ode as pen_ode
from kkt_collocation.run_vdp_ablation import constraint as vdp_g
from kkt_collocation.run_vdp_ablation import constraint_derivative as vdp_gdot
from kkt_collocation.run_vdp_ablation import vdp_ode


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def parse_controls(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.split(";") if item.strip()], dtype=float)


def audit_row(row: dict[str, str]) -> dict[str, object]:
    output: dict[str, object] = dict(row)
    success = parse_bool(row["success"])
    output["solver_success"] = success
    output["hds_audited"] = False
    output["hds_safe"] = False
    output["hds_gmax"] = float("nan")
    output["independent_objective"] = float("nan")
    output["objective_abs_difference"] = float("nan")
    output["hds_audit_seconds"] = float("nan")
    if not success or not row["controls"].strip():
        return output

    controls = parse_controls(row["controls"])
    n_zoh = int(float(row["n_zoh"]))
    if len(controls) != n_zoh:
        raise ValueError(f"{row['problem']} {row['point_id']}: expected {n_zoh} controls, got {len(controls)}")

    problem = row["problem"]
    if problem == "VDP":
        initial = np.array([float(row["x1_0"]), float(row["x2_0"]), 0.0])
        duration = 5.0 / n_zoh
        corrector = HDSLambdaCorrector(
            vdp_ode, vdp_g, vdp_gdot,
            (float(row["u_min"]), float(row["u_max"])),
            HDSLambdaConfig(grid_size=101, max_step_fraction=200.0),
        )
    elif problem == "Penicillin":
        initial = np.array([1.0, float(row["x2_0"]), 0.001, 250.0])
        duration = 40.0 / n_zoh
        corrector = HDSLambdaCorrector(
            pen_ode, pen_g, pen_gdot,
            (float(row["u_min"]), float(row["u_max"])),
            HDSLambdaConfig(grid_size=101, max_step_fraction=200.0),
        )
    else:
        raise ValueError(f"unknown problem {problem!r}")

    started = time.perf_counter()
    state = initial.copy()
    peak = -np.inf
    for control in controls:
        local_peak, state = corrector.segment_peak(state, float(control), duration)
        peak = max(peak, local_peak)
    audit_seconds = time.perf_counter() - started
    objective = float(state[2]) if problem == "VDP" else -float(state[2])
    matlab_objective = float(row["objective"])

    output.update({
        "hds_audited": True,
        "hds_safe": bool(peak <= 1e-8),
        "hds_gmax": float(peak),
        "independent_objective": objective,
        "objective_abs_difference": abs(objective - matlab_objective),
        "hds_audit_seconds": audit_seconds,
    })
    return output


def finite_mean(rows: list[dict[str, object]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else float("nan")


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scenario"]), str(row["problem"]))].append(row)
    output: dict[str, object] = {}
    for (scenario, problem), group in groups.items():
        solved = [row for row in group if bool(row["solver_success"])]
        audited = [row for row in solved if bool(row["hds_audited"])]
        output[f"{scenario}/{problem}"] = {
            "points": len(group),
            "solver_success": len(solved),
            "hds_safe": sum(bool(row["hds_safe"]) for row in audited),
            "mean_solve_seconds": finite_mean(solved, "solve_seconds"),
            "mean_hds_audit_seconds": finite_mean(audited, "hds_audit_seconds"),
            "max_hds_g": max((float(row["hds_gmax"]) for row in audited), default=float("nan")),
            "mean_objective": finite_mean(audited, "independent_objective"),
            "max_objective_abs_difference": max(
                (float(row["objective_abs_difference"]) for row in audited), default=float("nan")
            ),
            "mean_outer_iterations": finite_mean(solved, "outer_iterations"),
            "mean_final_constraint_nodes": finite_mean(solved, "final_constraint_nodes"),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "author_pcdo_code" / "01_Single_Constraint_Cases" / "Proposed_Method"
        / "JiangFu_Codex_Baseline" / "jiang_fu_baseline_raw.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "kkt_collocation" / "results" / "jiang_fu_algorithm1_baseline",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = [audit_row(row) for row in raw_rows]
    if not rows:
        raise RuntimeError("Jiang--Fu MATLAB result file is empty")

    fieldnames = list(rows[0].keys())
    with (args.output / "per_point.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "method": "Jiang--Fu Algorithm 1 (integral-transcription upper bound and inner approximation)",
        "source": "Automatica 191 (2026) 113123",
        "timing": "single cold MATLAB solve measured inside the authors' solver; HDS audit reported separately",
        "safety_rule": "event-located HDS g_max <= 1e-8",
        "groups": summarize(rows),
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
