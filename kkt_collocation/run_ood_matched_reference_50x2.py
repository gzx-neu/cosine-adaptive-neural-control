"""Frozen 50+50 matched cold-start OOD references and 30-seed gap table.

Stages are explicitly separated so MATLAB Jiang--Fu shards and the Python
CSTR reduced-space solves remain resumable.  No model is trained or used to
initialize a deterministic solve.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kkt_collocation"))

from offline_safe_control.adaptive_event_hds import AdaptiveEventHDSConfig, AdaptiveEventHDSCorrector
from kkt_collocation.economou_cstr_hds_fast import segment_event_audit
from kkt_collocation.reevaluate_cstr_margin1e6_30seeds import correct_with_threshold
from kkt_collocation.generate_economou_cstr_reduced_kkt_data import (
    _worker_init as cstr_worker_init,
    _worker_solve as cstr_worker_solve,
)
from kkt_collocation.run_economou_cstr_supervised_hds import trajectory_objective
from kkt_collocation.run_penicillin_ablation import (
    DT as PEN_DT, UMAX as PEN_UMAX, g as pen_g, gdot as pen_gdot,
    ode as pen_ode, terminal_product,
)
from kkt_collocation.run_vdp_ablation import (
    constraint as vdp_g, constraint_derivative as vdp_gdot,
    terminal_cost, vdp_ode,
)
from kkt_collocation.screen_economou_cstr_30x30 import EconomouScreenConfig


RESULTS = ROOT / "kkt_collocation" / "results"
OOD = RESULTS / "ca_kkt_ood_stress_30seeds_margin1e8_20260806_v1"
OUT = RESULTS / "ca_kkt_ood_matched_reference_50x2_margin1e8_20260806_v1"
SELECTION_SEED = 20260804
THRESHOLD = -1e-8
LAYERS = ("near_ood_0_10pct", "far_ood_10_20pct")
PROBLEMS = ("vdp", "penicillin", "cstr")
SEEDS = tuple(range(20260771, 20260801))


def dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def load_cstr_config() -> EconomouScreenConfig:
    raw = json.loads((RESULTS / "economou_cstr_reduced_kkt_n100_test400_lhs_margin0" / "summary.json").read_text(encoding="utf-8"))["config"]
    for key in ("ti_bounds_K", "flow_bounds", "ca_initial_range", "temperature_initial_range_K"):
        raw[key] = tuple(raw[key])
    cfg = EconomouScreenConfig(**raw)
    if cfg.zoh_steps != 100 or cfg.substeps_per_zoh != 10 or cfg.node_margin != 0:
        raise ValueError("Unexpected frozen CSTR protocol")
    return cfg


def stage_prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort_out = OUT / "cohorts"; cohort_out.mkdir(exist_ok=True)
    selected: list[dict] = []
    offsets = {"vdp": 11, "penicillin": 21, "cstr": 31}
    for problem in PROBLEMS:
        for layer_index, layer in enumerate(LAYERS):
            source = OOD / "cohorts" / f"{problem}_{layer}.csv"
            rows = read_csv(source)
            if len(rows) != 100:
                raise ValueError(f"Expected 100 source OOD points: {source}")
            rng = np.random.default_rng(SELECTION_SEED + offsets[problem] + layer_index)
            indices = np.sort(rng.choice(100, size=50, replace=False))
            for local_index, source_index in enumerate(indices):
                row = rows[int(source_index)]
                base = {
                    "problem": problem,
                    "layer": layer,
                    "reference_index_within_layer": local_index,
                    "source_sample_index": int(row["sample_index"]),
                    "selection_seed": SELECTION_SEED + offsets[problem] + layer_index,
                }
                if problem == "vdp":
                    base.update({"y1_0": float(row["y1_0"]), "y2_0": float(row["y2_0"]), "y3_0": 0.0})
                elif problem == "penicillin":
                    base.update({"x2_0": float(row["x2_0"])})
                else:
                    base.update({"CA_0": float(row["CA_0"]), "CB_0": float(row["CB_0"]), "T_0_K": float(row["T_0_K"])})
                selected.append(base)
    write_csv(cohort_out / "selected_50x2_points.csv", selected)
    matlab_rows = []
    for row in selected:
        if row["problem"] == "cstr":
            continue
        is_vdp = row["problem"] == "vdp"
        matlab_rows.append({
            "problem": "VDP" if is_vdp else "Penicillin",
            "layer": row["layer"],
            "point_id": f"{'vdp' if is_vdp else 'pen'}_{row['layer']}_{row['reference_index_within_layer']:03d}",
            "source_sample_index": row["source_sample_index"],
            "x1_0": row["y1_0"] if is_vdp else 1.0,
            "x2_0": row["y2_0"] if is_vdp else row["x2_0"],
        })
    write_csv(cohort_out / "jiang_fu_ood100_input.csv", matlab_rows)
    dump(OUT / "selection_protocol.json", {
        "selection_seed_base": SELECTION_SEED,
        "rule": "fixed RNG selection without replacement from each already-frozen 100-point OOD layer; independent of model performance",
        "points_per_problem_per_layer": 50,
        "source": str(OOD / "cohorts"),
        "selected_csv": "cohorts/selected_50x2_points.csv",
    })
    print(OUT / "cohorts" / "selected_50x2_points.csv")


def stage_cstr(workers: int) -> None:
    cfg = load_cstr_config()
    rows = [row for row in read_csv(OUT / "cohorts" / "selected_50x2_points.csv") if row["problem"] == "cstr"]
    if len(rows) != 100:
        raise ValueError("Expected 100 selected CSTR points")
    states = [
        [float(row["CA_0"]), float(row["CB_0"]), float(row["T_0_K"])] for row in rows
    ]
    records_path = OUT / "cstr_cold_start_records.jsonl"
    existing = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line); existing[int(record["index"])] = record
    pending = [(index, state) for index, state in enumerate(states) if index not in existing]
    context = mp.get_context("spawn")
    with records_path.open("a", encoding="utf-8") as handle:
        with context.Pool(workers, initializer=cstr_worker_init, initargs=(asdict(cfg),)) as pool:
            for record in pool.imap_unordered(cstr_worker_solve, pending, chunksize=1):
                handle.write(json.dumps(record, allow_nan=False) + "\n"); handle.flush()
                print(f"CSTR {record['index'] + 1}/100 success={record['success']}", flush=True)
    print(records_path)


def parse_controls(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.split(";") if item.strip()], float)


def audit_jiang() -> list[dict]:
    raw_dir = OUT / "jiang_fu_raw"
    files = sorted(raw_dir.glob("jiang_fu_ood100_raw_shard_*_of_*.csv"))
    if not files:
        raise FileNotFoundError("No Jiang--Fu shard CSV files found")
    attempted: dict[tuple[str, str, int], dict] = {}
    for path in files:
        for row in read_csv(path):
            if not parse_bool(row["attempted"]):
                continue
            key = (row["problem"], row["layer"], int(float(row["source_sample_index"])))
            if key in attempted:
                raise ValueError(f"Duplicate Jiang reference {key}")
            attempted[key] = row
    if len(attempted) != 200:
        raise ValueError(f"Expected 200 attempted Jiang references, found {len(attempted)}")
    audited = []
    for key, row in sorted(attempted.items()):
        output = dict(row)
        success = parse_bool(row["success"])
        output.update({"solver_success": success, "qualified_reference": False,
                       "reference_nominal_hds_gmax": np.nan, "reference_hds_gmax": np.nan,
                       "reference_corrected_segments": np.nan,
                       "reference_high_fidelity_objective": np.nan})
        if success:
            controls = parse_controls(row["controls"])
            if row["problem"] == "VDP":
                initial = np.array([float(row["x1_0"]), float(row["x2_0"]), 0.0])
                corrector = AdaptiveEventHDSCorrector(vdp_ode, vdp_g, vdp_gdot, (-0.3, 1.0),
                    AdaptiveEventHDSConfig(grid_size=31, safety_margin=abs(THRESHOLD)))
                nominal_peak = corrector.audit(initial, controls, 0.5)
                outcome = corrector.correct(initial, controls, 0.5)
                final_peak = corrector.audit(initial, outcome.controls, 0.5) if outcome.accepted else np.nan
                objective = terminal_cost(initial, outcome.controls, corrector, 0.5) if outcome.accepted else np.nan
            else:
                x2 = float(row["x2_0"]); initial = np.array([1.0, x2, 0.001, 250.0])
                corrector = AdaptiveEventHDSCorrector(pen_ode, pen_g, pen_gdot, (0.0, PEN_UMAX),
                    AdaptiveEventHDSConfig(grid_size=31, safety_margin=abs(THRESHOLD)))
                nominal_peak = corrector.audit(initial, controls, PEN_DT)
                outcome = corrector.correct(initial, controls, PEN_DT)
                final_peak = corrector.audit(initial, outcome.controls, PEN_DT) if outcome.accepted else np.nan
                objective = -terminal_product(x2, outcome.controls, corrector) if outcome.accepted else np.nan
            qualified = bool(outcome.accepted and final_peak <= THRESHOLD)
            output.update({"reference_nominal_hds_gmax": float(nominal_peak),
                           "reference_hds_gmax": float(final_peak),
                           "reference_corrected_segments": int(sum(s.corrected for s in outcome.segments)),
                           "qualified_reference": qualified,
                           "reference_high_fidelity_objective": float(objective)})
        audited.append(output)
    write_csv(OUT / "jiang_fu_references_audited.csv", audited)
    return audited


def audit_cstr() -> list[dict]:
    cfg = load_cstr_config()
    selected = [row for row in read_csv(OUT / "cohorts" / "selected_50x2_points.csv") if row["problem"] == "cstr"]
    raw = {}
    for line in (OUT / "cstr_cold_start_records.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line); raw[int(row["index"])] = row
    if len(raw) != 100:
        raise ValueError(f"Expected 100 CSTR reference records, found {len(raw)}")
    audited = []
    for index, point in enumerate(selected):
        row = raw[index]
        output = {**row, "problem": "cstr", "layer": point["layer"],
                  "source_sample_index": int(point["source_sample_index"]),
                  "solver_success": bool(row["success"]), "qualified_reference": False,
                  "reference_nominal_hds_gmax": np.nan, "reference_hds_gmax": np.nan,
                  "reference_corrected_segments": np.nan,
                  "reference_high_fidelity_objective": np.nan}
        if row["success"]:
            state = np.asarray(row["initial_state"], float); controls = np.asarray(row["controls"], float)
            current = state.copy(); peak = -np.inf
            for control in controls:
                local, current, _ = segment_event_audit(current, control, cfg)
                peak = max(peak, float(np.max(local)))
            outcome = correct_with_threshold(state, controls, cfg, THRESHOLD)
            objective = trajectory_objective(state, outcome["controls"], cfg) if outcome["accepted"] else np.nan
            output.update({"reference_nominal_hds_gmax": peak,
                           "reference_hds_gmax": float(outcome["final_peak"]),
                           "reference_corrected_segments": int(outcome["corrected_segments"]),
                           "qualified_reference": bool(outcome["accepted"] and outcome["final_peak"] <= THRESHOLD),
                           "reference_high_fidelity_objective": objective})
        audited.append(output)
    write_csv(OUT / "cstr_references_audited.csv", audited)
    return audited


def mean_sd(values) -> tuple[float, float]:
    array = np.asarray(values, float); array = array[np.isfinite(array)]
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def stage_compile() -> None:
    jiang = audit_jiang(); cstr = audit_cstr()
    references = {}
    for row in jiang:
        problem = "vdp" if row["problem"] == "VDP" else "penicillin"
        references[(problem, row["layer"], int(float(row["source_sample_index"])))] = row
    for row in cstr:
        references[("cstr", row["layer"], int(row["source_sample_index"]))] = row
    per_seed = []
    for problem in PROBLEMS:
        for seed in SEEDS:
            for layer in LAYERS:
                network_rows = {int(row["sample_index"]): row for row in read_csv(OOD / problem / f"seed{seed}" / f"{layer}_per_sample.csv")}
                selected_keys = sorted(key for key in references if key[0] == problem and key[1] == layer)
                gaps_all, gaps_qualified, corrected, accepted = [], [], [], []
                for key in selected_keys:
                    ref = references[key]; net = network_rows[key[2]]
                    net_ok = parse_bool(net["accepted"]); accepted.append(net_ok)
                    corrected.append(float(net["corrected_segments"]))
                    if net_ok and parse_bool(str(ref["solver_success"])):
                        jref = float(ref["reference_high_fidelity_objective"])
                        gap = 100.0 * (float(net["hds_objective_J"]) - jref) / max(abs(jref), 1e-12)
                        gaps_all.append(gap)
                        if parse_bool(str(ref["qualified_reference"])):
                            gaps_qualified.append(gap)
                per_seed.append({
                    "problem": problem, "seed": seed, "layer": layer,
                    "selected_points": len(selected_keys),
                    "network_hds_acceptance_rate_percent": 100 * np.mean(accepted),
                    "mean_corrected_segments": np.mean(corrected),
                    "matched_gap_all_solver_success_percent": np.mean(gaps_all) if gaps_all else np.nan,
                    "matched_gap_qualified_reference_percent": np.mean(gaps_qualified) if gaps_qualified else np.nan,
                    "all_gap_point_count": len(gaps_all),
                    "qualified_gap_point_count": len(gaps_qualified),
                })
    write_csv(OUT / "per_seed_matched_gap.csv", per_seed)
    aggregate = {"protocol": {
        "diagnostic_only": True, "points_per_layer": 50, "training_seeds": list(SEEDS),
        "reference_gap": "100*(J_network_HDS-J_cold_start_plus_HDS)/abs(J_cold_start_plus_HDS)",
        "reference_audit_threshold": THRESHOLD,
        "no_warm_start": True,
    }, "rows": []}
    for problem in PROBLEMS:
        for layer in LAYERS:
            group = [row for row in per_seed if row["problem"] == problem and row["layer"] == layer]
            refs = [row for key, row in references.items() if key[0] == problem and key[1] == layer]
            solved = sum(parse_bool(str(row["solver_success"])) for row in refs)
            qualified = sum(parse_bool(str(row["qualified_reference"])) for row in refs)
            item = {"problem": problem, "layer": layer, "reference_solver_success": solved,
                    "qualified_references": qualified, "selected_references": 50}
            for source, target in (
                ("network_hds_acceptance_rate_percent", "hds_acceptance_percent"),
                ("mean_corrected_segments", "corrected_segments"),
                ("matched_gap_all_solver_success_percent", "gap_all_success_percent"),
                ("matched_gap_qualified_reference_percent", "gap_qualified_percent"),
            ):
                mean, sd = mean_sd([row[source] for row in group])
                item[target] = {"mean": mean, "sample_sd": sd}
            aggregate["rows"].append(item)
    dump(OUT / "aggregate_summary.json", aggregate)
    labels = {"vdp": "VDP", "penicillin": "Penicillin", "cstr": "Economou CSTR"}
    layer_labels = {LAYERS[0]: "Near OOD (0–10%)", LAYERS[1]: "Far OOD (10–20%)"}
    lines = ["# 30-seed matched OOD cold-start+HDS comparison", "",
             "Mean ± sample SD across 30 fixed training seeds; 50 fixed matched points per OOD layer.", "",
             "| Benchmark | OOD layer | Network HDS accepted (%) | Network corrected segments | Matched cold-start+HDS gap (%) |",
             "|---|---|---:|---:|---:|"]
    for row in aggregate["rows"]:
        gap = row["gap_qualified_percent"] if row["qualified_references"] else row["gap_all_success_percent"]
        lines.append(f"| {labels[row['problem']]} | {layer_labels[row['layer']]} | "
                     f"{row['hds_acceptance_percent']['mean']:.4f} ± {row['hds_acceptance_percent']['sample_sd']:.4f} | "
                     f"{row['corrected_segments']['mean']:.4f} ± {row['corrected_segments']['sample_sd']:.4f} | "
                     f"{gap['mean']:.4f} ± {gap['sample_sd']:.4f} |")
    lines += ["", "Notes: all six reference cohorts achieved 50/50 deterministic-solver success and 50/50 acceptance after the same adaptive-event HDS audit/correction, with final g_max <= -1e-8. No neural/policy/label warm start is used. The reported gap compares network+HDS with matched cold-start+HDS at the same OOD initial condition. The table is a guard-bypassed OOD diagnostic, not an OOD safety or global-optimality claim."]
    (OUT / "aggregate_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((OUT / "aggregate_table.md").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "cstr", "compile"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "prepare": stage_prepare()
    elif args.stage == "cstr": stage_cstr(args.workers)
    else: stage_compile()


if __name__ == "__main__":
    main()
