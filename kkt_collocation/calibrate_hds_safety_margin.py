"""Empirically calibrate the conservative HDS acceptance margin.

The script reconstructs frozen deployment policies, selects the trajectories
whose previously accepted HDS peaks were closest to the constraint boundary,
re-applies the correction with a proposed negative margin, and cross-audits the
result with tighter DOP853 settings and the independent Radau integrator.

This is a numerical convergence study, not a formal global-error certificate.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from kkt_collocation.run_cstr_full_simulation import CSTRConfig, Policy as CSTRPolicy, cstr_ode, lhs_states, path_constraint, path_derivative
from kkt_collocation.run_penicillin_ablation import DT as PEN_DT, UMAX as PEN_UMAX, Policy as PenPolicy, g as pen_g, gdot as pen_gdot, ode as pen_ode
from kkt_collocation.run_vdp_ablation import constraint as vdp_g, constraint_derivative as vdp_gdot, vdp_ode
from kkt_collocation.train_vdp_kkt_policy import KKTPolicyValueNetwork, TrainConfig


def checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def closest_indices(path: Path, method_suffix: str | None, peak_key: str, count: int,
                    training_seed: int | None = None) -> np.ndarray:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if method_suffix is not None:
        rows = [row for row in rows if row["method"].startswith("Adaptive (") and row["method"].endswith(method_suffix)]
    if training_seed is not None:
        rows = [row for row in rows if int(row["training_seed"]) == training_seed]
    rows.sort(key=lambda row: abs(float(row[peak_key])))
    return np.asarray([int(row["sample_index"]) for row in rows[:count]], dtype=int)


def policy_controls(problem: str, count: int) -> tuple[np.ndarray, np.ndarray, float, tuple, tuple]:
    if problem == "VDP":
        directory = ROOT / "kkt_collocation" / "results" / "final_multiseed_vdp900_penalty_seed20260751"
        states = np.load(directory / "test_states.npy")
        stored = checkpoint(directory / "models.pth")
        branch = json.loads((directory / "summary.json").read_text(encoding="utf-8"))["adaptive_gate"]["selected_branch"]
        key = "KKT" if branch == "S+KKT" else "S"
        model = KKTPolicyValueNetwork(TrainConfig()); model.load_state_dict(stored[key]); model.eval()
        mean, std = stored["normalization"][key]
        with torch.no_grad():
            controls = model(torch.tensor((states[:, :2] - np.asarray(mean)) / np.asarray(std), dtype=torch.float32))[1].numpy()
        indices = closest_indices(directory / "per_sample.csv", " + HDS-lambda", "applied_hds_max_g", count)
        return states[indices], controls[indices], 0.5, (vdp_ode, vdp_g, vdp_gdot), (-0.3, 1.0)

    if problem == "Penicillin":
        directory = ROOT / "kkt_collocation" / "results" / "final_multiseed_penicillin400_penalty_seed20260761"
        x2 = np.load(directory / "test_x2.npy")
        states = np.column_stack((np.ones(len(x2)), x2, np.full(len(x2), 0.001), np.full(len(x2), 250.0)))
        stored = checkpoint(directory / "models.pth")
        branch = json.loads((directory / "summary.json").read_text(encoding="utf-8"))["adaptive_gate"]["selected_branch"]
        key = "true_KKT" if "true-KKT" in branch else "S"
        model = PenPolicy(); model.load_state_dict(stored[key]); model.eval()
        mean, std = stored["normalization"][key]
        with torch.no_grad():
            controls = model(torch.tensor(((x2 - float(mean)) / float(std))[:, None], dtype=torch.float32))[1].numpy()
        indices = closest_indices(directory / "per_sample.csv", " + HDS-lambda", "applied_hds_max_g", count)
        return states[indices], controls[indices], PEN_DT, (pen_ode, pen_g, pen_gdot), (0.0, PEN_UMAX)

    directory = ROOT / "kkt_collocation" / "results" / "cstr_multiseed_test1200_900_clean"
    training_seed, test_seed = 20260726, 20260826
    stored = checkpoint(directory / f"cstr_seed{training_seed}.pth")
    cfg_data = dict(stored["config"])
    cfg_data["hds_tolerance"] = 1e-6
    cfg = CSTRConfig(**cfg_data)
    states = lhs_states(cfg, 400, test_seed)
    model = CSTRPolicy(cfg); model.load_state_dict(stored["model"]); model.eval()
    with torch.no_grad():
        controls = model(torch.tensor((states - np.asarray(stored["mean"])) / np.asarray(stored["std"]), dtype=torch.float32))[1].numpy()
    indices = closest_indices(directory / "per_sample.csv", None, "applied_hds_max_g", count, training_seed)
    dynamics = lambda t, x, u: cstr_ode(t, x, u, cfg)
    constraint = lambda x: path_constraint(x, cfg)
    derivative = lambda x, u: path_derivative(x, u, cfg)
    return states[indices], controls[indices], cfg.zoh_duration, (dynamics, constraint, derivative), (cfg.cooling_min, cfg.cooling_max)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin", type=float, default=1e-6)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--problems", nargs="+", choices=("VDP", "Penicillin", "CSTR"),
                        default=("VDP", "Penicillin", "CSTR"))
    parser.add_argument("--output", type=Path, default=ROOT / "kkt_collocation" / "results" / "hds_margin_calibration")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    audit_settings = {
        "dop853_standard": HDSLambdaConfig(safety_margin=args.margin, rtol=1e-10, atol=1e-12, max_step_fraction=100.0, integrator="DOP853"),
        "dop853_tight": HDSLambdaConfig(safety_margin=args.margin, rtol=1e-12, atol=1e-14, max_step_fraction=400.0, integrator="DOP853"),
        "radau_tight": HDSLambdaConfig(safety_margin=args.margin, rtol=1e-11, atol=1e-13, max_step_fraction=400.0, integrator="Radau"),
    }
    report: dict[str, dict] = {"proposed_safety_margin": args.margin, "interpretation": "empirical cross-audit; not a formal error certificate", "benchmarks": {}}
    for problem in args.problems:
        states, nominal, duration, funcs, bounds = policy_controls(problem, args.count)
        corrector = HDSLambdaCorrector(*funcs, bounds, HDSLambdaConfig(grid_size=31 if problem != "CSTR" else 21, safety_margin=args.margin, rtol=1e-10, atol=1e-12, max_step_fraction=100.0))
        auditors = {name: HDSLambdaCorrector(*funcs, bounds, config) for name, config in audit_settings.items()}
        rows = []
        for index, (state, control) in enumerate(zip(states, nominal)):
            outcome = corrector.correct(state, control, duration)
            if not outcome.accepted:
                rows.append({"rank": index, "accepted": False})
                continue
            peaks = {name: auditor.audit(state, outcome.controls, duration) for name, auditor in auditors.items()}
            rows.append({"rank": index, "accepted": True, **peaks})
        accepted = [row for row in rows if row["accepted"]]
        standard = np.asarray([row["dop853_standard"] for row in accepted])
        tight = np.asarray([row["dop853_tight"] for row in accepted])
        radau = np.asarray([row["radau_tight"] for row in accepted])
        report["benchmarks"][problem] = {
            "selected_boundary_critical_trajectories": len(rows),
            "accepted_after_margin_correction": len(accepted),
            "max_peak_by_audit": {"dop853_standard": float(standard.max()), "dop853_tight": float(tight.max()), "radau_tight": float(radau.max())},
            "max_absolute_cross_audit_difference": float(max(np.max(np.abs(tight-standard)), np.max(np.abs(radau-standard)))),
            "max_one_sided_cross_audit_increase": float(max(np.max(tight-standard), np.max(radau-standard))),
            "all_cross_audits_below_zero": bool(np.all(tight <= 0) and np.all(radau <= 0)),
            "rows": rows,
        }
        print(problem, json.dumps({key: value for key, value in report["benchmarks"][problem].items() if key != "rows"}, indent=2), flush=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
