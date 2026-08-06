"""Re-evaluate frozen VALC policies with a conservative negative HDS margin.

No network is retrained.  For each frozen seed, the selected deployment model
is reconstructed, its nominal controls are corrected using
``g_hat_max <= -eta_safe``, and every accepted sequence is independently
re-audited with tighter DOP853 settings.  New artifacts are written to a
separate directory so the historical results remain intact.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from kkt_collocation.run_cstr_full_simulation import CSTRConfig, Policy as CSTRPolicy, cstr_ode, lhs_states, objective as cstr_objective, path_constraint, path_derivative
from kkt_collocation.run_penicillin_ablation import DT as PEN_DT, UMAX as PEN_UMAX, Policy as PenPolicy, g as pen_g, gdot as pen_gdot, ode as pen_ode, terminal_product
from kkt_collocation.run_vdp_ablation import constraint as vdp_g, constraint_derivative as vdp_gdot, terminal_cost, vdp_ode
from kkt_collocation.train_vdp_kkt_policy import KKTPolicyValueNetwork, TrainConfig


VDP_SEEDS = (20260751, 20260752, 20260753)
PEN_SEEDS = (20260761, 20260762, 20260763)
CSTR_TRAIN_SEEDS = (20260718, 20260725, 20260726)
CSTR_TEST_SEEDS = (20260724, 20260825, 20260826)
FIELDS = (
    "problem", "training_seed", "sample_index", "initial_1", "initial_2",
    "nominal_hds_max_g", "applied_hds_max_g", "independent_hds_max_g",
    "accepted", "fallback", "nominal_objective", "applied_objective",
    "objective_change", "corrected_segments", "mean_abs_lambda_minus_one",
    "inference_seconds", "correction_seconds",
)

_PROBLEM: str | None = None
_DECISION: HDSLambdaCorrector | None = None
_INDEPENDENT: HDSLambdaCorrector | None = None
_DURATION: float | None = None
_CSTR_CFG: CSTRConfig | None = None


def checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _worker_init(problem: str, margin: float, grid_size: int, cstr_config: dict | None) -> None:
    global _PROBLEM, _DECISION, _INDEPENDENT, _DURATION, _CSTR_CFG
    _PROBLEM = problem
    if problem == "VDP":
        funcs, bounds, _DURATION = (vdp_ode, vdp_g, vdp_gdot), (-0.3, 1.0), 0.5
    elif problem == "Penicillin":
        funcs, bounds, _DURATION = (pen_ode, pen_g, pen_gdot), (0.0, PEN_UMAX), PEN_DT
    else:
        if cstr_config is None:
            raise ValueError("CSTR configuration is required")
        _CSTR_CFG = CSTRConfig(**cstr_config)
        funcs = (
            lambda t, x, u: cstr_ode(t, x, u, _CSTR_CFG),
            lambda x: path_constraint(x, _CSTR_CFG),
            lambda x, u: path_derivative(x, u, _CSTR_CFG),
        )
        bounds = (_CSTR_CFG.cooling_min, _CSTR_CFG.cooling_max)
        _DURATION = _CSTR_CFG.zoh_duration
    _DECISION = HDSLambdaCorrector(
        *funcs, bounds,
        HDSLambdaConfig(grid_size=grid_size, safety_margin=margin, rtol=1e-10,
                        atol=1e-12, max_step_fraction=100.0, integrator="DOP853"),
    )
    _INDEPENDENT = HDSLambdaCorrector(
        *funcs, bounds,
        HDSLambdaConfig(grid_size=grid_size, safety_margin=margin, rtol=1e-12,
                        atol=1e-14, max_step_fraction=400.0, integrator="DOP853"),
    )


def _objective(state: np.ndarray, controls: np.ndarray) -> float:
    if _DECISION is None or _DURATION is None or _PROBLEM is None:
        raise RuntimeError("worker is not initialized")
    if _PROBLEM == "VDP":
        return terminal_cost(state, controls, _DECISION, _DURATION)
    if _PROBLEM == "Penicillin":
        return -terminal_product(float(state[1]), controls, _DECISION)
    if _CSTR_CFG is None:
        raise RuntimeError("CSTR configuration is unavailable")
    return cstr_objective(state, controls, _CSTR_CFG)


def _evaluate_one(task: tuple[int, int, np.ndarray, np.ndarray, float]) -> tuple[dict, np.ndarray | None]:
    if _DECISION is None or _INDEPENDENT is None or _DURATION is None or _PROBLEM is None:
        raise RuntimeError("worker is not initialized")
    training_seed, sample_index, state, nominal, inference = task
    nominal_peak = _DECISION.audit(state, nominal, _DURATION)
    nominal_objective = _objective(state, nominal)
    started = time.perf_counter()
    outcome = _DECISION.correct(state, nominal, _DURATION)
    correction_seconds = time.perf_counter() - started
    if not outcome.accepted or outcome.controls is None:
        row = {
            "problem": _PROBLEM, "training_seed": training_seed, "sample_index": sample_index,
            "initial_1": state[0], "initial_2": state[1], "nominal_hds_max_g": nominal_peak,
            "applied_hds_max_g": np.nan, "independent_hds_max_g": np.nan,
            "accepted": False, "fallback": True, "nominal_objective": nominal_objective,
            "applied_objective": np.nan, "objective_change": np.nan,
            "corrected_segments": sum(segment.corrected for segment in outcome.segments),
            "mean_abs_lambda_minus_one": np.nan, "inference_seconds": inference,
            "correction_seconds": correction_seconds,
        }
        return row, None
    applied = outcome.controls
    decision_peak = _DECISION.audit(state, applied, _DURATION)
    independent_peak = _INDEPENDENT.audit(state, applied, _DURATION)
    independently_safe = bool(independent_peak <= 0.0)
    applied_objective = _objective(state, applied) if independently_safe else np.nan
    lambdas = np.asarray([segment.lambda_value for segment in outcome.segments if segment.lambda_value is not None], dtype=float)
    row = {
        "problem": _PROBLEM, "training_seed": training_seed, "sample_index": sample_index,
        "initial_1": state[0], "initial_2": state[1], "nominal_hds_max_g": nominal_peak,
        "applied_hds_max_g": decision_peak, "independent_hds_max_g": independent_peak,
        "accepted": independently_safe, "fallback": not independently_safe,
        "nominal_objective": nominal_objective, "applied_objective": applied_objective,
        "objective_change": applied_objective - nominal_objective if independently_safe else np.nan,
        "corrected_segments": sum(segment.corrected for segment in outcome.segments),
        "mean_abs_lambda_minus_one": float(np.mean(np.abs(lambdas - 1.0))) if len(lambdas) else 0.0,
        "inference_seconds": inference, "correction_seconds": correction_seconds,
    }
    return row, applied if independently_safe else None


def _predict_vdp(directory: Path) -> tuple[np.ndarray, np.ndarray, float]:
    states = np.load(directory / "test_states.npy")
    stored = checkpoint(directory / "models.pth")
    branch = json.loads((directory / "summary.json").read_text(encoding="utf-8"))["adaptive_gate"]["selected_branch"]
    key = "KKT" if branch == "S+KKT" else "S"
    model = KKTPolicyValueNetwork(TrainConfig()); model.load_state_dict(stored[key]); model.eval()
    mean, std = stored["normalization"][key]
    x = torch.tensor((states[:, :2] - np.asarray(mean)) / np.asarray(std), dtype=torch.float32)
    started = time.perf_counter()
    with torch.no_grad(): controls = model(x)[1].numpy()
    return states, controls, (time.perf_counter() - started) / len(states)


def _predict_penicillin(directory: Path) -> tuple[np.ndarray, np.ndarray, float]:
    x2 = np.load(directory / "test_x2.npy")
    states = np.column_stack((np.ones(len(x2)), x2, np.full(len(x2), 0.001), np.full(len(x2), 250.0)))
    stored = checkpoint(directory / "models.pth")
    branch = json.loads((directory / "summary.json").read_text(encoding="utf-8"))["adaptive_gate"]["selected_branch"]
    key = "true_KKT" if "true-KKT" in branch else "S"
    model = PenPolicy(); model.load_state_dict(stored[key]); model.eval()
    mean, std = stored["normalization"][key]
    x = torch.tensor(((x2 - float(mean)) / float(std))[:, None], dtype=torch.float32)
    started = time.perf_counter()
    with torch.no_grad(): controls = model(x)[1].numpy()
    return states, controls, (time.perf_counter() - started) / len(states)


def _predict_cstr(directory: Path, training_seed: int, test_seed: int) -> tuple[np.ndarray, np.ndarray, float, dict]:
    stored = checkpoint(directory / f"cstr_seed{training_seed}.pth")
    cfg_dict = dict(stored["config"])
    cfg = CSTRConfig(**cfg_dict)
    states = lhs_states(cfg, 400, test_seed)
    model = CSTRPolicy(cfg); model.load_state_dict(stored["model"]); model.eval()
    x = torch.tensor((states - np.asarray(stored["mean"])) / np.asarray(stored["std"]), dtype=torch.float32)
    started = time.perf_counter()
    with torch.no_grad(): controls = model(x)[1].numpy()
    return states, controls, (time.perf_counter() - started) / len(states), cfg_dict


def _mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(array)), "sample_std": float(np.std(array, ddof=1))}


def _seed_summary(rows: list[dict], margin: float) -> dict[str, Any]:
    accepted = np.asarray([row["accepted"] for row in rows], dtype=bool)
    nominal = np.asarray([row["nominal_hds_max_g"] for row in rows], dtype=float)
    independent = np.asarray([row["independent_hds_max_g"] for row in rows], dtype=float)
    return {
        "samples": len(rows), "safety_margin": margin,
        "nominal_violation_rate_percent": float(100 * np.mean(nominal > 0.0)),
        "nominal_peak_max": float(nominal.max()),
        "accepted_rate_percent": float(100 * np.mean(accepted)),
        "fallback_rate_percent": float(100 * np.mean(~accepted)),
        "accepted_independent_peak_max": float(np.nanmax(independent[accepted])) if np.any(accepted) else np.nan,
        "corrected_segments_mean": float(np.mean([row["corrected_segments"] for row in rows])),
        "mean_objective_change": float(np.nanmean([row["objective_change"] for row in rows])),
        "mean_inference_ms": float(1e3 * np.mean([row["inference_seconds"] for row in rows])),
        "mean_correction_ms": float(1e3 * np.mean([row["correction_seconds"] for row in rows])),
    }


def evaluate_seed(problem: str, training_seed: int, states: np.ndarray, controls: np.ndarray,
                  inference: float, output: Path, margin: float, grid_size: int,
                  workers: int, cstr_config: dict | None) -> dict:
    tasks = [(training_seed, index, np.asarray(state, dtype=float), np.asarray(control, dtype=float), inference)
             for index, (state, control) in enumerate(zip(states, controls))]
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(problem, margin, grid_size, cstr_config)) as pool:
        results = list(pool.map(_evaluate_one, tasks, chunksize=2))
    rows = [item[0] for item in results]
    applied = np.asarray([item[1] if item[1] is not None else np.full(controls.shape[1], np.nan) for item in results])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output / "population_controls.npz", initial_states=states,
                        nominal_controls=controls, applied_controls=applied)
    summary = _seed_summary(rows, margin)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin", type=float, default=1e-6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "conservative_margin_1e6")
    parser.add_argument("--problems", nargs="+", choices=("VDP", "Penicillin", "CSTR"),
                        default=("VDP", "Penicillin", "CSTR"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    all_summaries: dict[str, dict] = {}
    for problem in args.problems:
        seeds = VDP_SEEDS if problem == "VDP" else (PEN_SEEDS if problem == "Penicillin" else CSTR_TRAIN_SEEDS)
        per_seed = {}
        for index, seed in enumerate(seeds):
            target = args.output / f"{problem.lower()}_seed{seed}"
            completed_summary = target / "summary.json"
            if completed_summary.exists():
                per_seed[str(seed)] = json.loads(completed_summary.read_text(encoding="utf-8"))
                print(f"{problem} seed {seed}: reusing completed conservative-margin audit", flush=True)
                continue
            if problem == "VDP":
                source = ROOT / "kkt_collocation" / "results" / f"final_multiseed_vdp900_penalty_seed{seed}"
                states, controls, inference = _predict_vdp(source); cfg = None; grid = 31
            elif problem == "Penicillin":
                source = ROOT / "kkt_collocation" / "results" / f"final_multiseed_penicillin400_penalty_seed{seed}"
                states, controls, inference = _predict_penicillin(source); cfg = None; grid = 31
            else:
                source = ROOT / "kkt_collocation" / "results" / "cstr_multiseed_test1200_900_clean"
                states, controls, inference, cfg = _predict_cstr(source, seed, CSTR_TEST_SEEDS[index]); grid = 31
            print(f"{problem} seed {seed}: evaluating {len(states)} frozen points", flush=True)
            per_seed[str(seed)] = evaluate_seed(problem, seed, states, controls, inference, target,
                                                args.margin, grid, args.workers, cfg)
            print(json.dumps(per_seed[str(seed)], indent=2), flush=True)
        metric_names = (
            "nominal_violation_rate_percent", "accepted_rate_percent", "fallback_rate_percent",
            "accepted_independent_peak_max", "corrected_segments_mean", "mean_objective_change",
            "mean_inference_ms", "mean_correction_ms",
        )
        all_summaries[problem] = {
            "per_seed": per_seed,
            "aggregate": {name: _mean_sd([per_seed[str(seed)][name] for seed in seeds]) for name in metric_names},
        }
    report = {
        "safety_rule": f"decision HDS peak <= -{args.margin:g}; independent tight-audit peak <= 0",
        "decision_integrator": {"method": "DOP853", "rtol": 1e-10, "atol": 1e-12, "max_step": "Delta t / 100"},
        "independent_audit": {"method": "DOP853", "rtol": 1e-12, "atol": 1e-14, "max_step": "Delta t / 400"},
        "cross_integrator_calibration": "boundary-critical trajectories also checked with Radau; see hds_margin_calibration artifacts",
        "benchmarks": all_summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
