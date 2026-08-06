"""Benchmark final policies against independently re-solved discretized NLPs.

The selected points are a fixed, evenly spaced subset of the final held-out
test set.  For each point, the exact same direct-RK4 NLP discretization that
generated the teacher labels is re-solved.  The resulting value is therefore
an *NLP reference*, not a claim of a global optimum of the continuous-time
infinite-dimensional problem.

The script also measures single-instance neural inference and the sequential
pre-execution HDS--lambda correction time.  The latter includes the candidate
simulations performed by ``correct`` but excludes the optional post-hoc audit
and terminal-objective evaluation used only for reporting.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kkt_collocation.generate_penicillin_kkt_data import PenicillinConfig, ReducedPenicillinProblem
from kkt_collocation.generate_vdp_kkt_data import VDPDirectTranscription, VDPTranscriptionConfig
from kkt_collocation.run_penicillin_ablation import (  # noqa: E402
    DT as PEN_DURATION,
    Policy as PenicillinPolicy,
    UMAX as PEN_UMAX,
    g as pen_constraint,
    gdot as pen_constraint_derivative,
    ode as pen_ode,
    terminal_product,
)
from kkt_collocation.run_vdp_ablation import (  # noqa: E402
    constraint as vdp_constraint,
    constraint_derivative as vdp_constraint_derivative,
    terminal_cost,
    vdp_ode,
)
from kkt_collocation.train_vdp_kkt_policy import KKTPolicyValueNetwork, TrainConfig
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector


_VDP_WORKER: VDPDirectTranscription | None = None
_PEN_WORKER: ReducedPenicillinProblem | None = None


def _init_vdp_worker(config: VDPTranscriptionConfig) -> None:
    global _VDP_WORKER
    _VDP_WORKER = VDPDirectTranscription(config)


def _solve_vdp_worker(task: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    if _VDP_WORKER is None:
        raise RuntimeError("VDP NLP worker was not initialized")
    state, warm = task
    return _VDP_WORKER.solve(state, warm)


def _init_pen_worker(config: PenicillinConfig, seed_x2: np.ndarray, seed_controls: np.ndarray) -> None:
    global _PEN_WORKER
    _PEN_WORKER = ReducedPenicillinProblem(config, seed_x2, seed_controls)


def _solve_pen_worker(x2: float) -> dict[str, Any]:
    if _PEN_WORKER is None:
        raise RuntimeError("Penicillin NLP worker was not initialized")
    return _PEN_WORKER.solve(x2)


def _subset_indices(total: int, count: int) -> np.ndarray:
    if not 1 <= count <= total:
        raise ValueError(f"reference count must be in [1, {total}]")
    return np.linspace(0, total - 1, count, dtype=int)


def _single_instance_vdp(model, mean, std, states: np.ndarray) -> tuple[np.ndarray, float]:
    """Measure ordinary one-state deployment inference, rather than batch throughput."""
    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    with torch.no_grad():
        for _ in range(10):
            model(torch.tensor(((states[:1, :2] - mean) / std), dtype=torch.float32))
        controls, timings = [], []
        for state in states:
            x = torch.tensor(((state[None, :2] - mean) / std), dtype=torch.float32)
            start = time.perf_counter()
            _, u = model(x)
            timings.append(time.perf_counter() - start)
            controls.append(u.numpy()[0])
    return np.asarray(controls), float(np.mean(timings))


def _single_instance_penicillin(model, mean: float, std: float, x2: np.ndarray) -> tuple[np.ndarray, float]:
    with torch.no_grad():
        for _ in range(10):
            model(torch.tensor(((x2[:1] - mean) / std)[:, None], dtype=torch.float32))
        controls, timings = [], []
        for value in x2:
            x = torch.tensor([[(value - mean) / std]], dtype=torch.float32)
            start = time.perf_counter()
            _, u = model(x)
            timings.append(time.perf_counter() - start)
            controls.append(u.numpy()[0])
    return np.asarray(controls), float(np.mean(timings))


def _evaluate_policy(
    name: str, initial_states: np.ndarray, controls: np.ndarray, reference_objectives: np.ndarray,
    corrector: HDSLambdaCorrector, duration: float, objective,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_index, (initial, nominal, reference) in enumerate(zip(initial_states, controls, reference_objectives)):
        nominal_peak = corrector.audit(initial, nominal, duration)
        nominal_objective = objective(initial, nominal)
        start = time.perf_counter()
        certificate = corrector.correct(initial, nominal, duration)
        correction_seconds = time.perf_counter() - start
        if certificate.accepted:
            applied = certificate.controls
            applied_objective = objective(initial, applied)
            applied_peak = corrector.audit(initial, applied, duration)
            lambdas = np.asarray([segment.lambda_value for segment in certificate.segments], dtype=float)
            corrected_segments = int(sum(segment.corrected for segment in certificate.segments))
        else:
            applied_objective, applied_peak = np.nan, np.nan
            lambdas = np.asarray([], dtype=float)
            corrected_segments = int(sum(segment.corrected for segment in certificate.segments))
        rows.append({
            "method": name,
            "sample_index": sample_index,
            "raw_hds_max_g": float(nominal_peak),
            "accepted": bool(certificate.accepted),
            "fallback": bool(certificate.requires_reoptimization),
            "applied_hds_max_g": float(applied_peak),
            "nlp_reference_objective": float(reference),
            "nominal_objective": float(nominal_objective),
            "applied_objective": float(applied_objective),
            "nominal_objective_gap": float(nominal_objective - reference),
            "applied_objective_gap": float(applied_objective - reference),
            "objective_change": float(applied_objective - nominal_objective),
            "corrected_segments": corrected_segments,
            "mean_abs_lambda_minus_one": float(np.mean(np.abs(lambdas - 1.0))),
            "hds_lambda_correction_seconds": float(correction_seconds),
        })
    return rows


def _summarise(rows: list[dict[str, Any]], inference_seconds: dict[str, float], nlp_seconds: np.ndarray) -> dict[str, Any]:
    report: dict[str, Any] = {
        "reference_samples": int(len(nlp_seconds)),
        "nlp_reference": {
            "mean_solve_seconds": float(np.mean(nlp_seconds)),
            "median_solve_seconds": float(np.median(nlp_seconds)),
            "max_solve_seconds": float(np.max(nlp_seconds)),
        },
        "methods": {},
        "timing_definition": {
            "nlp": "wall-clock direct-RK4 NLP solve, including its internal HDS acceptance audit",
            "inference": "mean CPU wall-clock time for one initial state after a warm-up",
            "hds_lambda": "sequential pre-execution HDS-lambda candidate search; excludes report-only post-correction audit and objective evaluation",
        },
    }
    for name in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == name]
        values = lambda key: np.asarray([row[key] for row in group], dtype=float)
        report["methods"][name] = {
            "raw_violation_rate": float(np.mean(values("raw_hds_max_g") > 1e-8)),
            "accepted_rate": float(np.mean(values("accepted"))),
            "fallback_rate": float(np.mean(values("fallback"))),
            "max_applied_hds_g": float(np.nanmax(values("applied_hds_max_g"))),
            "mean_nominal_objective_gap": float(np.mean(values("nominal_objective_gap"))),
            "mean_applied_objective_gap": float(np.nanmean(values("applied_objective_gap"))),
            "mean_objective_change": float(np.nanmean(values("objective_change"))),
            "mean_corrected_segments": float(np.mean(values("corrected_segments"))),
            "mean_abs_lambda_minus_one": float(np.nanmean(values("mean_abs_lambda_minus_one"))),
            "mean_hds_lambda_correction_seconds": float(np.mean(values("hds_lambda_correction_seconds"))),
            "mean_inference_seconds": float(inference_seconds[name]),
        }
    return report


def _run_vdp(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_dir = args.vdp_result_dir
    states_all = np.load(result_dir / "test_states.npy")
    indices = _subset_indices(len(states_all), args.reference_samples)
    states = states_all[indices]
    checkpoint = torch.load(result_dir / "models.pth", map_location="cpu", weights_only=False)
    policies, inference = {}, {}
    for label, key in (("S", "S"), ("S+KKT", "KKT")):
        model = KKTPolicyValueNetwork(TrainConfig()).eval()
        model.load_state_dict(checkpoint[key])
        mean, std = checkpoint["normalization"][key]
        policies[label], inference[label] = _single_instance_vdp(model, mean, std, states)
    config = VDPTranscriptionConfig(substeps_per_zoh=10, collocation_safety_margin=1e-3)
    tasks = [(state, policies["S"][index]) for index, state in enumerate(states)]
    if args.workers == 1:
        _init_vdp_worker(config)
        references = [_solve_vdp_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_vdp_worker, initargs=(config,)) as pool:
            references = list(pool.map(_solve_vdp_worker, tasks, chunksize=1))
    reference_objective = np.asarray([record["objective"] for record in references], dtype=float)
    nlp_seconds = np.asarray([record["solve_seconds"] for record in references], dtype=float)
    corrector = HDSLambdaCorrector(
        vdp_ode, vdp_constraint, vdp_constraint_derivative, (-0.3, 1.0),
        HDSLambdaConfig(grid_size=31, max_step_fraction=100.0),
    )
    objective = lambda initial, u: terminal_cost(initial, u, corrector, 0.5)
    rows = []
    for name in ("S", "S+KKT"):
        rows.extend(_evaluate_policy(name, states, policies[name], reference_objective, corrector, 0.5, objective))
    report = _summarise(rows, inference, nlp_seconds)
    report.update({
        "problem": "VDP",
        "reference_discretization": "10 ZOH controls, 10 RK4 substeps per ZOH, 101 path nodes, 1e-3 nodal interior margin",
        "test_indices": indices.tolist(),
        "source_final_result": str(result_dir),
    })
    return rows, report


def _run_penicillin(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_dir = args.penicillin_result_dir
    x2_all = np.load(result_dir / "test_x2.npy")
    indices = _subset_indices(len(x2_all), args.reference_samples)
    x2 = x2_all[indices]
    states = np.column_stack((np.ones(len(x2)), x2, np.full(len(x2), 0.001), np.full(len(x2), 250.0)))
    checkpoint = torch.load(result_dir / "models.pth", map_location="cpu", weights_only=False)
    policies, inference = {}, {}
    for label, key in (("S", "S"), ("S+true-KKT", "true_KKT")):
        model = PenicillinPolicy().eval()
        model.load_state_dict(checkpoint[key])
        mean, std = checkpoint["normalization"][key]
        policies[label], inference[label] = _single_instance_penicillin(model, float(mean), float(std), x2)
    with args.penicillin_data.open("rb") as handle:
        teacher = pickle.load(handle)
    seed_x2 = np.asarray(teacher["initial_state"], dtype=float)[:, 1]
    seed_controls = np.asarray(teacher["optimal_controls"], dtype=float)
    config = PenicillinConfig(substeps_per_zoh=80, safety_margin=1e-3)
    if args.workers == 1:
        _init_pen_worker(config, seed_x2, seed_controls)
        references = [_solve_pen_worker(float(value)) for value in x2]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_pen_worker,
                                 initargs=(config, seed_x2, seed_controls)) as pool:
            references = list(pool.map(_solve_pen_worker, [float(value) for value in x2], chunksize=1))
    reference_objective = np.asarray([record["objective"] for record in references], dtype=float)
    nlp_seconds = np.asarray([record["solve_seconds"] for record in references], dtype=float)
    corrector = HDSLambdaCorrector(
        pen_ode, pen_constraint, pen_constraint_derivative, (0.0, PEN_UMAX),
        HDSLambdaConfig(grid_size=31, max_step_fraction=100.0),
    )
    objective = lambda initial, u: -terminal_product(float(initial[1]), u, corrector)
    rows = []
    for name in ("S", "S+true-KKT"):
        rows.extend(_evaluate_policy(name, states, policies[name], reference_objective, corrector, PEN_DURATION, objective))
    report = _summarise(rows, inference, nlp_seconds)
    report.update({
        "problem": "Penicillin",
        "reference_discretization": "10 ZOH controls, 80 RK4 substeps per ZOH, 801 path nodes, 1e-3 nodal interior margin",
        "test_indices": indices.tolist(),
        "source_final_result": str(result_dir),
    })
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("vdp", "penicillin"), required=True)
    parser.add_argument("--reference-samples", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vdp-result-dir", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "final_vdp_900_adaptive_true_kkt_e200_test400_seed20260731")
    parser.add_argument("--penicillin-result-dir", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "final_penicillin_true_kkt_e200_w001_test400_seed20260741")
    parser.add_argument("--penicillin-data", type=Path,
                        default=ROOT / "kkt_collocation" / "data" / "penicillin_kkt_400_true_duals.pkl")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, report = _run_vdp(args) if args.problem == "vdp" else _run_penicillin(args)
    with (args.output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    np.save(args.output / "reference_test_indices.npy", np.asarray(report["test_indices"], dtype=int))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
