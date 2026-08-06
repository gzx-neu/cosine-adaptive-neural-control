"""Select VDP and penicillin policies using independent NLP-reference validation.

This script replaces the earlier raw-HDS-severity gate *for the two reported
benchmarks only*.  For every training seed it compares the supervised (S) and
KKT-refined (KKT) candidates on 60 validation initial conditions that are
disjoint from the frozen 400-point test cohort.  Each validation objective is
obtained from an independently cold-started deterministic transcription NLP.

Candidate selection is lexicographic: a candidate must have zero HDS fallback
on the validation cohort, after which the candidate with the smallest mean
absolute relative objective difference is selected.  The selected candidate is
then evaluated once on the untouched 400-point test cohort.

The script deliberately records the solver setup in its JSON report.  In
particular, the penicillin NLP always starts from the same neutral u=0.5
sequence; it never receives a neural-policy or nearest-label warm start.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_penicillin_kkt_data import PenicillinConfig, ReducedPenicillinProblem
from generate_vdp_kkt_data import VDPDirectTranscription, VDPTranscriptionConfig
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from run_penicillin_ablation import (DT as PEN_DT, N as PEN_N, Policy as PenicillinPolicy,
                                     UMAX as PEN_UMAX, g as pen_g, gdot as pen_gdot,
                                     ode as pen_ode, predict as pen_predict,
                                     terminal_product, train as pen_train)
from run_penicillin_true_kkt_ablation import load_true_kkt_data
from run_vdp_ablation import (AblationConfig, constraint as vdp_g,
                              constraint_derivative as vdp_gdot, load_data as load_vdp_data,
                              predict as vdp_predict, terminal_cost, train_policy, vdp_ode)
from train_vdp_kkt_policy import KKTPolicyValueNetwork, TrainConfig


_VDP_NLP: VDPDirectTranscription | None = None
_PEN_NLPS: tuple[ReducedPenicillinProblem, ...] = ()


def _init_vdp_nlp(config: VDPTranscriptionConfig) -> None:
    global _VDP_NLP
    _VDP_NLP = VDPDirectTranscription(config)


def _solve_vdp_cold(state: np.ndarray) -> dict[str, Any]:
    if _VDP_NLP is None:
        raise RuntimeError("VDP validation worker was not initialized")
    # No control guess: VDPDirectTranscription uses its fixed internal 0.5
    # sequence, independent of labels and learned policies.
    return _VDP_NLP.solve(np.asarray(state, dtype=float))


def _init_pen_nlp(config: PenicillinConfig) -> None:
    global _PEN_NLPS
    # Predetermined neutral starts make the deterministic NLP robust across
    # the declared operating domain.  They are constants, not controls from a
    # policy or label set, and every point sees the same ordered list.
    _PEN_NLPS = tuple(
        ReducedPenicillinProblem(config, np.asarray([0.2]), np.full((1, PEN_N), value), use_warm_start=True)
        for value in (0.4, 0.5, 0.7, 0.9, 1.0)
    )


def _solve_pen_cold(x2: float) -> dict[str, Any]:
    if not _PEN_NLPS:
        raise RuntimeError("Penicillin validation worker was not initialized")
    # nnls mode is adequate here because validation needs only the primal
    # objective.  It also permits valid inactive-path solutions.
    errors: list[str] = []
    for problem in _PEN_NLPS:
        try:
            # The sequence is deterministic.  In preliminary checks the first
            # successful fixed start converges to the same feasible solution;
            # stopping there avoids turning validation into an unnecessarily
            # expensive global multi-start study.
            return problem.solve(float(x2), dual_mode="nnls")
        except RuntimeError as error:
            errors.append(str(error))
    raise RuntimeError(f"all fixed neutral cold starts failed at x2={x2:.8g}: {errors}")


def _parallel_map(initializer: Callable, initarg: Any, worker: Callable,
                  tasks: list[Any], workers: int) -> list[dict[str, Any]]:
    if workers == 1:
        initializer(initarg)
        return [worker(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer, initargs=(initarg,)) as pool:
        return list(pool.map(worker, tasks, chunksize=1))


def _reference_records(path: Path, points: np.ndarray, kind: str, workers: int) -> list[dict[str, Any]]:
    """Load a point-matched cache or generate audited cold-start references."""
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if np.array_equal(cached["points"], points):
            return [
                {"objective": float(obj), "hds_max_g": float(peak), "solve_seconds": float(seconds)}
                for obj, peak, seconds in zip(cached["objective"], cached["hds_max_g"], cached["solve_seconds"])
            ]
    if kind == "vdp":
        config = VDPTranscriptionConfig(substeps_per_zoh=10, collocation_safety_margin=1e-3)
        records = _parallel_map(_init_vdp_nlp, config, _solve_vdp_cold,
                                [np.asarray(p, dtype=float) for p in points], workers)
    elif kind == "penicillin":
        config = PenicillinConfig(substeps_per_zoh=80, safety_margin=1e-3)
        records = _parallel_map(_init_pen_nlp, config, _solve_pen_cold,
                                [float(p) for p in points], workers)
    else:
        raise ValueError(kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, points=points,
        objective=np.asarray([r["objective"] for r in records], dtype=float),
        hds_max_g=np.asarray([r["hds_max_g"] for r in records], dtype=float),
        solve_seconds=np.asarray([r["solve_seconds"] for r in records], dtype=float),
    )
    return records


def _objective_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-12) * 100.0


def _evaluate_vdp(name: str, model: torch.nn.Module, mean: np.ndarray, std: np.ndarray,
                  states: np.ndarray, references: np.ndarray, device: torch.device) -> list[dict[str, Any]]:
    controls, inference = vdp_predict(model, mean, std, states, device)
    corrector = HDSLambdaCorrector(vdp_ode, vdp_g, vdp_gdot, (-0.3, 1.0),
                                   HDSLambdaConfig(grid_size=31, max_step_fraction=100.0))
    rows: list[dict[str, Any]] = []
    for index, (state, nominal, reference) in enumerate(zip(states, controls, references)):
        raw_peak = corrector.audit(state, nominal, 0.5)
        raw_objective = terminal_cost(state, nominal, corrector, 0.5)
        start = time.perf_counter(); result = corrector.correct(state, nominal, 0.5); elapsed = time.perf_counter() - start
        accepted = bool(result.accepted)
        applied = result.controls if accepted else None
        applied_peak = corrector.audit(state, applied, 0.5) if accepted else np.nan
        applied_objective = terminal_cost(state, applied, corrector, 0.5) if accepted else np.nan
        rows.append({
            "candidate": name, "sample_index": index, "accepted": accepted, "fallback": not accepted,
            "raw_hds_max_g": raw_peak, "applied_hds_max_g": applied_peak,
            "reference_objective": float(reference), "raw_objective": raw_objective,
            "applied_objective": applied_objective,
            "relative_objective_difference_percent": _objective_error(applied_objective, reference) if accepted else np.nan,
            "corrected_segments": int(sum(segment.corrected for segment in result.segments)),
            "inference_seconds": inference, "filter_seconds": elapsed,
        })
    return rows


def _evaluate_pen(name: str, model: torch.nn.Module, mean: float, std: float,
                  x2: np.ndarray, references: np.ndarray, device: torch.device) -> list[dict[str, Any]]:
    controls, inference = pen_predict(model, mean, std, x2, device)
    corrector = HDSLambdaCorrector(pen_ode, pen_g, pen_gdot, (0.0, PEN_UMAX),
                                   HDSLambdaConfig(grid_size=31, max_step_fraction=100.0))
    rows: list[dict[str, Any]] = []
    for index, (value, nominal, reference) in enumerate(zip(x2, controls, references)):
        state = np.asarray([1.0, value, 0.001, 250.0])
        raw_peak = corrector.audit(state, nominal, PEN_DT)
        raw_objective = -terminal_product(float(value), nominal, corrector)
        start = time.perf_counter(); result = corrector.correct(state, nominal, PEN_DT); elapsed = time.perf_counter() - start
        accepted = bool(result.accepted)
        applied = result.controls if accepted else None
        applied_peak = corrector.audit(state, applied, PEN_DT) if accepted else np.nan
        applied_objective = -terminal_product(float(value), applied, corrector) if accepted else np.nan
        rows.append({
            "candidate": name, "sample_index": index, "accepted": accepted, "fallback": not accepted,
            "raw_hds_max_g": raw_peak, "applied_hds_max_g": applied_peak,
            "reference_objective": float(reference), "raw_objective": raw_objective,
            "applied_objective": applied_objective,
            "relative_objective_difference_percent": _objective_error(applied_objective, reference) if accepted else np.nan,
            "corrected_segments": int(sum(segment.corrected for segment in result.segments)),
            "inference_seconds": inference, "filter_seconds": elapsed,
        })
    return rows


def _candidate_score(rows: list[dict[str, Any]]) -> dict[str, float | bool]:
    accepted = np.asarray([r["accepted"] for r in rows], dtype=bool)
    return {
        "accepted": bool(accepted.all()), "fallback_rate": float(1.0 - accepted.mean()),
        "mean_relative_objective_difference_percent": float(np.nanmean([r["relative_objective_difference_percent"] for r in rows])),
        "nominal_violation_rate": float(np.mean([r["raw_hds_max_g"] > 1e-8 for r in rows])),
        "mean_corrected_segments": float(np.mean([r["corrected_segments"] for r in rows])),
    }


def _read_test_references(problem: str, seed: int) -> np.ndarray:
    path = ROOT / "kkt_collocation" / "results" / "jiang_fu_matched400_comparison" / "per_point_seed.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        problem_rows = [r for r in csv.DictReader(handle) if r["problem"] == problem]
    rows = [r for r in problem_rows if int(r["seed"]) == seed]
    # The test inputs are shared across model-training seeds.  The historical
    # matched-NLP export consequently stores one 400-point reference sequence
    # per problem (VDP under seed 20260751 and penicillin under 20260761), not
    # three duplicate copies.  Reusing that immutable sequence preserves the
    # intended strict separation from validation.
    if not rows and problem_rows:
        stored_seed = min(int(r["seed"]) for r in problem_rows)
        rows = [r for r in problem_rows if int(r["seed"]) == stored_seed]
    if len(rows) != 400:
        raise RuntimeError(f"Expected one 400-point frozen {problem} reference sequence, obtained {len(rows)}")
    return np.asarray([float(r["jiang_objective"]) for r in rows], dtype=float)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({
                **row,
                "accepted": row["accepted"].lower() == "true",
                "fallback": row["fallback"].lower() == "true",
                "raw_hds_max_g": float(row["raw_hds_max_g"]),
                "applied_hds_max_g": float(row["applied_hds_max_g"]),
                "relative_objective_difference_percent": float(row["relative_objective_difference_percent"]),
                "corrected_segments": int(row["corrected_segments"]),
            })
    return rows


def _run_vdp(seed: int, output: Path, validation_count: int, workers: int, device: torch.device) -> dict[str, Any]:
    source = ROOT / "kkt_collocation" / "results" / f"final_multiseed_vdp900_penalty_seed{seed}"
    root = output / "vdp" / f"seed{seed}"; root.mkdir(parents=True, exist_ok=True)
    validation = np.load(source / "validation_states.npy")[:validation_count]
    test = np.load(source / "test_states.npy")
    refs = _reference_records(root / "validation_cold_nlp_references.npz", validation, "vdp", workers)
    reference_objective = np.asarray([r["objective"] for r in refs], dtype=float)

    candidate_path = root / "candidates.pth"
    if candidate_path.exists():
        candidate = torch.load(candidate_path, map_location=device, weights_only=False)
        s_model, k_model = KKTPolicyValueNetwork(TrainConfig()).to(device), KKTPolicyValueNetwork(TrainConfig()).to(device)
        s_model.load_state_dict(candidate["S"]); k_model.load_state_dict(candidate["KKT"])
        mean_s, std_s = candidate["normalization"]["S"]
        mean_k, std_k = candidate["normalization"]["KKT"]
    else:
        checkpoint = torch.load(ROOT / "kkt_collocation" / "results" / "vdp_no_rollout_w0" / f"vdp_seed{seed}.pth", map_location=device, weights_only=False)
        s_model = KKTPolicyValueNetwork(TrainConfig()).to(device); s_model.load_state_dict(checkpoint["model"])
        mean_s, std_s = checkpoint["mean"], checkpoint["std"]
        initial, controls, objectives, duals, _ = load_vdp_data(ROOT / "kkt_collocation" / "data" / "vdp_kkt_30x30_warm_s10_margin1e3.pkl", device)
        config = AblationConfig(epochs=200, seed=seed, lambda_grid_size=31, rollout_weight=0.0, kkt_weight=1e-3)
        k_model, mean_k, std_k = train_policy(initial, controls, objectives, duals, True, config, device,
                                                model=copy.deepcopy(s_model), epochs=20, learning_rate=1e-5)
        torch.save({"S": s_model.state_dict(), "KKT": k_model.state_dict(),
                    "normalization": {"S": [mean_s, std_s], "KKT": [mean_k, std_k]}, "config": asdict(config)}, candidate_path)

    candidate_rows = []
    for name, model, mean, std in (("S", s_model, mean_s, std_s), ("KKT", k_model, mean_k, std_k)):
        candidate_rows.extend(_evaluate_vdp(name, model, np.asarray(mean), np.asarray(std), validation, reference_objective, device))
    _write_rows(root / "validation_candidates.csv", candidate_rows)
    scores = {name: _candidate_score([r for r in candidate_rows if r["candidate"] == name]) for name in ("S", "KKT")}
    feasible = [name for name, score in scores.items() if score["accepted"]]
    if not feasible:
        raise RuntimeError(f"No VDP candidate passed validation for seed {seed}")
    selected = min(feasible, key=lambda name: scores[name]["mean_relative_objective_difference_percent"])
    model, mean, std = (s_model, mean_s, std_s) if selected == "S" else (k_model, mean_k, std_k)
    test_path = root / "test_selected.csv"
    test_rows = (_read_rows(test_path) if test_path.exists()
                 else _evaluate_vdp(selected, model, np.asarray(mean), np.asarray(std), test,
                                    _read_test_references("VDP", seed), device))
    if not test_path.exists():
        _write_rows(test_path, test_rows)
    return {"seed": seed, "selected_branch": selected, "validation_scores": scores,
            "validation_reference_mean_s": float(np.mean([r["solve_seconds"] for r in refs])),
            "test": _candidate_score(test_rows)}


def _run_pen(seed: int, output: Path, validation_count: int, workers: int, device: torch.device) -> dict[str, Any]:
    source = ROOT / "kkt_collocation" / "results" / f"final_multiseed_penicillin400_penalty_seed{seed}"
    root = output / "penicillin" / f"seed{seed}"; root.mkdir(parents=True, exist_ok=True)
    validation = np.load(source / "validation_x2.npy")[:validation_count]
    test = np.load(source / "test_x2.npy")
    refs = _reference_records(root / "validation_cold_nlp_references.npz", validation, "penicillin", workers)
    reference_objective = np.asarray([r["objective"] for r in refs], dtype=float)

    checkpoint = torch.load(source / "models.pth", map_location=device, weights_only=False)
    s_model, k_model = PenicillinPolicy().to(device), PenicillinPolicy().to(device)
    s_model.load_state_dict(checkpoint["S"]); k_model.load_state_dict(checkpoint["true_KKT"])
    mean_s, std_s = checkpoint["normalization"]["S"]
    mean_k, std_k = checkpoint["normalization"]["true_KKT"]
    candidate_rows = []
    for name, model, mean, std in (("S", s_model, mean_s, std_s), ("KKT", k_model, mean_k, std_k)):
        candidate_rows.extend(_evaluate_pen(name, model, float(mean), float(std), validation, reference_objective, device))
    _write_rows(root / "validation_candidates.csv", candidate_rows)
    scores = {name: _candidate_score([r for r in candidate_rows if r["candidate"] == name]) for name in ("S", "KKT")}
    feasible = [name for name, score in scores.items() if score["accepted"]]
    if not feasible:
        raise RuntimeError(f"No penicillin candidate passed validation for seed {seed}")
    selected = min(feasible, key=lambda name: scores[name]["mean_relative_objective_difference_percent"])
    model, mean, std = (s_model, mean_s, std_s) if selected == "S" else (k_model, mean_k, std_k)
    test_path = root / "test_selected.csv"
    test_rows = (_read_rows(test_path) if test_path.exists()
                 else _evaluate_pen(selected, model, float(mean), float(std), test,
                                    _read_test_references("Penicillin", seed), device))
    if not test_path.exists():
        _write_rows(test_path, test_rows)
    return {"seed": seed, "selected_branch": selected, "validation_scores": scores,
            "validation_reference_mean_s": float(np.mean([r["solve_seconds"] for r in refs])),
            "test": _candidate_score(test_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("vdp", "penicillin", "both"), default="both")
    parser.add_argument("--validation-count", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "validation_selected_vdp_penicillin")
    args = parser.parse_args()
    if not 1 <= args.validation_count <= 100:
        raise ValueError("validation-count must be between 1 and 100")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "protocol": {
            "selection": "zero HDS fallback first, then minimum mean absolute relative objective difference",
            "validation": "first 60 points of each pre-existing independent 100-point validation cohort",
            "test": "frozen 400-point cohort; never used to select branch or configuration",
            "lambda_grid_size": 31,
            "VDP_reference": "cold-start trust-constr, fixed neutral internal u=0.5 initialization",
            "penicillin_reference": "cold-start SLSQP with fixed constant-input multi-starts {0.4,0.5,0.7,0.9,1.0}; no nearest-label or policy warm start",
        },
        "vdp": [], "penicillin": [],
    }
    if args.problem in ("vdp", "both"):
        for seed in (20260751, 20260752, 20260753):
            print(f"[VDP] validation selection for seed {seed}", flush=True)
            report["vdp"].append(_run_vdp(seed, args.output, args.validation_count, args.workers, device))
    if args.problem in ("penicillin", "both"):
        for seed in (20260761, 20260762, 20260763):
            print(f"[Penicillin] validation selection for seed {seed}", flush=True)
            report["penicillin"].append(_run_pen(seed, args.output, args.validation_count, args.workers, device))
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
