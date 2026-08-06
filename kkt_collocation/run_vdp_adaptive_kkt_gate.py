"""VDP Never/Always/Adaptive-KKT comparison under the fixed validation gate.

This script uses the same 400 direct-transcription labels and path duals for
all branches.  A supervised policy is trained first.  The Always-KKT policy is
then obtained by KKT fine-tuning that exact supervised policy.  Adaptive uses
only the raw supervised-policy HDS audit on an independent validation set to
select one of those two policies.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.adaptive_kkt_gate import AdaptiveKKTThresholds, audit_raw_hds_peaks  # noqa: E402
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector  # noqa: E402
from run_vdp_ablation import (  # noqa: E402
    AblationConfig,
    constraint,
    constraint_derivative,
    evaluate,
    lhs_states,
    load_data,
    predict,
    summarise,
    terminal_cost,
    train_policy,
    vdp_ode,
)


_EVALUATION_CORRECTOR: HDSLambdaCorrector | None = None


def _evaluation_worker_init(grid_size: int) -> None:
    """Build one corrector in each process; HDS integration dominates this evaluation."""
    global _EVALUATION_CORRECTOR
    _EVALUATION_CORRECTOR = HDSLambdaCorrector(
        vdp_ode, constraint, constraint_derivative, (-0.3, 1.0),
        HDSLambdaConfig(grid_size=grid_size, max_step_fraction=100.0),
    )


def _evaluate_one(task: tuple[str, int, np.ndarray, np.ndarray, bool, float, float]) -> dict:
    if _EVALUATION_CORRECTOR is None:
        raise RuntimeError("HDS evaluator worker was not initialized")
    name, index, initial, nominal, correct, inference_seconds, duration = task
    nominal_peak = _EVALUATION_CORRECTOR.audit(initial, nominal, duration)
    nominal_cost = terminal_cost(initial, nominal, _EVALUATION_CORRECTOR, duration)
    start = time.perf_counter()
    outcome = _EVALUATION_CORRECTOR.correct(initial, nominal, duration) if correct else None
    filter_seconds = time.perf_counter() - start if correct else 0.0
    accepted = outcome.accepted if outcome is not None else True
    applied = outcome.controls if outcome is not None and accepted else nominal
    applied_peak = _EVALUATION_CORRECTOR.audit(initial, applied, duration) if accepted else np.nan
    applied_cost = terminal_cost(initial, applied, _EVALUATION_CORRECTOR, duration) if accepted else np.nan
    changes = np.asarray([segment.corrected for segment in outcome.segments], dtype=float) if outcome is not None else np.zeros(10)
    lambdas = np.asarray([segment.lambda_value for segment in outcome.segments if segment.lambda_value is not None], dtype=float) if outcome is not None else np.ones(10)
    return {
        "method": name, "sample_index": index, "y1_0": initial[0], "y2_0": initial[1],
        "nominal_hds_max_g": nominal_peak, "applied_hds_max_g": applied_peak,
        "accepted": accepted, "fallback": not accepted, "nominal_cost": nominal_cost,
        "applied_cost": applied_cost, "reference_cost": np.nan,
        "nominal_objective_gap": np.nan, "applied_objective_gap": np.nan,
        "objective_change": applied_cost - nominal_cost if accepted else np.nan,
        "corrected_segments": int(changes.sum()),
        "mean_abs_lambda_minus_one": float(np.mean(np.abs(lambdas - 1.0))),
        "inference_seconds": inference_seconds, "filter_seconds": filter_seconds,
    }


def evaluate_methods_parallel(
    methods: list[tuple[str, np.ndarray, bool, float]], states: np.ndarray, duration: float,
    grid_size: int, workers: int,
) -> list[dict]:
    """Evaluate all branches on identical states without changing any HDS logic."""
    tasks = [
        (name, index, np.asarray(state, dtype=float), np.asarray(controls[index], dtype=float), correct,
         inference_seconds, duration)
        for name, controls, correct, inference_seconds in methods
        for index, state in enumerate(states)
    ]
    if workers == 1:
        _evaluation_worker_init(grid_size)
        return [_evaluate_one(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers, initializer=_evaluation_worker_init, initargs=(grid_size,)) as pool:
        return list(pool.map(_evaluate_one, tasks, chunksize=4))


def raw_peaks(controls: np.ndarray, states: np.ndarray, corrector: HDSLambdaCorrector, duration: float) -> np.ndarray:
    return np.asarray([corrector.audit(state, control, duration) for state, control in zip(states, controls)], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "kkt_collocation" / "data" / "vdp_kkt_30x30_warm_s10_margin1e3.pkl",
                        help="Final protocol uses the 900-point 30x30 HDS-audited label set.")
    parser.add_argument("--output", type=Path, default=ROOT / "kkt_collocation" / "results" / "vdp_adaptive_true_kkt_gate")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--training-seed", type=int, default=20260714,
                        help="Random seed for network initialization/training; validation and test sets stay fixed.")
    parser.add_argument("--kkt-epochs", type=int, default=20)
    parser.add_argument("--kkt-weight", type=float, default=1e-3)
    parser.add_argument("--constraint-penalty-weight", type=float, default=1.0)
    parser.add_argument("--test-samples", type=int, default=400)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--validation-seed", type=int,
                        help="Independent validation LHS seed. Defaults to the historical config seed + 1.")
    parser.add_argument("--test-seed", type=int,
                        help="Independent final-test LHS seed. Defaults to the historical config seed + 2.")
    parser.add_argument("--lambda-grid-size", type=int, default=31)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel HDS-evaluation workers; use 1 for debugging.")
    parser.add_argument("--gate-rate", type=float, default=0.05)
    parser.add_argument("--gate-rate-normalized-violation", type=float, default=0.025)
    parser.add_argument("--gate-normalized-peak", type=float, default=0.03)
    parser.add_argument("--gate-engineering-scale", type=float, default=0.4)
    parser.add_argument("--gate-numerical-tolerance", type=float, default=1e-8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = AblationConfig(
        epochs=10 if args.smoke else args.epochs,
        test_samples=12 if args.smoke else args.test_samples,
        validation_samples=8 if args.smoke else args.validation_samples,
        lambda_grid_size=11 if args.smoke else args.lambda_grid_size,
        kkt_weight=args.kkt_weight,
        seed=args.training_seed,
    )
    kkt_epochs = 3 if args.smoke else args.kkt_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial, controls, objectives, duals, _ = load_data(args.data, device)
    args.output.mkdir(parents=True, exist_ok=True)
    validation_seed = config.seed + 1 if args.validation_seed is None else args.validation_seed
    test_seed = config.seed + 2 if args.test_seed is None else args.test_seed
    validation_states = lhs_states(config.validation_samples, validation_seed)
    test_states = lhs_states(config.test_samples, test_seed)
    np.save(args.output / "validation_states.npy", validation_states)
    np.save(args.output / "test_states.npy", test_states)

    supervised, mean_s, std_s = train_policy(initial, controls, objectives, duals, False, config, device)
    kkt, mean_k, std_k = train_policy(
        initial, controls, objectives, duals, True, config, device,
        model=copy.deepcopy(supervised), epochs=kkt_epochs, learning_rate=1e-5,
    )
    penalty, mean_p, std_p = train_policy(
        initial, controls, objectives, duals, False, config, device,
        model=copy.deepcopy(supervised), epochs=kkt_epochs, learning_rate=1e-5,
        path_penalty_weight=args.constraint_penalty_weight,
        constraint_scale=args.gate_engineering_scale,
    )
    corrector = HDSLambdaCorrector(
        vdp_ode, constraint, constraint_derivative, (-0.3, 1.0),
        HDSLambdaConfig(grid_size=config.lambda_grid_size, max_step_fraction=100.0),
    )
    u_s_val, _ = predict(supervised, mean_s, std_s, validation_states, device)
    thresholds = AdaptiveKKTThresholds(
        numerical_violation_tolerance=args.gate_numerical_tolerance,
        allowed_violation_rate=args.gate_rate,
        rate_normalized_violation=args.gate_rate_normalized_violation,
        allowed_normalized_peak_violation=args.gate_normalized_peak,
        engineering_constraint_scale=args.gate_engineering_scale,
    )
    gate = audit_raw_hds_peaks(raw_peaks(u_s_val, validation_states, corrector, config.duration), thresholds)
    u_s, time_s = predict(supervised, mean_s, std_s, test_states, device)
    u_k, time_k = predict(kkt, mean_k, std_k, test_states, device)
    u_p, time_p = predict(penalty, mean_p, std_p, test_states, device)
    adaptive_controls, adaptive_time = (u_k, time_k) if gate.kkt_refinement_required else (u_s, time_s)
    selected = "S+KKT" if gate.kkt_refinement_required else "S"
    rows = evaluate_methods_parallel(
        [("Never-KKT: S", u_s, False, time_s),
         ("Constraint-penalty: S+P", u_p, False, time_p),
         ("Always-KKT: S+KKT", u_k, False, time_k),
         ("Never-KKT + HDS-lambda", u_s, True, time_s),
         ("Constraint-penalty: S+P + HDS-lambda", u_p, True, time_p),
         ("Always-KKT + HDS-lambda", u_k, True, time_k)],
        test_states, config.duration, config.lambda_grid_size, args.workers,
    )
    # Adaptive is exactly one of the two precomputed branches.  Re-label its
    # rows rather than repeating identical ODE/HDS integrations.
    adaptive_raw_source = "Always-KKT: S+KKT" if gate.kkt_refinement_required else "Never-KKT: S"
    adaptive_safe_source = "Always-KKT + HDS-lambda" if gate.kkt_refinement_required else "Never-KKT + HDS-lambda"
    for source, target in ((adaptive_raw_source, f"Adaptive ({selected})"),
                           (adaptive_safe_source, f"Adaptive ({selected}) + HDS-lambda")):
        rows.extend([{**row, "method": target} for row in rows if row["method"] == source])
    with (args.output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {
        "config": asdict(config),
        "kkt_finetune_epochs": kkt_epochs,
        "constraint_penalty": {
            "weight": args.constraint_penalty_weight,
            "normalization_scale": args.gate_engineering_scale,
            "finetune_epochs": kkt_epochs,
            "learning_rate": 1e-5,
        },
        "label_source": str(args.data),
        "data_split": {
            "validation_design": "independent Latin-hypercube sample in the declared initial-state domain",
            "validation_seed": validation_seed,
            "test_design": "independent Latin-hypercube sample in the declared initial-state domain",
            "test_seed": test_seed,
        },
        "adaptive_gate": {"thresholds": asdict(thresholds), "supervised_raw_validation": asdict(gate), "selected_branch": selected},
        "methods": summarise(rows),
        "note": "The gate is based only on raw supervised-policy HDS peaks. HDS-lambda is an offline numerical correction/audit, not an online controller.",
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    torch.save({"S": supervised.state_dict(), "Penalty": penalty.state_dict(), "KKT": kkt.state_dict(),
                "normalization": {"S": [mean_s, std_s], "Penalty": [mean_p, std_p], "KKT": [mean_k, std_k]},
                "config": asdict(config)}, args.output / "models.pth")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
