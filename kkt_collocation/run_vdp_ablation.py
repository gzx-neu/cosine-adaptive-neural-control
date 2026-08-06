"""Run the four VDP ablations on one pre-registered in-domain test set.

Methods
-------
S       supervised policy;
S+KKT   identical policy with the reduced-space augmented-Lagrangian KKT loss;
S+HDS  supervised policy followed by the HDS--lambda corrector;
Full    KKT policy followed by the same HDS--lambda corrector.

All reference labels train both policies.  Validation and test initial states
are independently generated continuous points in the prescribed operating
domain.  Optional test-reference labels allow true objective-gap reporting on
a solver-labelled subset; otherwise the script honestly reports only the
nominal-to-corrected objective change.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
# SciPy/CasADi/Torch wheels on this Windows workstation may load distinct
# Intel OpenMP runtimes.  This only permits the documented local setup; it is
# not a numerical setting of the experiment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from offline_safe_control.kkt_regularization import augmented_lagrangian_kkt_residual
from kkt_collocation.train_vdp_kkt_policy import KKTPolicyValueNetwork, TrainConfig, differentiable_rollout


@dataclass(frozen=True)
class AblationConfig:
    horizon: float = 5.0
    zoh_steps: int = 10
    epochs: int = 500
    learning_rate: float = 1e-3
    kkt_weight: float = 1e-3
    rollout_weight: float = 0.5
    augmented_penalty: float = 10.0
    seed: int = 20260714
    validation_samples: int = 100
    test_samples: int = 400
    allowed_peak_violation: float = 1e-4
    allowed_violation_rate: float = 0.05
    lambda_grid_size: int = 101

    @property
    def duration(self) -> float:
        return self.horizon / self.zoh_steps


def vdp_ode(_time: float, state: np.ndarray, control: float) -> np.ndarray:
    y1, y2, _ = state
    return np.array([(1.0 - y2**2) * y1 - y2 + control, y1, y1**2 + y2**2 + control**2])


def constraint(state: np.ndarray) -> float:
    return float(-0.4 - state[0])


def constraint_derivative(state: np.ndarray, control: float) -> float:
    return float(-vdp_ode(0.0, state, control)[0])


def load_data(path: Path, device: torch.device):
    with path.open("rb") as handle:
        data = pickle.load(handle)
    initial = np.asarray(data["initial_state"], dtype=np.float32)
    controls = np.asarray(data["optimal_controls"], dtype=np.float32)
    objective = np.asarray(data["objective"], dtype=np.float32).reshape(-1, 1)
    duals = np.asarray(data["path_duals"], dtype=np.float32)
    if initial.shape[1:] != (3,) or controls.shape[1:] != (10,) or duals.shape[1:] != (101,):
        raise ValueError("data must be generated with 10 ZOH steps and 10 RK4 substeps per ZOH")
    return (torch.tensor(initial, device=device), torch.tensor(controls, device=device),
            torch.tensor(objective, device=device), torch.tensor(duals, device=device), data)


def lhs_states(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unit = np.empty((count, 2), dtype=np.float64)
    for dimension in range(2):
        unit[:, dimension] = (rng.permutation(count) + rng.random(count)) / count
    lower, upper = np.array([-0.1, 0.9]), np.array([0.1, 1.1])
    return np.column_stack((lower + unit * (upper - lower), np.zeros(count)))


def train_policy(initial, controls_ref, objective_ref, duals_ref, use_kkt: bool, config: AblationConfig, device,
                 model=None, epochs: int | None = None, learning_rate: float | None = None,
                 path_penalty_weight: float = 0.0, constraint_scale: float = 1.0):
    train_config = TrainConfig(epochs=config.epochs)
    if model is None:
        torch.manual_seed(config.seed)
        model = KKTPolicyValueNetwork(train_config).to(device)
    else:
        model = model.to(device)
    input_mean = initial[:, :2].mean(dim=0)
    input_std = initial[:, :2].std(dim=0).clamp_min(1e-8)
    objective_mean = objective_ref.mean()
    objective_std = objective_ref.std().clamp_min(1e-8)
    x = (initial[:, :2] - input_mean) / input_std
    target_j = (objective_ref - objective_mean) / objective_std
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate if learning_rate is None else learning_rate)
    total_epochs = config.epochs if epochs is None else epochs
    for epoch in range(1, total_epochs + 1):
        predicted_j, predicted_u = model(x)
        rollout_j, path_g = differentiable_rollout(initial, predicted_u, train_config)
        supervision = nn.functional.mse_loss(predicted_u, controls_ref) + 0.1 * nn.functional.mse_loss(predicted_j, target_j)
        consistency = nn.functional.mse_loss(predicted_j * objective_std + objective_mean, rollout_j.unsqueeze(1))
        loss = supervision + config.rollout_weight * consistency
        if path_penalty_weight > 0.0:
            normalized_violation = torch.relu(path_g / constraint_scale)
            loss = loss + path_penalty_weight * normalized_violation.square().mean()
        if use_kkt:
            residual = augmented_lagrangian_kkt_residual(rollout_j, predicted_u, path_g, duals_ref, config.augmented_penalty)
            loss = loss + config.kkt_weight * residual.total
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch in (1, total_epochs) or epoch % 100 == 0:
            name = "S+KKT" if use_kkt else ("S+P" if path_penalty_weight > 0 else "S")
            print(f"{name:5s} epoch={epoch:4d} loss={loss.item():.3e}")
    return model.eval(), input_mean.cpu().numpy(), input_std.cpu().numpy()


def predict(model, mean, std, states, device):
    x = torch.tensor((states[:, :2] - mean) / std, dtype=torch.float32, device=device)
    start = time.perf_counter()
    with torch.no_grad():
        _, controls = model(x)
    elapsed = (time.perf_counter() - start) / len(states)
    return controls.cpu().numpy(), elapsed


def terminal_cost(initial: np.ndarray, controls: np.ndarray, corrector: HDSLambdaCorrector, duration: float) -> float:
    state = np.asarray(initial, dtype=float)
    for control in controls:
        _, state = corrector.segment_peak(state, float(control), duration)
    return float(state[2])


def validation_audit(controls, states, corrector, config):
    violations = np.asarray([max(0.0, corrector.audit(x, u, config.duration)) for x, u in zip(states, controls)])
    peak, rate = float(violations.max()), float(np.mean(violations > config.allowed_peak_violation))
    return {"peak_violation": peak, "mean_violation": float(violations.mean()), "violation_rate": rate,
            "kkt_extra_training_required": bool(peak > config.allowed_peak_violation or rate > config.allowed_violation_rate)}


def evaluate(method: str, nominal_controls, states, correct: bool, corrector, inference_seconds, config, reference=None):
    rows = []
    for index, (initial, nominal) in enumerate(zip(states, nominal_controls)):
        nominal_peak = corrector.audit(initial, nominal, config.duration)
        nominal_cost = terminal_cost(initial, nominal, corrector, config.duration)
        start = time.perf_counter()
        outcome = corrector.correct(initial, nominal, config.duration) if correct else None
        filter_seconds = time.perf_counter() - start if correct else 0.0
        accepted = outcome.accepted if outcome is not None else True
        applied = outcome.controls if outcome is not None and accepted else nominal
        applied_peak = corrector.audit(initial, applied, config.duration) if accepted else np.nan
        applied_cost = terminal_cost(initial, applied, corrector, config.duration) if accepted else np.nan
        changes = np.asarray([s.corrected for s in outcome.segments], dtype=float) if outcome is not None else np.zeros(10)
        lambdas = np.asarray([s.lambda_value for s in outcome.segments if s.lambda_value is not None], dtype=float) if outcome is not None else np.ones(10)
        reference_cost = float(reference[index]) if reference is not None and index < len(reference) else np.nan
        rows.append({"method": method, "sample_index": index, "y1_0": initial[0], "y2_0": initial[1],
                     "nominal_hds_max_g": nominal_peak, "applied_hds_max_g": applied_peak,
                     "accepted": accepted, "fallback": not accepted, "nominal_cost": nominal_cost,
                     "applied_cost": applied_cost, "reference_cost": reference_cost,
                     "nominal_objective_gap": nominal_cost-reference_cost if np.isfinite(reference_cost) else np.nan,
                     "applied_objective_gap": applied_cost-reference_cost if np.isfinite(reference_cost) and accepted else np.nan,
                     "objective_change": applied_cost-nominal_cost if accepted else np.nan,
                     "corrected_segments": int(changes.sum()), "mean_abs_lambda_minus_one": float(np.mean(np.abs(lambdas-1.0))),
                     "inference_seconds": inference_seconds, "filter_seconds": filter_seconds})
    return rows


def summarise(rows):
    by_method = {}
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        finite = lambda key: np.asarray([r[key] for r in group], dtype=float)
        accepted = np.asarray([r["accepted"] for r in group], dtype=bool)
        applied_peaks = finite("applied_hds_max_g")
        nominal_peaks = finite("nominal_hds_max_g")
        by_method[method] = {"samples": len(group), "accepted_rate": float(accepted.mean()), "fallback_rate": float(1-accepted.mean()),
                             "nominal_violation_rate": float(np.mean(nominal_peaks > 1e-8)),
                             "nominal_severe_violation_rate": float(np.mean(nominal_peaks > 0.025*0.4)),
                             "mean_positive_nominal_violation": float(np.maximum(nominal_peaks, 0.).mean()),
                             "max_nominal_hds_g": float(nominal_peaks.max()),
                             "accepted_max_hds_g": float(np.nanmax(applied_peaks)), "mean_corrected_segments": float(finite("corrected_segments").mean()),
                             "mean_abs_lambda_minus_one": float(finite("mean_abs_lambda_minus_one").mean()),
                             "mean_nominal_gap": float(np.nanmean(finite("nominal_objective_gap"))),
                             "mean_applied_gap": float(np.nanmean(finite("applied_objective_gap"))),
                             "mean_objective_change": float(np.nanmean(finite("objective_change"))),
                             "mean_inference_seconds": float(finite("inference_seconds").mean()), "mean_filter_seconds": float(finite("filter_seconds").mean())}
    return by_method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--test-reference-data", type=Path)
    parser.add_argument("--test-states-npy", type=Path,
                        help="Pre-generated (n,3) in-domain test states; enables solver labels to be made before evaluation.")
    parser.add_argument("--models-file", type=Path,
                        help="Previously saved models.pth.  When supplied, skip training and evaluate these exact policies.")
    parser.add_argument("--train-only", action="store_true",
                        help="Train both policies, save models.pth, then exit before any HDS evaluation.")
    parser.add_argument("--output", type=Path, default=ROOT / "kkt_collocation" / "results" / "vdp_ablation")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lambda-grid-size", type=int, default=101)
    parser.add_argument("--test-samples", type=int, default=400)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = AblationConfig(epochs=10 if args.smoke else args.epochs, test_samples=12 if args.smoke else args.test_samples, validation_samples=8 if args.smoke else args.validation_samples, lambda_grid_size=21 if args.smoke else args.lambda_grid_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial, controls, objectives, duals, _ = load_data(args.data, device)
    args.output.mkdir(parents=True, exist_ok=True)
    validation_states = lhs_states(config.validation_samples, config.seed+1)
    test_states = lhs_states(config.test_samples, config.seed+2) if args.test_states_npy is None else np.load(args.test_states_npy)
    if test_states.ndim != 2 or test_states.shape[1] != 3:
        raise ValueError("--test-states-npy must contain an (n,3) array")
    if len(test_states) != config.test_samples:
        raise ValueError("--test-samples must equal the number of supplied test states")
    np.save(args.output / "validation_states.npy", validation_states)
    np.save(args.output / "test_states.npy", test_states)
    corrector = HDSLambdaCorrector(vdp_ode, constraint, constraint_derivative, (-0.3, 1.0), HDSLambdaConfig(grid_size=config.lambda_grid_size, max_step_fraction=100.0))
    if args.models_file is None:
        supervised, mean_s, std_s = train_policy(initial, controls, objectives, duals, False, config, device)
        kkt, mean_k, std_k = train_policy(initial, controls, objectives, duals, True, config, device)
        torch.save({"supervised": supervised.state_dict(), "kkt": kkt.state_dict(), "mean_s": mean_s, "std_s": std_s, "mean_k": mean_k, "std_k": std_k, "config": asdict(config)}, args.output / "models.pth")
        training_config = asdict(config)
    else:
        checkpoint = torch.load(args.models_file, map_location=device, weights_only=False)
        supervised = KKTPolicyValueNetwork(TrainConfig()).to(device)
        kkt = KKTPolicyValueNetwork(TrainConfig()).to(device)
        supervised.load_state_dict(checkpoint["supervised"])
        kkt.load_state_dict(checkpoint["kkt"])
        supervised.eval(); kkt.eval()
        mean_s, std_s = np.asarray(checkpoint["mean_s"]), np.asarray(checkpoint["std_s"])
        mean_k, std_k = np.asarray(checkpoint["mean_k"]), np.asarray(checkpoint["std_k"])
        training_config = checkpoint.get("config", asdict(config))
    if args.train_only:
        print(f"Saved trained policies to {args.output / 'models.pth'}")
        return
    u_s_val, _ = predict(supervised, mean_s, std_s, validation_states, device)
    u_k_val, _ = predict(kkt, mean_k, std_k, validation_states, device)
    u_s, t_s = predict(supervised, mean_s, std_s, test_states, device)
    u_k, t_k = predict(kkt, mean_k, std_k, test_states, device)
    reference = None
    if args.test_reference_data:
        with args.test_reference_data.open("rb") as handle:
            ref = pickle.load(handle)
        ref_states = np.asarray(ref["initial_state"])
        if len(ref_states) > len(test_states) or not np.allclose(ref_states, test_states[:len(ref_states)], atol=1e-10):
            raise ValueError("test-reference states must equal the first test states saved by this run")
        reference = np.asarray(ref["objective"], dtype=float)
    rows = []
    rows += evaluate("S", u_s, test_states, False, corrector, t_s, config, reference)
    rows += evaluate("S+KKT", u_k, test_states, False, corrector, t_k, config, reference)
    rows += evaluate("S+HDS-lambda", u_s, test_states, True, corrector, t_s, config, reference)
    rows += evaluate("Full: S+KKT+HDS-lambda", u_k, test_states, True, corrector, t_k, config, reference)
    with (args.output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {"evaluation_config": asdict(config), "training_config": training_config, "data": str(args.data), "validation": {"S": validation_audit(u_s_val, validation_states, corrector, config), "S+KKT": validation_audit(u_k_val, validation_states, corrector, config)}, "methods": summarise(rows), "objective_reference_note": "Objective gaps are reported only when --test-reference-data is supplied."}
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
