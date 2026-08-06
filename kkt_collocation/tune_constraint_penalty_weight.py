"""Pre-register S+Penalty weights on fixed validation sets only.

Each candidate starts from the exact same supervised checkpoint and receives
the same fine-tuning budget as Always-KKT.  Selection first minimizes severe
continuous-time HDS risk among candidates whose mean nominal-objective
deterioration from S is at most one percent.  Final test states are never read.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from kkt_collocation.run_vdp_ablation import (
    AblationConfig, constraint as vdp_g, constraint_derivative as vdp_gdot,
    lhs_states, load_data as load_vdp, predict as predict_vdp,
    terminal_cost, train_policy, vdp_ode,
)
from kkt_collocation.run_penicillin_ablation import (
    Config as PenConfig, DT, UMAX, g as pen_g, gdot as pen_gdot, lhs as pen_lhs,
    ode as pen_ode, predict as predict_pen, terminal_product, train as train_pen,
)
from kkt_collocation.run_penicillin_true_kkt_ablation import load_true_kkt_data


def _risk(peaks: np.ndarray, scale: float) -> dict:
    positive = np.maximum(peaks, 0.0)
    severe_threshold = 0.025 * scale
    return {
        "raw_violation_rate": float(np.mean(peaks > 1e-8)),
        "severe_violation_rate": float(np.mean(positive > severe_threshold)),
        "maximum_violation": float(positive.max()),
        "mean_violation": float(positive.mean()),
        "normalized_peak": float(positive.max() / scale),
    }


def _select(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row["mean_relative_objective_deterioration"] <= 0.01]
    pool = eligible or rows
    selected = min(pool, key=lambda row: (
        row["risk"]["severe_violation_rate"], row["risk"]["normalized_peak"],
        row["risk"]["mean_violation"], row["mean_abs_control_change"],
    ))
    return {"selected_weight": selected["weight"], "one_percent_objective_guard_satisfied": bool(eligible),
            "selection_rule": "lexicographic severe-rate, normalized peak, mean violation, control change; candidates first restricted to <=1% mean nominal-objective deterioration from S"}


def tune_vdp(args, device) -> dict:
    cfg = AblationConfig(epochs=args.epochs, seed=args.training_seed, validation_samples=args.validation_samples,
                         lambda_grid_size=31, kkt_weight=1e-3)
    initial, controls, objectives, duals, _ = load_vdp(args.vdp_data, device)
    states = lhs_states(cfg.validation_samples, args.vdp_validation_seed)
    supervised, mean_s, std_s = train_policy(initial, controls, objectives, duals, False, cfg, device)
    u_s, _ = predict_vdp(supervised, mean_s, std_s, states, device)
    corrector = HDSLambdaCorrector(vdp_ode, vdp_g, vdp_gdot, (-.3, 1.), HDSLambdaConfig(grid_size=31, max_step_fraction=100.))
    base_obj = np.asarray([terminal_cost(x, u, corrector, cfg.duration) for x, u in zip(states, u_s)])
    rows = []
    for weight in args.weights:
        model, mean_p, std_p = train_policy(
            initial, controls, objectives, duals, False, cfg, device,
            model=copy.deepcopy(supervised), epochs=args.finetune_epochs, learning_rate=1e-5,
            path_penalty_weight=weight, constraint_scale=.4,
        )
        u_p, _ = predict_vdp(model, mean_p, std_p, states, device)
        peaks = np.asarray([corrector.audit(x, u, cfg.duration) for x, u in zip(states, u_p)])
        obj = np.asarray([terminal_cost(x, u, corrector, cfg.duration) for x, u in zip(states, u_p)])
        deterioration = np.mean((obj-base_obj)/np.maximum(np.abs(base_obj), 1e-8))
        rows.append({"weight": weight, "risk": _risk(peaks, .4),
                     "mean_relative_objective_deterioration": float(deterioration),
                     "mean_abs_control_change": float(np.mean(np.abs(u_p-u_s)))})
    return {"problem": "vdp", "training_seed": cfg.seed, "validation_seed": args.vdp_validation_seed,
            "validation_samples": len(states), "candidates": rows, **_select(rows)}


def tune_pen(args, device) -> dict:
    cfg = PenConfig(epochs=args.epochs, kkt_epochs=args.finetune_epochs, seed=args.training_seed,
                    validation_samples=args.validation_samples, substeps=80, rollout_weight=0.)
    x2, controls, objective, duals, _ = load_true_kkt_data(args.pen_data)
    validation = pen_lhs(cfg.validation_samples, args.pen_validation_seed)
    supervised, mean_s, std_s = train_pen(x2, controls, objective, duals, False, cfg, device)
    normalized = torch.tensor(((x2-mean_s)/std_s)[:, None], dtype=torch.float32, device=device)
    with torch.no_grad(): _, anchor = supervised(normalized)
    u_s, _ = predict_pen(supervised, mean_s, std_s, validation, device)
    corrector = HDSLambdaCorrector(pen_ode, pen_g, pen_gdot, (0., UMAX), HDSLambdaConfig(grid_size=31, max_step_fraction=100.))
    base_obj = np.asarray([terminal_product(value, u, corrector) for value, u in zip(validation, u_s)])
    rows = []
    for weight in args.weights:
        model, mean_p, std_p = train_pen(
            x2, controls, objective, duals, False, cfg, device,
            model=copy.deepcopy(supervised), epochs=args.finetune_epochs,
            learning_rate=cfg.kkt_learning_rate, anchor_controls=anchor.cpu().numpy(),
            path_penalty_weight=weight, constraint_scale=.5,
        )
        u_p, _ = predict_pen(model, mean_p, std_p, validation, device)
        peaks = np.asarray([corrector.audit(np.array([1., value, .001, 250.]), u, DT)
                            for value, u in zip(validation, u_p)])
        obj = np.asarray([terminal_product(value, u, corrector) for value, u in zip(validation, u_p)])
        deterioration = np.mean((base_obj-obj)/np.maximum(np.abs(base_obj), 1e-8))
        rows.append({"weight": weight, "risk": _risk(peaks, .5),
                     "mean_relative_objective_deterioration": float(deterioration),
                     "mean_abs_control_change": float(np.mean(np.abs(u_p-u_s)))})
    return {"problem": "penicillin", "training_seed": cfg.seed, "validation_seed": args.pen_validation_seed,
            "validation_samples": len(validation), "candidates": rows, **_select(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("vdp", "penicillin"), required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--training-seed", type=int, default=20260760)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--vdp-validation-seed", type=int, default=20260730)
    parser.add_argument("--pen-validation-seed", type=int, default=20260740)
    parser.add_argument("--vdp-data", type=Path, default=ROOT/"kkt_collocation"/"data"/"vdp_kkt_30x30_warm_s10_margin1e3.pkl")
    parser.add_argument("--pen-data", type=Path, default=ROOT/"kkt_collocation"/"data"/"penicillin_kkt_400_true_duals.pkl")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = tune_vdp(args, device) if args.problem == "vdp" else tune_pen(args, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
