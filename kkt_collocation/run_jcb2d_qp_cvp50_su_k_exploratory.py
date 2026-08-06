"""Exploratory JCB CVP50 comparison: control-only supervision vs control-only+KKT.

This preserves the latest JCB exact-flow CVP50 QP labels, 100 frozen
cold-start QP references, 31-candidate HDS correction, and full path-plus-box
finite-dimensional KKT residual.  It intentionally removes objective-value
supervision from both methods:

* S-u: 210 epochs of normalized control MSE.
* S-u+K: 200 epochs of normalized control MSE, then 10 continuation epochs
  of control MSE + 1e-2 normalized KKT residual + 1.0 normalized anchor MSE.

The output is exploratory and must not be combined with the three-benchmark
formal protocol.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from generate_jcb_reduced_kkt_data import JCBConfig, ReducedJCBQP
from run_jcb2d_jiang_valc import g, gdot, initial, lhs, ode
from run_jcb2d_qp_cvp50_two_stage_vs_s import (
    Experiment,
    Policy,
    cold_references,
    dump,
    evaluate,
    full_kkt,
    load_labels,
    seed_all,
)
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "kkt_collocation/results/jcb2d_qp_cvp50_nodes10_30x30/records.jsonl"
DEFAULT_OUTPUT = ROOT / "kkt_collocation/results/exploratory_jcb2d_qp_cvp50_su_k10_unified_v1"


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--split-seed", type=int, default=20260771)
    parser.add_argument("--reference-file", type=Path, default=None,
                        help="Reuse an existing frozen cold-QP+HDS reference JSON; never regenerate it.")
    parser.add_argument("--project-conflicting-kkt-gradient", action="store_true",
                        help="Exploratory S-u+K only: remove KKT gradient components opposing control-MSE plus anchor.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    # This exploratory driver deliberately does not inherit the old JCB
    # two-stage defaults (200+20, alpha=1e-3, anchor=0.1).  It follows the
    # common short-continuation comparison used for the S-u+K question.
    exp = Experiment(
        seed=args.seed,
        split_seed=args.split_seed,
        supervised_epochs=200,
        continuation_epochs=10,
        kkt_weight=1e-2,
        anchor_weight=1.0,
    )
    metadata = json.loads((args.labels.parent / "summary.json").read_text(encoding="utf-8"))
    raw = dict(metadata["config"])
    for field in ("control_bounds", "x1_initial_range", "x2_initial_range"):
        raw[field] = tuple(raw[field])
    cfg = JCBConfig(**raw)
    p, u_ref, _j_ref, mu, bound, recorded = load_labels(args.labels)

    validation = lhs(exp.validation_count, type("Bounds", (), {"x1_bounds": cfg.x1_initial_range, "x2_bounds": cfg.x2_initial_range})(), exp.split_seed + 1)
    test = lhs(exp.test_count, type("Bounds", (), {"x1_bounds": cfg.x1_initial_range, "x2_bounds": cfg.x2_initial_range})(), exp.split_seed + 2)
    np.save(args.output / "validation_initial_conditions.npy", validation)
    np.save(args.output / "test_initial_conditions.npy", test)

    pu = torch.tensor(p, dtype=torch.float64)
    uu = torch.tensor(u_ref, dtype=torch.float64, requires_grad=True)
    teacher_terms = full_kkt(
        pu,
        uu,
        torch.tensor(mu, dtype=torch.float64),
        torch.tensor(bound, dtype=torch.float64),
        cfg,
        exp.augmented_penalty,
        include_bounds=True,
    )
    teacher_check = {
        "recorded_stationarity_norm_mean": float(recorded.mean()),
        "recorded_stationarity_norm_max": float(recorded.max()),
        "torch_total_kkt_residual": float(teacher_terms["total"].detach()),
        "interpretation": f"finite-dimensional exact-flow CVP{cfg.zoh_steps} QP path and box-bound KKT quantities; not continuous-time multipliers",
    }
    dump(args.output / "teacher_kkt_self_check.json", teacher_check)

    corrector = HDSLambdaCorrector(
        ode,
        g,
        gdot,
        cfg.control_bounds,
        HDSLambdaConfig(grid_size=exp.lambda_grid, safety_margin=0.0, max_step_fraction=1.0),
    )
    if args.reference_file is None:
        references = cold_references(test, ReducedJCBQP(cfg), corrector, cfg, args.output)
        reference_source = "generated in this exploratory run"
    else:
        references = json.loads(args.reference_file.read_text(encoding="utf-8"))
        if len(references) != len(test):
            raise ValueError("Frozen reference count does not match the deterministic test cohort")
        for point, reference in zip(test, references):
            if not np.allclose(point, [reference["x1_0"], reference["x2_0"]], rtol=0.0, atol=1e-12):
                raise ValueError("Frozen reference initial conditions do not match the deterministic test cohort")
        dump(args.output / "cold_start_references.json", references)
        reference_source = str(args.reference_file)

    pt = torch.tensor(p, dtype=torch.float32)
    ut = torch.tensor(u_ref, dtype=torch.float32)
    mt = torch.tensor(mu, dtype=torch.float32)
    bt = torch.tensor(bound, dtype=torch.float32)
    mean, std = p.mean(0), p.std(0).clip(1e-6)
    normalized_x = torch.tensor((p - mean) / std, dtype=torch.float32)
    low, high = cfg.control_bounds

    seed_all(exp.seed)
    prototype = Policy(cfg.zoh_steps, low, high)
    initial_state = copy.deepcopy(prototype.state_dict())

    def train(method: str) -> tuple[Policy, dict]:
        model = Policy(cfg.zoh_steps, low, high)
        model.load_state_dict(initial_state)
        history: list[dict] = []
        failure: str | None = None
        started = time.perf_counter()
        if method == "S-u":
            stages = [("supervised", exp.total_epochs, exp.supervised_lr, None)]
        elif method == "S-u+K":
            stages = [("supervised", exp.supervised_epochs, exp.supervised_lr, None)]
        else:
            raise ValueError(method)

        for stage, epochs, learning_rate, anchor in stages:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            for epoch in range(1, epochs + 1):
                try:
                    _, controls = model(normalized_x)
                    control_mse = nn.functional.mse_loss(
                        (controls - low) / (high - low),
                        (ut - low) / (high - low),
                    )
                    loss = control_mse
                    kkt = None
                    anchor_mse = None
                    base_loss = None
                    kkt_loss = None
                    if stage == "continuation":
                        kkt = full_kkt(pt, controls, mt, bt, cfg, exp.augmented_penalty, include_bounds=True)
                        normalized_kkt = kkt["total"] / kkt["total"].detach().clamp_min(torch.finfo(kkt["total"].dtype).tiny)
                        anchor_mse = nn.functional.mse_loss(
                            (controls - low) / (high - low),
                            (anchor - low) / (high - low),
                        )
                        base_loss = control_mse + exp.anchor_weight * anchor_mse
                        kkt_loss = exp.kkt_weight * normalized_kkt
                        loss = base_loss + kkt_loss
                    if not _finite(loss):
                        raise FloatingPointError("non-finite loss")
                    optimizer.zero_grad(set_to_none=True)
                    if args.project_conflicting_kkt_gradient and stage == "continuation":
                        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
                        base_grad = torch.autograd.grad(base_loss, parameters, retain_graph=True, allow_unused=True)
                        kkt_grad = torch.autograd.grad(kkt_loss, parameters, allow_unused=True)
                        dot = sum((left * right).sum() for left, right in zip(base_grad, kkt_grad)
                                  if left is not None and right is not None)
                        base_norm_sq = sum((gradient * gradient).sum() for gradient in base_grad
                                           if gradient is not None).clamp_min(torch.finfo(loss.dtype).tiny)
                        coefficient = torch.minimum(dot / base_norm_sq, torch.zeros_like(dot))
                        for parameter, left, right in zip(parameters, base_grad, kkt_grad):
                            if left is None and right is None:
                                continue
                            value = torch.zeros_like(parameter) if left is None else left
                            if right is not None:
                                value = value + right
                                if left is not None:
                                    value = value - coefficient * left
                            parameter.grad = value
                    else:
                        loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), exp.gradient_clip_norm)
                    if not _finite(gradient_norm):
                        raise FloatingPointError("non-finite gradient")
                    optimizer.step()
                    if not all(_finite(parameter) for parameter in model.parameters()):
                        raise FloatingPointError("non-finite parameter after update")
                    if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                        history.append({
                            "stage": stage,
                            "epoch": epoch,
                            "loss": float(loss.detach()),
                            "control_mse_normalized": float(control_mse.detach()),
                            "kkt_residual": None if kkt is None else float(kkt["total"].detach()),
                            "anchor_mse_normalized": None if anchor_mse is None else float(anchor_mse.detach()),
                        })
                except (RuntimeError, FloatingPointError) as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                    break
            if failure:
                break
            if method == "S-u+K" and stage == "supervised":
                with torch.no_grad():
                    anchor_controls = model(normalized_x)[1].detach()
                stages.append(("continuation", exp.continuation_epochs, exp.continuation_lr, anchor_controls))

        _, final_controls = model(normalized_x)
        final_kkt = full_kkt(pt, final_controls, mt, bt, cfg, exp.augmented_penalty, include_bounds=True)
        record = {
            "completed": failure is None,
            "failure": failure,
            "seconds": time.perf_counter() - started,
            "control_mse_normalized": float(nn.functional.mse_loss((final_controls - low) / (high - low), (ut - low) / (high - low)).detach()),
            "kkt_residual": float(final_kkt["total"].detach()),
            "kkt_stationarity": float(final_kkt["stationarity"].detach()),
            "teacher_use_in_training": {
                "controls": True,
                "objectives": False,
                "anchor": method == "S-u+K",
                "path_multipliers": method == "S-u+K",
                "bound_multipliers": method == "S-u+K",
            },
        }
        torch.save({"model": model.state_dict(), "state_mean": mean, "state_std": std, "training": record}, args.output / f"{method}.pth")
        dump(args.output / f"{method}_training_log.json", {"training": record, "history": history})
        return model, record

    methods: dict[str, dict] = {}
    for name in ("S-u", "S-u+K"):
        print(f"[JCB seed {exp.seed}] training {name}", flush=True)
        model, training = train(name)
        deployment = evaluate(name, model, mean, std, test, references, cfg, exp, args.output)
        methods[name] = {"training": training, "deployment": deployment}

    cold_seconds = float(np.mean([row["solve_seconds"] for row in references]))
    for result in methods.values():
        result["deployment"]["mean_cold_qp_seconds"] = cold_seconds
        result["deployment"]["speedup_vs_cold_qp"] = cold_seconds / result["deployment"]["mean_total_predeployment_seconds"]

    summary = {
        "formal_protocol": False,
        "nonformal_exploration_warning": "DO NOT INCLUDE IN FORMAL TABLE",
        "benchmark": f"JCB-2D exact-flow CVP{cfg.zoh_steps} QP",
        "config": asdict(exp),
        "methods": methods,
        "label_source": str(args.labels),
        "label_cold_start_note": metadata["cold_start_protocol"],
        "teacher_kkt_self_check": teacher_check,
        "reference": {
            "method": f"same reduced-space exact-flow CVP{cfg.zoh_steps} QP, fixed zero-control cold start",
            "count": len(references),
            "audited": int(sum(row["audit_accepted"] for row in references)),
            "mean_seconds": cold_seconds,
            "source": reference_source,
        },
        "pcgrad": {
            "enabled": args.project_conflicting_kkt_gradient,
            "rule": "Project KKT-gradient components opposing control-MSE plus anchor during the 10 continuation epochs only.",
        },
        "kkt_structure": "path and box-bound finite-dimensional transcription KKT residual",
        "hds_statement": "Continuous-time numerical audit evidence under the declared model and numerical settings; not a real-system absolute safety guarantee.",
    }
    dump(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
