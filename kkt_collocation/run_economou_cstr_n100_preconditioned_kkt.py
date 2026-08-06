"""Single fixed preconditioned-KKT continuation ablation for CSTR N=100.

This is intentionally independent from the existing CSTR experiment.  It
changes only the stationarity term used during the 20-epoch continuation:
each raw control-gradient coordinate is divided by a fixed, training-label
force scale.  The scale is calculated before training from the 740 training
labels, never from validation or held-out test controls.  The complete
finite-dimensional reduced-transcription KKT system (path and box terms) is
otherwise unchanged.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from run_economou_cstr_supervised_hds import PolicyValue
from run_economou_cstr_two_stage_vs_kkt_only import (
    Config, ROOT, dump_json, evaluate_method, kkt_terms, load_labels,
    rollout_flat, set_seed,
)
from screen_economou_cstr_30x30 import EconomouScreenConfig


BASELINE = ROOT / "kkt_collocation/results/economou_cstr_n100_label_holdout_s_sk_single"
LABELS = ROOT / "kkt_collocation/results/economou_cstr_reduced_kkt_n100_rk10_margin0_ca030_050_t410_420/records.jsonl"
OUTPUT = ROOT / "kkt_collocation/results/economou_cstr_n100_preconditioned_kkt_single_retry"


def fixed_split():
    """Return exactly the prior 740 / 60 / 100 held-out-label split."""
    previous = json.loads((BASELINE / "summary.json").read_text(encoding="utf-8"))
    test = np.asarray(previous["split"]["test_label_indices"], dtype=int)
    validation = np.asarray(previous["split"]["validation_label_indices"], dtype=int)
    train = np.setdiff1d(np.arange(900), np.r_[test, validation], assume_unique=False)
    if (len(train), len(validation), len(test)) != (740, 60, 100):
        raise RuntimeError("Prior frozen split is not the expected 740/60/100 partition")
    return train, validation, test


def force_scale(initial, flat, path_duals, bound_duals, cstr, rho):
    """Fixed RMS force scale per physical control coordinate at teacher labels.

    The individual objective, path-dual, box-dual and augmented-penalty
    gradients are nonzero at a KKT point even though their signed sum is near
    zero.  Their RMS combination is therefore a suitable, fixed scale for
    the cancellation residual.  It is estimated solely on the training
    labels and detached before policy training.
    """
    objective, g = rollout_flat(initial, flat, cstr)
    low = torch.tensor([cstr.ti_bounds_K[0], cstr.flow_bounds[0]], dtype=flat.dtype, device=flat.device).repeat(cstr.zoh_steps)
    high = torch.tensor([cstr.ti_bounds_K[1], cstr.flow_bounds[1]], dtype=flat.dtype, device=flat.device).repeat(cstr.zoh_steps)
    mu_lo, mu_hi = bound_duals[:, 0], bound_duals[:, 1]
    pieces = (
        objective,
        (path_duals * g).sum(1),
        (mu_lo * (low - flat)).sum(1) + (mu_hi * (flat - high)).sum(1),
        .5 * rho * torch.relu(g).square().sum(1),
    )
    gradients = [torch.autograd.grad(piece.sum(), flat, retain_graph=True)[0] for piece in pieces]
    scale = torch.sqrt(sum(grad.square() for grad in gradients).mean(0)).clamp_min(1e-6)
    return scale.detach()


def preconditioned_terms(initial, flat, path_duals, bound_duals, cstr, rho, scale):
    """Full KKT residual with only stationarity coordinate-preconditioned."""
    objective, g = rollout_flat(initial, flat, cstr)
    low = torch.tensor([cstr.ti_bounds_K[0], cstr.flow_bounds[0]], dtype=flat.dtype, device=flat.device).repeat(cstr.zoh_steps)
    high = torch.tensor([cstr.ti_bounds_K[1], cstr.flow_bounds[1]], dtype=flat.dtype, device=flat.device).repeat(cstr.zoh_steps)
    mu_lo, mu_hi = bound_duals[:, 0], bound_duals[:, 1]
    violation = torch.relu(g)
    lagrangian = (objective + (path_duals * g).sum(1)
                  + (mu_lo * (low - flat)).sum(1) + (mu_hi * (flat - high)).sum(1)
                  + .5 * rho * violation.square().sum(1))
    gradient = torch.autograd.grad(lagrangian.sum(), flat, create_graph=True)[0]
    stationarity = (gradient / scale).square().mean()
    primal = violation.square().mean()
    comp_path = (path_duals * g).square().mean()
    comp_bounds = ((mu_lo * (low - flat)).square().mean()
                   + (mu_hi * (flat - high)).square().mean())
    return {"total": stationarity + primal + comp_path + comp_bounds,
            "stationarity_preconditioned": stationarity,
            "primal": primal, "complementarity_path": comp_path,
            "complementarity_bounds": comp_bounds}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    cfg = Config()
    set_seed(cfg.seed)

    label_summary = json.loads((LABELS.parent / "summary.json").read_text(encoding="utf-8"))
    cstr_config = dict(label_summary["config"])
    for name in ("ti_bounds_K", "flow_bounds", "ca_initial_range", "temperature_initial_range_K"):
        cstr_config[name] = tuple(cstr_config[name])
    cstr = EconomouScreenConfig(**cstr_config)
    if cstr.zoh_steps != 100 or cstr.substeps_per_zoh != 10:
        raise RuntimeError("This ablation is frozen to the N=100 RK10 CSTR labels")
    states_all, controls_all, objectives_all, path_all, bounds_all, _ = load_labels(LABELS)
    train_idx, validation_idx, test_idx = fixed_split()
    states, u_ref, j_ref = states_all[train_idx], controls_all[train_idx], objectives_all[train_idx]
    path, bounds = path_all[train_idx], bounds_all[train_idx]
    test_states = states_all[test_idx]
    references = json.loads((BASELINE / "heldout_label_references.json").read_text(encoding="utf-8"))
    np.save(OUTPUT / "validation_initial_conditions.npy", states_all[validation_idx])
    np.save(OUTPUT / "test_initial_conditions.npy", test_states)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(states, dtype=torch.float32, device=device)
    ur = torch.tensor(u_ref, dtype=torch.float32, device=device)
    jr = torch.tensor(j_ref[:, None], dtype=torch.float32, device=device)
    dr = torch.tensor(path, dtype=torch.float32, device=device)
    br = torch.tensor(bounds, dtype=torch.float32, device=device)
    mean = states[:, [0, 2]].mean(0)
    std = states[:, [0, 2]].std(0).clip(1e-6)
    normalized_x = torch.tensor((states[:, [0, 2]] - mean) / std, dtype=torch.float32, device=device)
    jmean, jstd = jr.mean(), jr.std().clamp_min(1e-6)
    low = torch.tensor([cstr.ti_bounds_K[0], cstr.flow_bounds[0]], dtype=torch.float32, device=device)
    span = torch.tensor([70.0, 1.0], dtype=torch.float32, device=device)

    # The scale calculation is a pre-training operation using training labels only.
    teacher_flat = ur.reshape(len(ur), -1).detach().clone().requires_grad_(True)
    scales = force_scale(x, teacher_flat, dr, br, cstr, cfg.augmented_penalty)
    scale_np = scales.detach().cpu().numpy()
    scale_summary = {
        "definition": "Per-control RMS of the individual objective, path-dual, box-dual, and augmented-penalty forces at the 740 training teacher labels.",
        "source": "training labels only; fixed and detached before all policy updates",
        "minimum": float(scale_np.min()), "maximum": float(scale_np.max()),
        "temperature_mean": float(scale_np[0::2].mean()), "flow_mean": float(scale_np[1::2].mean()),
    }
    dump_json(OUTPUT / "preconditioner.json", scale_summary)

    def metrics(model, use_preconditioned=False, anchor=None):
        predicted_j, predicted_u = model(normalized_x)
        result = {
            "control_mse": nn.functional.mse_loss((predicted_u - low) / span, (ur - low) / span),
            "objective_mse": nn.functional.mse_loss(predicted_j, (jr - jmean) / jstd),
        }
        result["supervised"] = result["control_mse"] + .1 * result["objective_mse"]
        flat = predicted_u.reshape(len(predicted_u), -1)
        if use_preconditioned:
            result.update({f"pre_{key}": value for key, value in preconditioned_terms(
                x, flat, dr, br, cstr, cfg.augmented_penalty, scales).items()})
        if anchor is not None:
            result["anchor"] = nn.functional.mse_loss((predicted_u - low) / span, (anchor - low) / span)
        return result

    # Deterministic reproduction of the original 200-epoch supervised stage.
    set_seed(cfg.seed)
    model = PolicyValue(cstr).to(device)
    history, failure = [], None
    began = time.perf_counter()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.supervised_lr)
    for epoch in range(1, cfg.supervised_epochs + 1):
        record = metrics(model)
        optimizer.zero_grad(set_to_none=True); record["supervised"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        history.append({"stage": "supervised", "epoch": epoch, "loss": float(record["supervised"].detach()),
                        **{key: float(value.detach()) for key, value in record.items()}})
    with torch.no_grad():
        _, anchor = model(normalized_x)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.continuation_lr)
    for epoch in range(1, cfg.continuation_epochs + 1):
        try:
            record = metrics(model, use_preconditioned=True, anchor=anchor)
            # Retain the original experiment's per-batch scalar normalization;
            # only the coordinate weighting inside stationarity has changed.
            loss = (record["supervised"]
                    + cfg.kkt_weight * record["pre_total"] / record["pre_total"].detach().clamp_min(1.0)
                    + cfg.anchor_weight * record["anchor"])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite continuation loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            history.append({"stage": "preconditioned_continuation", "epoch": epoch, "loss": float(loss.detach()),
                            **{key: float(value.detach()) for key, value in record.items()}})
            print(f"preconditioned continuation {epoch}/{cfg.continuation_epochs}: loss={float(loss.detach()):.3e}", flush=True)
        except (RuntimeError, FloatingPointError) as error:
            failure = f"{type(error).__name__}: {error}"
            break

    with torch.enable_grad():
        final_supervised = metrics(model)
        _, final_u = model(normalized_x)
        raw = kkt_terms(x, final_u.reshape(len(final_u), -1), dr, br, cstr, cfg.augmented_penalty)
        pre = preconditioned_terms(x, final_u.reshape(len(final_u), -1), dr, br, cstr, cfg.augmented_penalty, scales)
    training = {
        "completed": failure is None, "failure": failure, "seconds": time.perf_counter() - began,
        "control_mse_normalized": float(final_supervised["control_mse"].detach()),
        "objective_mse_normalized": float(final_supervised["objective_mse"].detach()),
        "raw_kkt_residual": float(raw["total"].detach()), "raw_kkt_stationarity": float(raw["stationarity"].detach()),
        "preconditioned_kkt_residual": float(pre["total"].detach()),
        "preconditioned_stationarity": float(pre["stationarity_preconditioned"].detach()),
    }
    torch.save({"model": model.state_dict(), "state_mean": mean, "state_std": std,
                "config": asdict(cfg), "training": training, "preconditioner": scale_summary}, OUTPUT / "S+K-preconditioned.pth")
    dump_json(OUTPUT / "S+K-preconditioned_training_log.json", {"training": training, "history": history})

    deployment = evaluate_method("S+K-preconditioned", model, mean, std, test_states, cstr, cfg, OUTPUT, references)
    baseline = json.loads((BASELINE / "summary.json").read_text(encoding="utf-8"))["methods"]
    result = {
        "status": "completed", "config": asdict(cfg), "label_source": str(LABELS),
        "protocol": "One fixed preconditioned stationarity continuation ablation. No validation/test model selection, no new cold-start solves, and no changes to HDS or lambda candidates.",
        "split": {"train_count": 740, "validation_count": 60, "test_count": 100,
                  "test_label_indices": test_idx.tolist(), "validation_label_indices": validation_idx.tolist()},
        "preconditioner": scale_summary,
        "reference": {"source": "the prior frozen held-out discrete teacher labels", "count": 100,
                      "note": "screening label-gap only; not a new cold-start NLP benchmark"},
        "methods": {"S+K-preconditioned": {"training": training, "deployment": deployment},
                    "existing_S": baseline["S"], "existing_raw_S+K": baseline["S+K"]},
        "hds_statement": "Continuous-time numerical audit evidence under the declared model and numerical settings; not a real-system absolute safety guarantee.",
    }
    dump_json(OUTPUT / "summary.json", result)
    table = ["| Method | HDS label-gap (%) | Corrected segments | Raw KKT residual | Training stable |",
             "|---|---:|---:|---:|---|"]
    for label, item in (("S", baseline["S"]), ("raw S+K", baseline["S+K"]), ("preconditioned S+K", result["methods"]["S+K-preconditioned"])):
        train, deploy = item["training"], item["deployment"]
        raw_value = train.get("kkt_residual", train.get("raw_kkt_residual"))
        table.append(f"| {label} | {deploy['mean_hds_relative_gap_percent']:.4f} | {deploy['mean_corrected_segments']:.2f} | {raw_value:.3e} | {train['completed']} |")
    (OUTPUT / "summary_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
