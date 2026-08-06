"""Unified four-method ablation for the VDP and penicillin benchmarks.

The formal protocol is deliberately fixed in this driver:

* S-u: 200 control-supervised epochs at 1e-3, then 10 at 1e-5.
* S-uJ: the same schedule with control and normalized-objective supervision.
* S+K: 200 S-uJ epochs followed by 10 discrete-KKT continuation epochs.
* K-only: 200 direct discrete-KKT epochs at 1e-3, then 10 at 1e-5.

All four branches load one captured random initialization.  K-only's training
path never reads teacher controls or teacher objectives.  Its only label-side
quantity is the finite-dimensional path-multiplier array belonging to the
benchmark's established reduced-space transcription.

This file is separate from historical result drivers and refuses to overwrite
an existing seed directory.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.kkt_regularization import (  # noqa: E402
    augmented_lagrangian_kkt_residual,
)
from kkt_collocation.run_two_stage_vs_kkt_only_ablation import (  # noqa: E402
    Problem,
    dump_json,
    evaluate,
    set_seed,
)


FORMAL_METHODS = ("S-u", "S-uJ", "S+K", "K-only")
EXPLORATORY_METHODS = FORMAL_METHODS + ("S-u+K",)


@dataclass(frozen=True)
class UnifiedConfig:
    supervised_epochs: int = 200
    continuation_epochs: int = 10
    supervised_learning_rate: float = 1e-3
    continuation_learning_rate: float = 1e-5
    kkt_weight_vdp: float = 1e-2
    kkt_weight_penicillin: float = 1e-2
    augmented_penalty: float = 10.0
    anchor_weight: float = 1.0
    objective_weight: float = 0.1
    lambda_grid_size: int = 31
    rollout_consistency_weight: float = 0.0
    optimizer_reset_at_epoch_200: bool = True
    project_conflicting_kkt_gradient: bool = False
    kkt_conflict_projection_fraction: float = 0.0
    adaptive_kkt_conflict_projection: bool = False
    cosine_adaptive_kkt_conflict_projection: bool = False
    norm_balanced_kkt_conflict_projection: bool = False
    cosine_joint_kkt_projection_norm_balance: bool = False
    relative_convergence_kkt_scaling: bool = False
    sharpened_cosine_kkt_conflict_projection: bool = False
    tensor_consensus_sharpened_kkt_conflict_projection: bool = False

    @property
    def total_epochs(self) -> int:
        return self.supervised_epochs + self.continuation_epochs


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().detach().cpu())


def _control_coordinates(problem: Problem, controls: torch.Tensor) -> torch.Tensor:
    if problem.name == "vdp":
        low = problem.train_cfg.u_min
        span = problem.train_cfg.u_max - problem.train_cfg.u_min
    else:
        low = 0.0
        span = 2.0
    return (controls - low) / span


def _supervised_terms(
    problem: Problem, model: nn.Module, *, include_objective: bool
) -> dict[str, torch.Tensor]:
    predicted_j, controls = model(problem.normalized_train_input())
    control_mse = nn.functional.mse_loss(
        _control_coordinates(problem, controls),
        _control_coordinates(problem, problem.u_ref),
    )
    objective_mse = nn.functional.mse_loss(
        predicted_j, (problem.j_ref - problem.j_mean) / problem.j_std
    )
    loss = control_mse
    if include_objective:
        loss = loss + problem.cfg.objective_weight * objective_mse
    return {
        "loss": loss,
        "control_mse_normalized": control_mse,
        "objective_mse_normalized": objective_mse,
        "controls": controls,
    }


def _kkt_terms(problem: Problem, model: nn.Module) -> dict[str, torch.Tensor]:
    # Deliberately do not access problem.u_ref, problem.j_ref, or an anchor in
    # this function.  It is the sole loss path used by K-only.
    _, controls = model(problem.normalized_train_input())
    objective, path_g = problem.rollout(controls)
    residual = augmented_lagrangian_kkt_residual(
        objective,
        controls,
        path_g,
        problem.mu_ref,
        problem.cfg.augmented_penalty,
    )
    denominator = residual.total.detach().clamp_min(torch.finfo(residual.total.dtype).tiny)
    return {
        "loss": residual.total / denominator,
        "kkt_raw": residual.total,
        "kkt_stationarity": residual.stationarity,
        "kkt_primal": residual.primal_feasibility,
        "kkt_complementarity": residual.complementarity,
        "controls": controls,
    }


def _optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    *,
    base_loss: torch.Tensor | None = None,
    kkt_loss: torch.Tensor | None = None,
    kkt_conflict_projection_fraction: float = 0.0,
    adaptive_kkt_conflict_projection: bool = False,
    cosine_adaptive_kkt_conflict_projection: bool = False,
    norm_balanced_kkt_conflict_projection: bool = False,
    cosine_joint_kkt_projection_norm_balance: bool = False,
    relative_convergence_kkt_scale: float | None = None,
    sharpened_cosine_kkt_conflict_projection: bool = False,
    tensor_consensus_sharpened_kkt_conflict_projection: bool = False,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    if not _finite(loss):
        raise FloatingPointError("non-finite training loss")
    optimizer.zero_grad(set_to_none=True)
    projection_fraction_used = 0.0
    conflict_ratio_value = 0.0
    conflict_cosine_value = 0.0
    base_gradient_norm_value = 0.0
    kkt_gradient_norm_value = 0.0
    projected_kkt_gradient_norm_value = 0.0
    kkt_gradient_scale_value = 1.0
    tensor_conflict_consensus_value = 0.0
    if (kkt_conflict_projection_fraction > 0.0 or adaptive_kkt_conflict_projection
            or cosine_adaptive_kkt_conflict_projection
            or norm_balanced_kkt_conflict_projection
            or cosine_joint_kkt_projection_norm_balance
            or relative_convergence_kkt_scale is not None
            or sharpened_cosine_kkt_conflict_projection
            or tensor_consensus_sharpened_kkt_conflict_projection):
        if base_loss is None or kkt_loss is None:
            raise ValueError("KKT conflict projection requires separate base and KKT losses")
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        base_grad = torch.autograd.grad(base_loss, parameters, retain_graph=True, allow_unused=True)
        kkt_grad = torch.autograd.grad(kkt_loss, parameters, allow_unused=True)
        dot = sum(
            (left * right).sum()
            for left, right in zip(base_grad, kkt_grad)
            if left is not None and right is not None
        )
        base_norm_sq = sum(
            (gradient * gradient).sum()
            for gradient in base_grad if gradient is not None
        ).clamp_min(torch.finfo(loss.dtype).tiny)
        kkt_norm_sq = sum(
            (gradient * gradient).sum()
            for gradient in kkt_grad if gradient is not None
        ).clamp_min(torch.finfo(loss.dtype).tiny)
        normalized_dot = dot / base_norm_sq
        conflict_ratio = torch.clamp(-normalized_dot, min=0.0)
        conflict_cosine = dot / torch.sqrt(base_norm_sq * kkt_norm_sq)
        tensor_weight_total = torch.zeros((), dtype=loss.dtype, device=loss.device)
        tensor_conflict_weight = torch.zeros((), dtype=loss.dtype, device=loss.device)
        if tensor_consensus_sharpened_kkt_conflict_projection:
            for left, right in zip(base_grad, kkt_grad):
                if left is None or right is None:
                    continue
                tensor_dot = (left * right).sum()
                tensor_weight = torch.sqrt(
                    (left * left).sum().clamp_min(torch.finfo(loss.dtype).tiny)
                    * (right * right).sum().clamp_min(torch.finfo(loss.dtype).tiny)
                )
                tensor_weight_total = tensor_weight_total + tensor_weight
                tensor_conflict_weight = tensor_conflict_weight + tensor_weight * (
                    tensor_dot < 0
                ).to(loss.dtype)
            tensor_conflict_consensus = tensor_conflict_weight / tensor_weight_total.clamp_min(
                torch.finfo(loss.dtype).tiny
            )
            tensor_conflict_consensus_value = float(tensor_conflict_consensus.detach())
        if (norm_balanced_kkt_conflict_projection
                or relative_convergence_kkt_scale is not None):
            projection_fraction = torch.ones((), dtype=loss.dtype, device=loss.device)
        elif cosine_joint_kkt_projection_norm_balance:
            projection_fraction = torch.clamp(-conflict_cosine, min=0.0, max=1.0)
        elif sharpened_cosine_kkt_conflict_projection:
            conflict_strength = torch.clamp(-conflict_cosine, min=0.0, max=1.0)
            one_minus_conflict = 1.0 - conflict_strength
            projection_fraction = conflict_strength.square() / (
                conflict_strength.square()
                + one_minus_conflict.square()
                + torch.finfo(loss.dtype).eps
            )
        elif tensor_consensus_sharpened_kkt_conflict_projection:
            conflict_strength = torch.clamp(-conflict_cosine, min=0.0, max=1.0)
            one_minus_conflict = 1.0 - conflict_strength
            sharpened_fraction = conflict_strength.square() / (
                conflict_strength.square()
                + one_minus_conflict.square()
                + torch.finfo(loss.dtype).eps
            )
            projection_fraction = conflict_strength + tensor_conflict_consensus * (
                sharpened_fraction - conflict_strength
            )
        elif cosine_adaptive_kkt_conflict_projection:
            projection_fraction = torch.clamp(-conflict_cosine, min=0.0, max=1.0)
        elif adaptive_kkt_conflict_projection:
            projection_fraction = torch.clamp(conflict_ratio, max=1.0)
        else:
            projection_fraction = torch.as_tensor(
                kkt_conflict_projection_fraction, dtype=loss.dtype, device=loss.device
            )
        coefficient = projection_fraction * torch.minimum(normalized_dot, torch.zeros_like(dot))
        projection_fraction_used = float(
            torch.where(dot < 0, projection_fraction, torch.zeros_like(projection_fraction)).detach()
        )
        conflict_ratio_value = float(conflict_ratio.detach())
        conflict_cosine_value = float(conflict_cosine.detach())
        if (norm_balanced_kkt_conflict_projection
                or cosine_joint_kkt_projection_norm_balance
                or relative_convergence_kkt_scale is not None):
            projected_kkt_grad = [
                None if right is None else (
                    right if left is None else right - coefficient * left
                )
                for left, right in zip(base_grad, kkt_grad)
            ]
            projected_kkt_norm_sq = sum(
                (gradient * gradient).sum()
                for gradient in projected_kkt_grad if gradient is not None
            )
            base_gradient_norm = torch.sqrt(base_norm_sq)
            kkt_gradient_norm = torch.sqrt(kkt_norm_sq)
            projected_kkt_gradient_norm = torch.sqrt(projected_kkt_norm_sq)
            if cosine_joint_kkt_projection_norm_balance:
                conflict_strength = torch.clamp(-conflict_cosine, min=0.0, max=1.0)
                norm_ratio = torch.clamp(
                    projected_kkt_gradient_norm / base_gradient_norm, min=1.0
                )
                kkt_gradient_scale = norm_ratio.pow(-conflict_strength)
            elif relative_convergence_kkt_scale is not None:
                kkt_gradient_scale = torch.as_tensor(
                    relative_convergence_kkt_scale, dtype=loss.dtype, device=loss.device
                )
            else:
                epsilon = torch.finfo(loss.dtype).eps
                kkt_gradient_scale = torch.clamp(
                    base_gradient_norm / (projected_kkt_gradient_norm + epsilon), max=1.0
                )
            base_gradient_norm_value = float(base_gradient_norm.detach())
            kkt_gradient_norm_value = float(kkt_gradient_norm.detach())
            projected_kkt_gradient_norm_value = float(projected_kkt_gradient_norm.detach())
            kkt_gradient_scale_value = float(kkt_gradient_scale.detach())
            for parameter, left, projected_right in zip(
                parameters, base_grad, projected_kkt_grad
            ):
                if left is None and projected_right is None:
                    continue
                value = torch.zeros_like(parameter) if left is None else left
                if projected_right is not None:
                    value = value + kkt_gradient_scale * projected_right
                parameter.grad = value
        else:
            # Preserve the established fixed/adaptive projection update exactly.
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
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not _finite(gradient_norm):
        raise FloatingPointError("non-finite gradient norm")
    optimizer.step()
    if not all(_finite(parameter) for parameter in model.parameters()):
        raise FloatingPointError("non-finite model parameter after optimizer step")
    return (
        float(gradient_norm.detach()), projection_fraction_used,
        conflict_ratio_value, conflict_cosine_value,
        base_gradient_norm_value, kkt_gradient_norm_value,
        projected_kkt_gradient_norm_value, kkt_gradient_scale_value,
        tensor_conflict_consensus_value,
    )


def _train_stage(
    problem: Problem,
    model: nn.Module,
    *,
    method: str,
    stage: str,
    epochs: int,
    learning_rate: float,
    anchor: torch.Tensor | None,
    history: list[dict],
) -> tuple[bool, str | None]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    relative_base_loss_initial: float | None = None
    relative_kkt_raw_initial: float | None = None
    for epoch in range(1, epochs + 1):
        try:
            base_loss: torch.Tensor | None = None
            kkt_loss: torch.Tensor | None = None
            relative_convergence_kkt_scale: float | None = None
            relative_convergence_diagnostics: dict[str, float] | None = None
            if method == "K-only":
                terms = _kkt_terms(problem, model)
                loss = terms["loss"]
            elif stage in ("supervised", "supervised_decay"):
                terms = _supervised_terms(
                    problem, model, include_objective=(method not in ("S-u", "S-u+K"))
                )
                loss = terms["loss"]
            elif method in ("S+K", "S-u+K") and stage == "continuation":
                supervised = _supervised_terms(problem, model, include_objective=(method == "S+K"))
                kkt = _kkt_terms(problem, model)
                anchor_mse = nn.functional.mse_loss(
                    _control_coordinates(problem, supervised["controls"]),
                    _control_coordinates(problem, anchor),
                )
                base_loss = supervised["loss"] + problem.cfg.anchor_weight * anchor_mse
                kkt_loss = problem.cfg.kkt_weight_vdp * kkt["loss"]
                loss = base_loss + kkt_loss
                terms = {
                    **supervised,
                    **{key: value for key, value in kkt.items() if key != "controls"},
                    "anchor_mse_normalized": anchor_mse,
                }
            else:
                raise ValueError(f"Unsupported stage {method}/{stage}")

            if (
                problem.cfg.relative_convergence_kkt_scaling
                and method == "S-u+K"
                and stage == "continuation"
            ):
                current_base_loss = float(base_loss.detach())
                current_kkt_raw = float(kkt["kkt_raw"].detach())
                if relative_base_loss_initial is None:
                    relative_base_loss_initial = current_base_loss
                    relative_kkt_raw_initial = current_kkt_raw
                if relative_base_loss_initial <= 0.0 or relative_kkt_raw_initial <= 0.0:
                    raise FloatingPointError(
                        "relative-convergence scaling requires positive initial losses"
                    )
                rho_base = max(current_base_loss / relative_base_loss_initial, 1e-12)
                rho_kkt = max(current_kkt_raw / relative_kkt_raw_initial, 1e-12)
                multiplier_unclipped = float(np.sqrt(rho_kkt / rho_base))
                relative_convergence_kkt_scale = (
                    1.0 if epoch == 1
                    else float(np.clip(multiplier_unclipped, 0.1, 10.0))
                )
                relative_convergence_diagnostics = {
                    "relative_convergence_base_loss": current_base_loss,
                    "relative_convergence_kkt_raw_residual": current_kkt_raw,
                    "relative_convergence_base_loss_initial": relative_base_loss_initial,
                    "relative_convergence_kkt_raw_residual_initial": relative_kkt_raw_initial,
                    "relative_convergence_rho_base": rho_base,
                    "relative_convergence_rho_kkt": rho_kkt,
                    "relative_convergence_multiplier_unclipped": multiplier_unclipped,
                    "relative_convergence_multiplier_applied": relative_convergence_kkt_scale,
                }

            (gradient_norm, projection_fraction_used, conflict_ratio, conflict_cosine,
             base_gradient_norm, kkt_gradient_norm, projected_kkt_gradient_norm,
             kkt_gradient_scale, tensor_conflict_consensus) = _optimizer_step(
                model, optimizer, loss,
                base_loss=base_loss,
                kkt_loss=kkt_loss,
                kkt_conflict_projection_fraction=(
                    problem.cfg.kkt_conflict_projection_fraction
                    if method == "S-u+K" and stage == "continuation" else 0.0
                ),
                adaptive_kkt_conflict_projection=(
                    problem.cfg.adaptive_kkt_conflict_projection
                    if method == "S-u+K" and stage == "continuation" else False
                ),
                cosine_adaptive_kkt_conflict_projection=(
                    problem.cfg.cosine_adaptive_kkt_conflict_projection
                    if method == "S-u+K" and stage == "continuation" else False
                ),
                norm_balanced_kkt_conflict_projection=(
                    problem.cfg.norm_balanced_kkt_conflict_projection
                    if method == "S-u+K" and stage == "continuation" else False
                ),
                cosine_joint_kkt_projection_norm_balance=(
                    problem.cfg.cosine_joint_kkt_projection_norm_balance
                    if method == "S-u+K" and stage == "continuation" else False
                ),
                relative_convergence_kkt_scale=relative_convergence_kkt_scale,
                sharpened_cosine_kkt_conflict_projection=(
                    problem.cfg.sharpened_cosine_kkt_conflict_projection
                    if method == "S-u+K" and stage == "continuation" else False
                ),
                tensor_consensus_sharpened_kkt_conflict_projection=(
                    problem.cfg.tensor_consensus_sharpened_kkt_conflict_projection
                    if method == "S-u+K" and stage == "continuation" else False
                ),
            )
            record = {
                "stage": stage,
                "epoch": epoch,
                "learning_rate": learning_rate,
                "loss": float(loss.detach()),
                "gradient_norm_before_clip": gradient_norm,
                "kkt_projection_fraction_used": projection_fraction_used,
                "kkt_conflict_ratio": conflict_ratio,
                "kkt_conflict_cosine": conflict_cosine,
                "base_gradient_norm": base_gradient_norm,
                "kkt_gradient_norm_before_projection": kkt_gradient_norm,
                "kkt_gradient_norm_after_projection": projected_kkt_gradient_norm,
                "kkt_gradient_scale": kkt_gradient_scale,
                "tensor_conflict_consensus": tensor_conflict_consensus,
            }
            if relative_convergence_diagnostics is not None:
                record.update(relative_convergence_diagnostics)
            for key, value in terms.items():
                if key not in ("controls", "loss"):
                    record[key] = float(value.detach())
            history.append(record)
            if epoch == 1 or epoch == epochs or epoch % 20 == 0:
                print(
                    f"[{problem.name} seed {problem.seed}] {method} {stage} "
                    f"{epoch}/{epochs} loss={float(loss.detach()):.3e}",
                    flush=True,
                )
        except (FloatingPointError, RuntimeError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
    return True, None


def train_method(
    problem: Problem, method: str, base_state: dict, output: Path
) -> tuple[nn.Module, dict, list[dict]]:
    if method not in EXPLORATORY_METHODS:
        raise ValueError(method)
    model = problem.make_model()
    model.load_state_dict(base_state)
    history: list[dict] = []
    started = time.perf_counter()
    completed = True
    failure: str | None = None

    if method == "S-u":
        stages = (("supervised", problem.cfg.supervised_epochs, problem.cfg.supervised_learning_rate),
                  ("supervised_decay", problem.cfg.continuation_epochs, problem.cfg.continuation_learning_rate))
    elif method == "S-uJ":
        stages = (("supervised", problem.cfg.supervised_epochs, problem.cfg.supervised_learning_rate),
                  ("supervised_decay", problem.cfg.continuation_epochs, problem.cfg.continuation_learning_rate))
    elif method == "K-only":
        stages = (("kkt_direct", problem.cfg.supervised_epochs, problem.cfg.supervised_learning_rate),
                  ("kkt_direct_decay", problem.cfg.continuation_epochs, problem.cfg.continuation_learning_rate))
    else:
        stages = (("supervised", problem.cfg.supervised_epochs, problem.cfg.supervised_learning_rate),)

    anchor: torch.Tensor | None = None
    for stage, epochs, learning_rate in stages:
        completed, failure = _train_stage(
            problem,
            model,
            method=method,
            stage=stage,
            epochs=epochs,
            learning_rate=learning_rate,
            anchor=anchor,
            history=history,
        )
        if not completed:
            break

    if completed and method in ("S+K", "S-u+K"):
        with torch.no_grad():
            _, anchor = model(problem.normalized_train_input())
        completed, failure = _train_stage(
            problem,
            model,
            method=method,
            stage="continuation",
            epochs=problem.cfg.continuation_epochs,
            learning_rate=problem.cfg.continuation_learning_rate,
            anchor=anchor.detach(),
            history=history,
        )

    elapsed = time.perf_counter() - started
    model.eval()
    diagnostics: dict[str, float | None] = {
        "final_control_mse_normalized": None,
        "final_objective_mse_normalized": None,
        "final_kkt_residual": None,
        "kkt_stationarity": None,
        "kkt_primal": None,
        "kkt_complementarity": None,
    }
    try:
        supervised = _supervised_terms(problem, model, include_objective=True)
        with torch.enable_grad():
            kkt = _kkt_terms(problem, model)
        diagnostics.update({
            "final_control_mse_normalized": float(supervised["control_mse_normalized"].detach()),
            "final_objective_mse_normalized": float(supervised["objective_mse_normalized"].detach()),
            "final_kkt_residual": float(kkt["kkt_raw"].detach()),
            "kkt_stationarity": float(kkt["kkt_stationarity"].detach()),
            "kkt_primal": float(kkt["kkt_primal"].detach()),
            "kkt_complementarity": float(kkt["kkt_complementarity"].detach()),
        })
    except RuntimeError as exc:
        if completed:
            completed = False
            failure = f"final diagnostic failed: {type(exc).__name__}: {exc}"

    training = {
        "completed": completed,
        "numerical_failure": not completed,
        "failure_reason": failure,
        "epochs_completed": len(history),
        "train_seconds": elapsed,
        **diagnostics,
        "teacher_use_in_training": (
            {"controls": False, "objectives": False, "anchor": False, "path_multipliers": True}
            if method == "K-only"
            else {"controls": True, "objectives": method not in ("S-u", "S-u+K"), "anchor": method in ("S+K", "S-u+K"), "path_multipliers": method in ("S+K", "S-u+K")}
        ),
    }
    checkpoint = {
        "model": model.state_dict(),
        "method": method,
        "problem": problem.name,
        "normalization": {
            "mean": problem.mean.detach().cpu(),
            "std": problem.std.detach().cpu(),
            "objective_mean": problem.j_mean.detach().cpu(),
            "objective_std": problem.j_std.detach().cpu(),
        },
        "training": training,
        "config": asdict(problem.cfg),
    }
    torch.save(checkpoint, output / f"{method}.pth")
    dump_json(output / f"{method}_training_log.json", {"training": training, "history": history})
    return model, training, history


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _deployment_summary(rows: list[dict], base: dict) -> dict:
    result = dict(base)
    for source, target in (
        ("nominal_relative_objective_gap", "nominal_gap_percent"),
        ("hds_relative_objective_gap", "hds_gap_percent"),
    ):
        value = result.get(source)
        result[target] = None if value is None or not np.isfinite(value) else 100.0 * value
    result["accepted_network_policy_gap_note"] = (
        "Objective gaps use accepted neural-network controls only. Optimizer fallback is never treated as a neural-policy result."
    )
    return result


def _seed_directory(output_root: Path, benchmark: str, seed: int, smoke: bool) -> Path:
    suffix = "_SMOKE_NONFORMAL" if smoke else ""
    return output_root / benchmark / f"seed{seed}{suffix}"


def run_seed(
    benchmark: str,
    seed: int,
    output_root: Path,
    *,
    phase: str,
    workers: int,
    smoke: bool,
    project_conflicting_kkt_gradient: bool = False,
    kkt_conflict_projection_fraction: float = 0.0,
    supervised_epochs: int | None = None,
    continuation_epochs: int | None = None,
    kkt_weight: float | None = None,
    augmented_penalty: float | None = None,
    anchor_weight: float | None = None,
    adaptive_kkt_conflict_projection: bool = False,
    cosine_adaptive_kkt_conflict_projection: bool = False,
    norm_balanced_kkt_conflict_projection: bool = False,
    cosine_joint_kkt_projection_norm_balance: bool = False,
    relative_convergence_kkt_scaling: bool = False,
    sharpened_cosine_kkt_conflict_projection: bool = False,
    tensor_consensus_sharpened_kkt_conflict_projection: bool = False,
    evaluation_stop: int | None = None,
    methods: tuple[str, ...] = FORMAL_METHODS,
) -> dict:
    formal_cfg = UnifiedConfig()
    cfg = (
        UnifiedConfig(
            supervised_epochs=2,
            continuation_epochs=1,
            norm_balanced_kkt_conflict_projection=norm_balanced_kkt_conflict_projection,
            cosine_joint_kkt_projection_norm_balance=cosine_joint_kkt_projection_norm_balance,
            relative_convergence_kkt_scaling=relative_convergence_kkt_scaling,
            sharpened_cosine_kkt_conflict_projection=sharpened_cosine_kkt_conflict_projection,
            tensor_consensus_sharpened_kkt_conflict_projection=(
                tensor_consensus_sharpened_kkt_conflict_projection
            ),
        )
        if smoke
        else UnifiedConfig(
            supervised_epochs=UnifiedConfig.supervised_epochs if supervised_epochs is None else supervised_epochs,
            continuation_epochs=UnifiedConfig.continuation_epochs if continuation_epochs is None else continuation_epochs,
            kkt_weight_vdp=UnifiedConfig.kkt_weight_vdp if kkt_weight is None else kkt_weight,
            kkt_weight_penicillin=UnifiedConfig.kkt_weight_penicillin if kkt_weight is None else kkt_weight,
            augmented_penalty=UnifiedConfig.augmented_penalty if augmented_penalty is None else augmented_penalty,
            anchor_weight=UnifiedConfig.anchor_weight if anchor_weight is None else anchor_weight,
            project_conflicting_kkt_gradient=project_conflicting_kkt_gradient,
            kkt_conflict_projection_fraction=(1.0 if project_conflicting_kkt_gradient else kkt_conflict_projection_fraction),
            adaptive_kkt_conflict_projection=adaptive_kkt_conflict_projection,
            cosine_adaptive_kkt_conflict_projection=cosine_adaptive_kkt_conflict_projection,
            norm_balanced_kkt_conflict_projection=norm_balanced_kkt_conflict_projection,
            cosine_joint_kkt_projection_norm_balance=cosine_joint_kkt_projection_norm_balance,
            relative_convergence_kkt_scaling=relative_convergence_kkt_scaling,
            sharpened_cosine_kkt_conflict_projection=sharpened_cosine_kkt_conflict_projection,
            tensor_consensus_sharpened_kkt_conflict_projection=(
                tensor_consensus_sharpened_kkt_conflict_projection
            ),
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_formal_protocol = (
        not smoke and not project_conflicting_kkt_gradient and not adaptive_kkt_conflict_projection
        and not cosine_adaptive_kkt_conflict_projection
        and not norm_balanced_kkt_conflict_projection
        and not cosine_joint_kkt_projection_norm_balance
        and not relative_convergence_kkt_scaling
        and not sharpened_cosine_kkt_conflict_projection
        and not tensor_consensus_sharpened_kkt_conflict_projection
        and kkt_conflict_projection_fraction == 0.0 and tuple(methods) == FORMAL_METHODS
    )
    set_seed(seed)
    problem = Problem(benchmark, device, seed, cfg)
    output = _seed_directory(output_root, benchmark, seed, smoke)

    if phase in ("train", "all"):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite seed directory: {output}")
        output.mkdir(parents=True)
        np.save(output / "validation_states.npy", problem.validation_states)
        np.save(output / "test_states.npy", problem.test_states)
        config = {
            "formal_protocol": is_formal_protocol,
            "nonformal_smoke_warning": "DO NOT INCLUDE IN FORMAL TABLE" if not is_formal_protocol else None,
            "benchmark": benchmark,
            "seed": seed,
            "methods": methods,
            "experiment": asdict(cfg),
            "device": str(device),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "label_source": str(problem.data_path),
            "fixed_split_source": "existing main-experiment validation/test arrays",
            "validation_sha256": _array_sha256(problem.validation_states),
            "test_sha256": _array_sha256(problem.test_states),
            "objective_direction": "minimize terminal cost" if benchmark == "vdp" else "minimize J=-final_x3",
            "kkt_multiplier_statement": "finite-dimensional discretized-transcription NLP path multipliers; not continuous-time multipliers",
            "hds_statement": "continuous-time numerical audit evidence under the declared model and numerical settings; not a real-system absolute safety guarantee",
            "optimizer_fallback_statement": "optimizer fallback, not neural-policy result",
        }
        dump_json(output / "config.json", config)
        set_seed(seed)
        prototype = problem.make_model()
        base_state = copy.deepcopy(prototype.state_dict())
        training: dict[str, dict] = {}
        for method in methods:
            print(f"[{benchmark} seed {seed}] training {method}", flush=True)
            _, method_training, _ = train_method(problem, method, base_state, output)
            training[method] = method_training
        dump_json(output / "training_summary.json", {"config": config, "methods": training})
        dump_json(output / "validation_summary.json", {
            "validation_samples": int(len(problem.validation_states)),
            "validation_sha256": config["validation_sha256"],
            "selection_statement": "No validation or test result changes the frozen formal hyperparameters, epochs, methods, or seeds.",
        })

    if phase in ("evaluate", "all"):
        if not output.exists():
            raise FileNotFoundError(f"Missing trained seed directory: {output}")
        deployment: dict[str, dict] = {}
        training_summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
        evaluation_stop = 3 if smoke else evaluation_stop
        for method in methods:
            training = training_summary["methods"][method]
            if not training["completed"]:
                deployment[method] = {
                    "not_evaluable": True,
                    "reason": "training numerical failure; no deployable neural policy",
                    "optimizer_fallback_is_neural_policy": False,
                }
                continue
            checkpoint = torch.load(output / f"{method}.pth", map_location=device, weights_only=False)
            model = problem.make_model()
            model.load_state_dict(checkpoint["model"])
            model.eval()
            print(f"[{benchmark} seed {seed}] HDS evaluation {method}", flush=True)
            rows, summary = evaluate(problem, method, model, 0.0, stop=evaluation_stop, workers=workers)
            _write_rows(output / f"test_per_sample_{method}.csv", rows)
            deployment[method] = _deployment_summary(rows, summary)
        result = {
            "formal_protocol": is_formal_protocol,
            "benchmark": benchmark,
            "seed": seed,
            "training": training_summary["methods"],
            "deployment": deployment,
            "reference_gap_note": (
                "VDP and penicillin gaps use frozen pointwise Jiang--Fu cold-start references in matching order. "
                + ("This exploratory screen evaluates only the declared leading subset."
                   if evaluation_stop is not None else "All 400 test points are included.")
            ),
        }
        dump_json(output / "summary.json", result)
        return result

    return json.loads((output / "training_summary.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("vdp", "penicillin"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "kkt_collocation/results/unified_su_suj_sk_konly_20seeds_v1",
    )
    parser.add_argument("--phase", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--methods", nargs="+", choices=EXPLORATORY_METHODS, default=FORMAL_METHODS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--project-conflicting-kkt-gradient", action="store_true")
    parser.add_argument(
        "--adaptive-kkt-conflict-projection", action="store_true",
        help="Exploratory: eta_t=min(1,max(0,-<g_base,g_kkt>/||g_base||^2)).",
    )
    parser.add_argument(
        "--cosine-adaptive-kkt-conflict-projection", action="store_true",
        help="Exploratory: eta_t=max(0,-cos(g_base,g_kkt)).",
    )
    parser.add_argument(
        "--norm-balanced-kkt-conflict-projection", action="store_true",
        help=(
            "Exploratory S-u+K only: fully project conflicting KKT gradient, then "
            "scale it by min(1, ||g_base||/(||g_kkt_projected||+eps))."
        ),
    )
    parser.add_argument(
        "--cosine-joint-kkt-projection-norm-balance", action="store_true",
        help=(
            "Exploratory S-u+K only: c=max(0,-cos) controls both projection "
            "fraction c and projected-KKT scale R**(-c)."
        ),
    )
    parser.add_argument(
        "--relative-convergence-kkt-scaling", action="store_true",
        help=(
            "Exploratory S-u+K only: full PCGrad, then scale projected KKT by "
            "clip(sqrt((R/R0)/(B/B0)), 0.1, 10), with first continuation epoch 1."
        ),
    )
    parser.add_argument(
        "--sharpened-cosine-kkt-conflict-projection", action="store_true",
        help=(
            "Exploratory S-u+K only: eta=c^2/(c^2+(1-c)^2), "
            "where c=max(0,-cos(g_base,g_kkt))."
        ),
    )
    parser.add_argument(
        "--tensor-consensus-sharpened-kkt-conflict-projection", action="store_true",
        help=(
            "Exploratory S-u+K only: interpolate linear and sharpened cosine "
            "projection using the norm-product-weighted fraction of conflicting "
            "parameter tensors."
        ),
    )
    parser.add_argument("--continuation-epochs", type=int, default=None, help="Exploratory override; S-u remains 200+n epochs for fairness.")
    parser.add_argument("--supervised-epochs", type=int, default=None, help="Exploratory supervised-stage override.")
    parser.add_argument("--evaluation-stop", type=int, default=None, help="Exploratory leading test-subset size; omit for all 400.")
    parser.add_argument("--kkt-weight", type=float, default=None, help="Exploratory alpha_KKT override. Does not rescale label multipliers.")
    parser.add_argument("--augmented-penalty", type=float, default=None, help="Exploratory rho override.")
    parser.add_argument("--anchor-weight", type=float, default=None, help="Exploratory anchor-weight override.")
    parser.add_argument(
        "--kkt-conflict-projection-fraction", type=float, default=0.0,
        help="Exploratory S-u+K only: fraction in [0,1] of the conflicting KKT component to remove (1 = PCGrad).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if not 0.0 <= args.kkt_conflict_projection_fraction <= 1.0:
        raise ValueError("--kkt-conflict-projection-fraction must lie in [0, 1]")
    if sum((args.project_conflicting_kkt_gradient, args.adaptive_kkt_conflict_projection,
            args.cosine_adaptive_kkt_conflict_projection,
            args.norm_balanced_kkt_conflict_projection,
            args.cosine_joint_kkt_projection_norm_balance,
            args.relative_convergence_kkt_scaling,
            args.sharpened_cosine_kkt_conflict_projection,
            args.tensor_consensus_sharpened_kkt_conflict_projection,
            args.kkt_conflict_projection_fraction != 0.0)) > 1:
        raise ValueError("Choose only one KKT conflict-projection mode")
    if any(value is not None and value < 1 for value in (args.supervised_epochs, args.continuation_epochs, args.evaluation_stop)):
        raise ValueError("epoch and evaluation-stop overrides must be positive")
    if any(value is not None and value < 0.0 for value in (args.kkt_weight, args.augmented_penalty, args.anchor_weight)):
        raise ValueError("exploratory weights must be nonnegative")
    result = run_seed(
        args.benchmark,
        args.seed,
        args.output_root,
        phase=args.phase,
        workers=args.workers,
        smoke=args.smoke,
        project_conflicting_kkt_gradient=args.project_conflicting_kkt_gradient,
        kkt_conflict_projection_fraction=args.kkt_conflict_projection_fraction,
        supervised_epochs=args.supervised_epochs,
        continuation_epochs=args.continuation_epochs,
        kkt_weight=args.kkt_weight,
        augmented_penalty=args.augmented_penalty,
        anchor_weight=args.anchor_weight,
        adaptive_kkt_conflict_projection=args.adaptive_kkt_conflict_projection,
        cosine_adaptive_kkt_conflict_projection=args.cosine_adaptive_kkt_conflict_projection,
        norm_balanced_kkt_conflict_projection=args.norm_balanced_kkt_conflict_projection,
        cosine_joint_kkt_projection_norm_balance=args.cosine_joint_kkt_projection_norm_balance,
        relative_convergence_kkt_scaling=args.relative_convergence_kkt_scaling,
        sharpened_cosine_kkt_conflict_projection=args.sharpened_cosine_kkt_conflict_projection,
        tensor_consensus_sharpened_kkt_conflict_projection=(
            args.tensor_consensus_sharpened_kkt_conflict_projection
        ),
        evaluation_stop=args.evaluation_stop,
        methods=tuple(args.methods),
    )
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
