"""Unified four-method N=100 Economou CSTR training ablation.

This is a new training-only driver.  It does not read the frozen 400-point
test reference and refuses to overwrite an existing seed directory.  The
formal methods are S-u, S-uJ, S+K (200+10), and direct K-only (210), all from
one captured initialization and all using the same 200/10 learning-rate
calendar.
"""
from __future__ import annotations

import argparse
import copy
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kkt_collocation.run_economou_cstr_supervised_hds import PolicyValue
from kkt_collocation.run_economou_cstr_two_stage_vs_kkt_only import (
    ROOT,
    dump_json,
    kkt_terms,
    load_labels,
    set_seed,
)
from kkt_collocation.screen_economou_cstr_30x30 import EconomouScreenConfig


LABELS = ROOT / "kkt_collocation/results/economou_cstr_reduced_kkt_n100_rk10_margin0_ca030_050_t410_420/records.jsonl"
FORMAL_METHODS = ("S-u", "S-uJ", "S+K", "K-only")
EXPLORATORY_METHODS = FORMAL_METHODS + ("S-u+K",)


@dataclass(frozen=True)
class UnifiedCSTRConfig:
    seed: int
    supervised_epochs: int = 200
    continuation_epochs: int = 10
    supervised_lr: float = 1e-3
    continuation_lr: float = 1e-5
    objective_weight: float = 0.1
    kkt_weight: float = 0.01
    augmented_penalty: float = 10.0
    anchor_weight: float = 1.0
    rollout_consistency_weight: float = 0.0
    lambda_grid_size: int = 31
    optimizer_reset_at_epoch_200: bool = True

    @property
    def total_epochs(self) -> int:
        return self.supervised_epochs + self.continuation_epochs


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--labels", type=Path, default=LABELS,
        help="Complete 900-label RK10 CSTR teacher set.  This permits a separately named exploratory N=40 study without changing the formal N=100 default.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "kkt_collocation/results/unified_su_suj_sk_konly_20seeds_v1",
    )
    parser.add_argument("--methods", nargs="+", choices=EXPLORATORY_METHODS, default=FORMAL_METHODS)
    # These overrides are deliberately available only for separately labelled
    # exploratory runs.  The defaults remain the frozen formal protocol.
    parser.add_argument("--supervised-epochs", type=int, default=None)
    parser.add_argument("--continuation-epochs", type=int, default=None)
    parser.add_argument("--kkt-weight", type=float, default=None)
    parser.add_argument("--anchor-weight", type=float, default=None)
    parser.add_argument(
        "--project-conflicting-kkt-gradient", action="store_true",
        help="Exploratory S-u+K only: remove the KKT gradient component opposing supervised-u plus anchor.",
    )
    parser.add_argument(
        "--kkt-conflict-projection-fraction", type=float, default=0.0,
        help="Exploratory S-u+K only: fraction in [0,1] of the conflicting KKT component to remove (1 = PCGrad).",
    )
    parser.add_argument(
        "--cosine-adaptive-kkt-conflict-projection", action="store_true",
        help="Exploratory S-u+K only: eta_t=max(0,-cos(g_base,g_kkt)).",
    )
    parser.add_argument(
        "--history-adaptive-kkt-conflict-projection-q", type=float, default=None,
        help=(
            "Exploratory S-u+K only: with c=max(0,-cos) and an EMA m of c, "
            "use eta=c+(1-c)*m**q."
        ),
    )
    parser.add_argument(
        "--history-conflict-ema-beta", type=float, default=0.8,
        help="EMA beta for history-adaptive KKT conflict projection (default: 0.8).",
    )
    parser.add_argument(
        "--pcgrad-kkt-norm-cap-ratio", type=float, default=None,
        help=(
            "Exploratory S-u+K only: full PCGrad followed by an adaptive cap "
            "||g_K_projected|| <= ratio*||g_base||."
        ),
    )
    parser.add_argument(
        "--cosine-joint-kkt-projection-norm-balance", action="store_true",
        help=(
            "Exploratory S-u+K only: with c=max(0,-cos), use projection fraction c "
            "and scale the projected KKT gradient by R**(-c), "
            "R=max(1,||g_K_projected||/||g_base||)."
        ),
    )
    parser.add_argument(
        "--relative-convergence-kkt-scaling", action="store_true",
        help=(
            "Exploratory S-u+K only: full PCGrad followed by instantaneous "
            "clip(sqrt((R/R0)/(B/B0)), 0.1, 10); the first-epoch scale is 1."
        ),
    )
    parser.add_argument(
        "--sharpened-cosine-kkt-conflict-projection", action="store_true",
        help=(
            "Exploratory S-u+K only: with c=max(0,-cos), use the smooth "
            "projection fraction eta=c^2/(c^2+(1-c)^2)."
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.kkt_conflict_projection_fraction <= 1.0:
        raise ValueError("--kkt-conflict-projection-fraction must lie in [0, 1]")
    if (args.history_adaptive_kkt_conflict_projection_q is not None
            and args.history_adaptive_kkt_conflict_projection_q <= 0.0):
        raise ValueError("--history-adaptive-kkt-conflict-projection-q must be positive")
    if not 0.0 <= args.history_conflict_ema_beta < 1.0:
        raise ValueError("--history-conflict-ema-beta must lie in [0, 1)")
    if args.pcgrad_kkt_norm_cap_ratio is not None and args.pcgrad_kkt_norm_cap_ratio <= 0.0:
        raise ValueError("--pcgrad-kkt-norm-cap-ratio must be positive")
    if sum((args.project_conflicting_kkt_gradient,
            args.cosine_adaptive_kkt_conflict_projection,
            args.history_adaptive_kkt_conflict_projection_q is not None,
            args.pcgrad_kkt_norm_cap_ratio is not None,
            args.cosine_joint_kkt_projection_norm_balance,
            args.relative_convergence_kkt_scaling,
            args.sharpened_cosine_kkt_conflict_projection,
            args.kkt_conflict_projection_fraction != 0.0)) > 1:
        raise ValueError(
            "Choose only one fixed/full/cosine/history/norm-capped/joint/"
            "relative-convergence projection mode"
        )
    conflict_projection_fraction = 1.0 if args.project_conflicting_kkt_gradient else args.kkt_conflict_projection_fraction

    formal_cfg = UnifiedCSTRConfig(seed=args.seed)
    if args.smoke:
        cfg = UnifiedCSTRConfig(seed=args.seed, supervised_epochs=2, continuation_epochs=1)
    else:
        overrides = {
            "supervised_epochs": args.supervised_epochs,
            "continuation_epochs": args.continuation_epochs,
            "kkt_weight": args.kkt_weight,
            "anchor_weight": args.anchor_weight,
        }
        supplied = {key: value for key, value in overrides.items() if value is not None}
        cfg = UnifiedCSTRConfig(seed=args.seed, **supplied)
    if cfg.supervised_epochs < 1 or cfg.continuation_epochs < 1:
        raise ValueError("Both training-stage lengths must be positive")
    if cfg.kkt_weight < 0 or cfg.anchor_weight < 0:
        raise ValueError("KKT and anchor weights must be non-negative")
    is_formal_protocol = (
        not args.smoke
        and cfg == formal_cfg
        and conflict_projection_fraction == 0.0
        and not args.cosine_adaptive_kkt_conflict_projection
        and args.history_adaptive_kkt_conflict_projection_q is None
        and args.pcgrad_kkt_norm_cap_ratio is None
        and not args.cosine_joint_kkt_projection_norm_balance
        and not args.relative_convergence_kkt_scaling
        and not args.sharpened_cosine_kkt_conflict_projection
        and set(args.methods).issubset(FORMAL_METHODS)
    )
    suffix = "_SMOKE_NONFORMAL" if args.smoke else ""
    output_dir = args.output_root / "cstr" / f"seed{args.seed}{suffix}"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite independent directory: {output_dir}")
    output_dir.mkdir(parents=True)

    labels_path = args.labels.resolve()
    label_summary = json.loads((labels_path.parent / "summary.json").read_text(encoding="utf-8"))
    label_config = dict(label_summary["config"])
    for field in ("ti_bounds_K", "flow_bounds", "ca_initial_range", "temperature_initial_range_K"):
        label_config[field] = tuple(label_config[field])
    cstr = EconomouScreenConfig(**label_config)
    states, u_ref, j_ref, path_duals, bound_duals, label_stationarity = load_labels(labels_path)
    if len(states) != 900 or cstr.substeps_per_zoh != 10:
        raise ValueError("Expected complete 900-label RK10 CSTR data")
    if cstr.node_margin < 0:
        raise ValueError(f"node_margin must be non-negative, got {cstr.node_margin}")
    expected_path_components = 2 * cstr.zoh_steps * cstr.substeps_per_zoh
    expected_bound_components = 4 * cstr.zoh_steps
    if (path_duals.shape != (900, expected_path_components)
            or bound_duals.shape[0] != 900
            or int(np.prod(bound_duals.shape[1:])) != expected_bound_components):
        raise ValueError(
            f"Unexpected multiplier shapes path={path_duals.shape}, bounds={bound_duals.shape}"
        )

    # This audit is label-only and independent of the random seed.  It checks
    # the exact finite-dimensional transcription before any model is trained.
    x64 = torch.tensor(states, dtype=torch.float64)
    u64 = torch.tensor(u_ref.reshape(900, -1), dtype=torch.float64, requires_grad=True)
    d64 = torch.tensor(path_duals, dtype=torch.float64)
    b64 = torch.tensor(bound_duals, dtype=torch.float64)
    teacher = kkt_terms(x64, u64, d64, b64, cstr, cfg.augmented_penalty)
    teacher_rms = float(torch.sqrt(teacher["stationarity"]).detach())
    if not np.isfinite(teacher_rms) or teacher_rms > 1e-3:
        raise RuntimeError(f"Label KKT self-check failed: RMS stationarity={teacher_rms:.3e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(states, dtype=torch.float32, device=device)
    ur = torch.tensor(u_ref, dtype=torch.float32, device=device)
    jr = torch.tensor(j_ref[:, None], dtype=torch.float32, device=device)
    dr = torch.tensor(path_duals, dtype=torch.float32, device=device)
    br = torch.tensor(bound_duals, dtype=torch.float32, device=device)
    state_mean = states[:, [0, 2]].mean(0)
    state_std = states[:, [0, 2]].std(0).clip(1e-6)
    normalized_x = torch.tensor(
        (states[:, [0, 2]] - state_mean) / state_std,
        dtype=torch.float32,
        device=device,
    )
    objective_mean = jr.mean()
    objective_std = jr.std().clamp_min(1e-6)
    low = torch.tensor(
        [cstr.ti_bounds_K[0], cstr.flow_bounds[0]], dtype=torch.float32, device=device
    )
    span = torch.tensor(
        [cstr.ti_bounds_K[1] - cstr.ti_bounds_K[0], cstr.flow_bounds[1] - cstr.flow_bounds[0]],
        dtype=torch.float32,
        device=device,
    )

    def supervised_terms(model: PolicyValue, *, include_objective: bool) -> dict[str, torch.Tensor]:
        prediction_j, prediction_u = model(normalized_x)
        control_mse = nn.functional.mse_loss(
            (prediction_u - low) / span,
            (ur - low) / span,
        )
        objective_mse = nn.functional.mse_loss(
            prediction_j, (jr - objective_mean) / objective_std
        )
        loss = control_mse + (cfg.objective_weight * objective_mse if include_objective else 0.0)
        return {
            "loss": loss,
            "control_mse_normalized": control_mse,
            "objective_mse_normalized": objective_mse,
            "controls": prediction_u,
        }

    def discrete_kkt_terms(model: PolicyValue) -> dict[str, torch.Tensor]:
        # Sole loss path for K-only: no ur, jr, objective_mean/std, or anchor.
        _, prediction_u = model(normalized_x)
        terms = kkt_terms(
            x,
            prediction_u.reshape(len(x), -1),
            dr,
            br,
            cstr,
            cfg.augmented_penalty,
        )
        denominator = terms["total"].detach().clamp_min(torch.finfo(terms["total"].dtype).tiny)
        return {
            "loss": terms["total"] / denominator,
            "controls": prediction_u,
            **{f"kkt_{name}": value for name, value in terms.items() if name not in ("objective", "path_g")},
        }

    set_seed(cfg.seed)
    prototype = PolicyValue(cstr).to(device)
    initial_state = copy.deepcopy(prototype.state_dict())

    protocol = {
        "formal_protocol": is_formal_protocol,
        "nonformal_smoke_warning": "DO NOT INCLUDE IN FORMAL TABLE" if not is_formal_protocol else None,
        "benchmark": f"economou_cstr_n{cstr.zoh_steps}_rk10_margin{cstr.node_margin:g}",
        "methods": list(args.methods),
        "config": asdict(cfg),
        "exploration_overrides": {
            "supervised_epochs": args.supervised_epochs,
            "continuation_epochs": args.continuation_epochs,
            "kkt_weight": args.kkt_weight,
            "anchor_weight": args.anchor_weight,
            "project_conflicting_kkt_gradient": args.project_conflicting_kkt_gradient,
            "kkt_conflict_projection_fraction": conflict_projection_fraction,
            "cosine_adaptive_kkt_conflict_projection": args.cosine_adaptive_kkt_conflict_projection,
            "history_adaptive_kkt_conflict_projection_q": args.history_adaptive_kkt_conflict_projection_q,
            "history_conflict_ema_beta": args.history_conflict_ema_beta,
            "pcgrad_kkt_norm_cap_ratio": args.pcgrad_kkt_norm_cap_ratio,
            "cosine_joint_kkt_projection_norm_balance": args.cosine_joint_kkt_projection_norm_balance,
            "relative_convergence_kkt_scaling": args.relative_convergence_kkt_scaling,
            "sharpened_cosine_kkt_conflict_projection": args.sharpened_cosine_kkt_conflict_projection,
        },
        "label_source": str(labels_path),
        "label_protocol": (
            "900 independent cold-start reduced-space RK10 labels; "
            f"node_margin={cstr.node_margin:g}"
        ),
        "path_multiplier_components": expected_path_components,
        "control_bound_multiplier_components": expected_bound_components,
        "kkt_multiplier_statement": "finite-dimensional discretized-transcription NLP multipliers; not continuous-time multipliers",
        "test_isolation": "No frozen 400-point test reference is read by this training driver.",
        "hds_statement": "continuous-time numerical audit evidence under declared model/numerics; not a real-system absolute safety guarantee",
        "reproducibility": {
            "seed": cfg.seed,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": sys.version,
            "device": str(device),
        },
        "teacher_kkt_self_check": {
            "recorded_stationarity_norm_mean": float(label_stationarity.mean()),
            "torch_rms_stationarity": teacher_rms,
            "torch_total_kkt_residual": float(teacher["total"].detach()),
        },
    }
    dump_json(output_dir / "config.json", protocol)

    def train_stage(
        model: PolicyValue,
        *,
        method: str,
        stage: str,
        epochs: int,
        learning_rate: float,
        anchor: torch.Tensor | None,
        history: list[dict],
    ) -> tuple[bool, str | None]:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        conflict_strength_ema = 0.0
        relative_convergence_base0: float | None = None
        relative_convergence_kkt0: float | None = None
        for epoch in range(1, epochs + 1):
            try:
                if method == "K-only":
                    record = discrete_kkt_terms(model)
                    loss = record["loss"]
                elif stage in ("supervised", "supervised_decay"):
                    record = supervised_terms(model, include_objective=(method not in ("S-u", "S-u+K")))
                    loss = record["loss"]
                elif method in ("S+K", "S-u+K") and stage == "continuation":
                    supervised = supervised_terms(model, include_objective=(method == "S+K"))
                    kkt = discrete_kkt_terms(model)
                    anchor_mse = nn.functional.mse_loss(
                        (supervised["controls"] - low) / span,
                        (anchor - low) / span,
                    )
                    base_loss = supervised["loss"] + cfg.anchor_weight * anchor_mse
                    kkt_loss = cfg.kkt_weight * kkt["loss"]
                    loss = base_loss + kkt_loss
                    record = {
                        **supervised,
                        **{key: value for key, value in kkt.items() if key != "controls"},
                        "anchor_mse_normalized": anchor_mse,
                    }
                else:
                    raise ValueError(f"Unsupported stage {method}/{stage}")

                if not _finite(loss):
                    raise FloatingPointError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                if (
                    (conflict_projection_fraction > 0.0
                     or args.cosine_adaptive_kkt_conflict_projection
                     or args.history_adaptive_kkt_conflict_projection_q is not None
                     or args.pcgrad_kkt_norm_cap_ratio is not None
                     or args.cosine_joint_kkt_projection_norm_balance
                     or args.relative_convergence_kkt_scaling
                     or args.sharpened_cosine_kkt_conflict_projection)
                    and method == "S-u+K"
                    and stage == "continuation"
                ):
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
                    conflict_cosine = dot / torch.sqrt(base_norm_sq * kkt_norm_sq)
                    conflict_strength = torch.clamp(-conflict_cosine, min=0.0, max=1.0)
                    if (args.pcgrad_kkt_norm_cap_ratio is not None
                            or args.relative_convergence_kkt_scaling):
                        projection_fraction = torch.ones((), dtype=loss.dtype, device=loss.device)
                    elif args.cosine_joint_kkt_projection_norm_balance:
                        projection_fraction = conflict_strength
                    elif args.sharpened_cosine_kkt_conflict_projection:
                        one_minus_conflict = 1.0 - conflict_strength
                        projection_fraction = conflict_strength.square() / (
                            conflict_strength.square()
                            + one_minus_conflict.square()
                            + torch.finfo(loss.dtype).eps
                        )
                    elif (args.cosine_adaptive_kkt_conflict_projection
                            or args.history_adaptive_kkt_conflict_projection_q is not None):
                        if args.history_adaptive_kkt_conflict_projection_q is None:
                            projection_fraction = conflict_strength
                        else:
                            conflict_strength_ema = (
                                args.history_conflict_ema_beta * conflict_strength_ema
                                + (1.0 - args.history_conflict_ema_beta)
                                * float(conflict_strength.detach())
                            )
                            ema = torch.as_tensor(
                                conflict_strength_ema, dtype=loss.dtype, device=loss.device
                            )
                            projection_fraction = conflict_strength + (
                                (1.0 - conflict_strength)
                                * ema.pow(args.history_adaptive_kkt_conflict_projection_q)
                            )
                    else:
                        projection_fraction = torch.as_tensor(
                            conflict_projection_fraction, dtype=loss.dtype, device=loss.device
                        )
                    coefficient = projection_fraction * torch.minimum(
                        dot / base_norm_sq, torch.zeros_like(dot)
                    )
                    if (args.pcgrad_kkt_norm_cap_ratio is not None
                            or args.cosine_joint_kkt_projection_norm_balance
                            or args.relative_convergence_kkt_scaling):
                        projected_kkt_grad = []
                        for left, right in zip(base_grad, kkt_grad):
                            if right is None:
                                projected_kkt_grad.append(None)
                            elif left is None:
                                projected_kkt_grad.append(right)
                            else:
                                projected_kkt_grad.append(right - coefficient * left)
                        projected_kkt_norm_sq = sum(
                            (gradient * gradient).sum()
                            for gradient in projected_kkt_grad if gradient is not None
                        ).clamp_min(torch.finfo(loss.dtype).tiny)
                        if args.cosine_joint_kkt_projection_norm_balance:
                            norm_ratio = torch.clamp(
                                torch.sqrt(projected_kkt_norm_sq / base_norm_sq), min=1.0
                            )
                            norm_balance_scale = norm_ratio.pow(-conflict_strength)
                            record["kkt_projected_to_base_norm_ratio"] = norm_ratio.detach()
                        elif args.relative_convergence_kkt_scaling:
                            base_value = float(base_loss.detach())
                            kkt_value = float(kkt["kkt_total"].detach())
                            first_relative_epoch = relative_convergence_base0 is None
                            if first_relative_epoch:
                                relative_convergence_base0 = base_value
                                relative_convergence_kkt0 = kkt_value
                            assert relative_convergence_base0 is not None
                            assert relative_convergence_kkt0 is not None
                            rho_base = max(
                                base_value / max(relative_convergence_base0, 1e-12), 1e-12
                            )
                            rho_kkt = max(
                                kkt_value / max(relative_convergence_kkt0, 1e-12), 1e-12
                            )
                            scale_unclipped = (rho_kkt / rho_base) ** 0.5
                            scale_value = (
                                1.0 if first_relative_epoch
                                else min(10.0, max(0.1, scale_unclipped))
                            )
                            norm_balance_scale = torch.as_tensor(
                                scale_value, dtype=loss.dtype, device=loss.device
                            )
                            record.update({
                                "relative_convergence_base_loss": base_loss.detach(),
                                "relative_convergence_kkt_residual": kkt["kkt_total"].detach(),
                                "relative_convergence_base_loss_initial": torch.as_tensor(
                                    relative_convergence_base0,
                                    dtype=loss.dtype,
                                    device=loss.device,
                                ),
                                "relative_convergence_kkt_residual_initial": torch.as_tensor(
                                    relative_convergence_kkt0,
                                    dtype=loss.dtype,
                                    device=loss.device,
                                ),
                                "relative_convergence_rho_base": torch.as_tensor(
                                    rho_base, dtype=loss.dtype, device=loss.device
                                ),
                                "relative_convergence_rho_kkt": torch.as_tensor(
                                    rho_kkt, dtype=loss.dtype, device=loss.device
                                ),
                                "relative_convergence_scale_unclipped": torch.as_tensor(
                                    scale_unclipped, dtype=loss.dtype, device=loss.device
                                ),
                                "relative_convergence_kkt_scale": norm_balance_scale.detach(),
                            })
                        else:
                            norm_balance_scale = torch.clamp(
                                args.pcgrad_kkt_norm_cap_ratio
                                * torch.sqrt(base_norm_sq / projected_kkt_norm_sq),
                                max=1.0,
                            )
                        for parameter, left, projected in zip(
                            parameters, base_grad, projected_kkt_grad
                        ):
                            if left is None and projected is None:
                                continue
                            value = torch.zeros_like(parameter) if left is None else left
                            if projected is not None:
                                value = value + norm_balance_scale * projected
                            parameter.grad = value
                        record["base_gradient_norm"] = torch.sqrt(base_norm_sq).detach()
                        record["kkt_gradient_norm"] = torch.sqrt(kkt_norm_sq).detach()
                        record["kkt_projected_gradient_norm"] = torch.sqrt(
                            projected_kkt_norm_sq
                        ).detach()
                        record["kkt_norm_balance_scale"] = norm_balance_scale.detach()
                    else:
                        for parameter, left, right in zip(parameters, base_grad, kkt_grad):
                            if left is None and right is None:
                                continue
                            value = torch.zeros_like(parameter) if left is None else left
                            if right is not None:
                                value = value + right
                                if left is not None:
                                    value = value - coefficient * left
                            parameter.grad = value
                    record["kkt_gradient_projection_coefficient"] = coefficient.detach()
                    record["kkt_projection_fraction_used"] = torch.where(
                        dot < 0, projection_fraction, torch.zeros_like(projection_fraction)
                    ).detach()
                    record["kkt_conflict_cosine"] = conflict_cosine.detach()
                    if args.history_adaptive_kkt_conflict_projection_q is not None:
                        record["kkt_conflict_strength_ema"] = torch.as_tensor(
                            conflict_strength_ema, dtype=loss.dtype, device=loss.device
                        )
                else:
                    loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not _finite(gradient_norm):
                    raise FloatingPointError("non-finite gradient norm")
                optimizer.step()
                if not all(_finite(parameter) for parameter in model.parameters()):
                    raise FloatingPointError("non-finite model parameter after optimizer step")
                row = {
                    "stage": stage,
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "loss": float(loss.detach()),
                    "gradient_norm_before_clip": float(gradient_norm.detach()),
                }
                for key, value in record.items():
                    if key not in ("controls", "loss"):
                        row[key] = float(value.detach())
                history.append(row)
                if epoch == 1 or epoch == epochs or epoch % 20 == 0:
                    print(
                        f"[CSTR seed {cfg.seed}] {method} {stage} {epoch}/{epochs} "
                        f"loss={float(loss.detach()):.3e}",
                        flush=True,
                    )
            except (RuntimeError, FloatingPointError) as exc:
                return False, f"{type(exc).__name__}: {exc}"
        return True, None

    def run_method(method: str) -> dict:
        model = PolicyValue(cstr).to(device)
        model.load_state_dict(initial_state)
        history: list[dict] = []
        started = time.perf_counter()
        if method in ("S-u", "S-uJ"):
            stages = (
                ("supervised", cfg.supervised_epochs, cfg.supervised_lr),
                ("supervised_decay", cfg.continuation_epochs, cfg.continuation_lr),
            )
        elif method == "K-only":
            stages = (
                ("kkt_direct", cfg.supervised_epochs, cfg.supervised_lr),
                ("kkt_direct_decay", cfg.continuation_epochs, cfg.continuation_lr),
            )
        else:
            stages = (("supervised", cfg.supervised_epochs, cfg.supervised_lr),)

        completed = True
        failure: str | None = None
        for stage, epochs, learning_rate in stages:
            completed, failure = train_stage(
                model,
                method=method,
                stage=stage,
                epochs=epochs,
                learning_rate=learning_rate,
                anchor=None,
                history=history,
            )
            if not completed:
                break
        if completed and method in ("S+K", "S-u+K"):
            with torch.no_grad():
                _, frozen = model(normalized_x)
            completed, failure = train_stage(
                model,
                method=method,
                stage="continuation",
                epochs=cfg.continuation_epochs,
                learning_rate=cfg.continuation_lr,
                anchor=frozen.detach(),
                history=history,
            )

        training = {
            "completed": completed,
            "numerical_failure": not completed,
            "failure_reason": failure,
            "epochs_completed": len(history),
            "seconds": time.perf_counter() - started,
            "teacher_use_in_training": (
                {"controls": False, "objectives": False, "anchor": False, "path_multipliers": True, "bound_multipliers": True}
                if method == "K-only"
                else {"controls": True, "objectives": method not in ("S-u", "S-u+K"), "anchor": method in ("S+K", "S-u+K"), "path_multipliers": method in ("S+K", "S-u+K"), "bound_multipliers": method in ("S+K", "S-u+K")}
            ),
        }
        try:
            supervised = supervised_terms(model, include_objective=True)
            with torch.enable_grad():
                kkt = discrete_kkt_terms(model)
            training.update({
                "control_mse_normalized": float(supervised["control_mse_normalized"].detach()),
                "objective_mse_normalized": float(supervised["objective_mse_normalized"].detach()),
                "kkt_residual": float(kkt["kkt_total"].detach()),
                "kkt_stationarity": float(kkt["kkt_stationarity"].detach()),
                "kkt_primal": float(kkt["kkt_primal"].detach()),
                "kkt_complementarity_path": float(kkt["kkt_complementarity_path"].detach()),
                "kkt_complementarity_bounds": float(kkt["kkt_complementarity_bounds"].detach()),
            })
        except RuntimeError as exc:
            if completed:
                completed = False
                training["completed"] = False
                training["numerical_failure"] = True
                training["failure_reason"] = f"final diagnostic failed: {type(exc).__name__}: {exc}"

        torch.save(
            {
                "model": model.state_dict(),
                "state_mean": state_mean,
                "state_std": state_std,
                "config": asdict(cfg),
                "method": method,
                "training": training,
            },
            output_dir / f"{method}.pth",
        )
        dump_json(output_dir / f"{method}_training_log.json", {"training": training, "history": history})
        return training

    methods: dict[str, dict] = {}
    for method in args.methods:
        print(f"[CSTR seed {cfg.seed}] training {method}", flush=True)
        methods[method] = run_method(method)

    summary = {
        "status": "completed" if all(value["completed"] for value in methods.values()) else "completed_with_training_failures",
        "protocol": protocol,
        "methods": methods,
        "later_evaluation_requirement": "Evaluate every completed checkpoint on the same frozen 400-point cold-start reference; report all-400 and the 286-reference qualified subset.",
    }
    dump_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
