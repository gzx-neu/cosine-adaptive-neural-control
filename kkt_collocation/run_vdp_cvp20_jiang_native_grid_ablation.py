"""VDP CVP20 Jiang--Fu native-grid supervised/KKT continuation ablation.

This is deliberately separate from the historical 10-ZOH-node VDP scripts.
It consumes the already solved 20-ZOH Jiang--Fu labels.  Each stored path
multiplier remains paired with its *own* final upper-bound interval grid.

The differentiable constraint evaluator reproduces the authors' upper-bound
functional on those frozen grids with RK4.  It is therefore an independent
PyTorch numerical reproduction, not a claim of bitwise identity with the
MATLAB ode45/fmincon derivative implementation.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    horizon: float = 5.0
    zoh_steps: int = 20
    hidden_dim: int = 256
    supervised_epochs: int = 200
    continuation_epochs: int = 10
    supervised_lr: float = 1e-3
    continuation_lr: float = 1e-5
    kkt_weight: float = 1e-3
    anchor_weight: float = 1.0
    augmented_penalty: float = 10.0
    upper_bound_smoothing: float = 1e-3
    rk4_substeps_per_native_interval: int = 4
    u_min: float = -0.3
    u_max: float = 1.0

    @property
    def control_dt(self) -> float:
        return self.horizon / self.zoh_steps


class Policy(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.body = nn.Sequential(
            nn.Linear(2, cfg.hidden_dim), nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(),
        )
        self.controls = nn.Linear(cfg.hidden_dim, cfg.zoh_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.controls(self.body(x))
        return (self.cfg.u_min + self.cfg.u_max) / 2.0 + (self.cfg.u_max - self.cfg.u_min) / 2.0 * torch.tanh(raw)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vdp_rhs(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    x1, x2, _ = state.unbind(dim=1)
    return torch.stack(
        (control - x2 - x1 * (x2.square() - 1.0), x1, control.square() + x1.square() + x2.square()),
        dim=1,
    )


def _smooth_positive(value: torch.Tensor, smoothing: float) -> torch.Tensor:
    return 0.5 * (value + torch.sqrt(value.square() + smoothing * smoothing))


def native_upper_bound_constraints(
    initial: torch.Tensor,
    controls: torch.Tensor,
    grids: torch.Tensor,
    lengths: torch.Tensor,
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return terminal x3 and Jiang--Fu upper-bound constraints.

    The original MATLAB code resets q to g=-x1-.4 at every stored adaptive
    interval, then integrates qdot=smoothmax(gdot,0).  Grid padding is masked,
    so no padded value ever enters a constraint or a gradient.
    """
    batch, max_grid = grids.shape
    state = initial
    padded_constraints = torch.zeros((batch, max_grid - 1), dtype=state.dtype, device=state.device)
    rows = torch.arange(batch, device=state.device)
    for interval in range(max_grid - 1):
        active = lengths > interval + 1
        if not bool(active.any()):
            break
        active_f = active.to(state.dtype).unsqueeze(1)
        start = grids[:, interval]
        end = torch.where(active, grids[:, interval + 1], start)
        h = ((end - start) / cfg.rk4_substeps_per_native_interval).unsqueeze(1)
        q = -state[:, :1] - 0.4
        t = start
        for _ in range(cfg.rk4_substeps_per_native_interval):
            # Final grids include every control-switch time.  This defensive
            # clamp makes t=TF use the final ZOH segment rather than index 20.
            index = torch.floor((t + 1e-12) / cfg.control_dt).to(torch.long).clamp(0, cfg.zoh_steps - 1)
            u = controls[rows, index].unsqueeze(1)

            def coupled(x: torch.Tensor, qq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                dx = vdp_rhs(x, u.squeeze(1))
                return dx, _smooth_positive(-dx[:, :1], cfg.upper_bound_smoothing)

            k1x, k1q = coupled(state, q)
            k2x, k2q = coupled(state + 0.5 * h * k1x, q + 0.5 * h * k1q)
            k3x, k3q = coupled(state + 0.5 * h * k2x, q + 0.5 * h * k2q)
            k4x, k4q = coupled(state + h * k3x, q + h * k3q)
            next_state = state + h * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
            next_q = q + h * (k1q + 2.0 * k2q + 2.0 * k3q + k4q) / 6.0
            state = active_f * next_state + (1.0 - active_f) * state
            q = active_f * next_q + (1.0 - active_f) * q
            t = t + h.squeeze(1)
        padded_constraints[:, interval] = torch.where(active, q.squeeze(1), torch.zeros_like(q.squeeze(1)))
    return state[:, 2], padded_constraints


def kkt_terms(data: dict[str, torch.Tensor], model: Policy, cfg: Config) -> dict[str, torch.Tensor]:
    controls = model(data["input"])
    objective, constraints = native_upper_bound_constraints(
        data["initial"], controls, data["grids"], data["grid_lengths"], cfg
    )
    path_mask = data["path_mask"]
    mu = data["path_multipliers"]
    lower = data["lower_multipliers"]
    upper = data["upper_multipliers"]
    lagrangian = objective + (mu * constraints * path_mask).sum(dim=1)
    lagrangian = lagrangian + (lower * (cfg.u_min - controls) + upper * (controls - cfg.u_max)).sum(dim=1)
    stationarity = torch.autograd.grad(lagrangian.sum(), controls, create_graph=True)[0].square().mean()
    path_violation = (torch.relu(constraints) * path_mask).square().sum() / path_mask.sum().clamp_min(1.0)
    lower_g = cfg.u_min - controls
    upper_g = controls - cfg.u_max
    bound_violation = torch.relu(lower_g).square().mean() + torch.relu(upper_g).square().mean()
    complementarity = ((mu * constraints * path_mask).square().sum() / path_mask.sum().clamp_min(1.0))
    complementarity = complementarity + (lower * lower_g).square().mean() + (upper * upper_g).square().mean()
    total = stationarity + cfg.augmented_penalty * (path_violation + bound_violation) + complementarity
    return {"raw": total, "stationarity": stationarity, "primal": path_violation + bound_violation,
            "complementarity": complementarity, "controls": controls, "objective": objective}


def supervised_loss(data: dict[str, torch.Tensor], model: Policy, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    controls = model(data["input"])
    scale = cfg.u_max - cfg.u_min
    return nn.functional.mse_loss((controls - cfg.u_min) / scale, (data["controls"] - cfg.u_min) / scale), controls


def projected_step(model: Policy, optimizer: torch.optim.Optimizer, base: torch.Tensor, kkt: torch.Tensor | None, fraction: float) -> float:
    optimizer.zero_grad(set_to_none=True)
    params = [p for p in model.parameters() if p.requires_grad]
    if kkt is None or fraction == 0.0:
        (base if kkt is None else base + kkt).backward()
    else:
        bg = torch.autograd.grad(base, params, retain_graph=True)
        kg = torch.autograd.grad(kkt, params)
        dot = sum((a * b).sum() for a, b in zip(bg, kg))
        norm_sq = sum((a * a).sum() for a in bg).clamp_min(torch.finfo(base.dtype).tiny)
        coeff = fraction * torch.minimum(dot / norm_sq, torch.zeros_like(dot))
        for p, a, b in zip(params, bg, kg):
            p.grad = a + b - coeff * a
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    return float(norm.detach())


def load_data(path: Path, device: torch.device, kkt_info: Path | None = None) -> dict[str, torch.Tensor]:
    d = np.load(path)
    required = {"initial_state", "optimal_controls", "final_constraint_grid_padded", "final_constraint_grid_length",
                "path_multipliers_padded", "path_multiplier_length", "lower_bound_multipliers", "upper_bound_multipliers"}
    if not required.issubset(d.files):
        raise ValueError(f"missing labels: {sorted(required.difference(d.files))}")
    initial = d["initial_state"].astype(np.float32)
    controls = d["optimal_controls"].astype(np.float32)
    grids = d["final_constraint_grid_padded"].astype(np.float32)
    lengths = d["final_constraint_grid_length"].astype(np.int64)
    multiplier_lengths = d["path_multiplier_length"].astype(np.int64)
    if initial.shape != (400, 3) or controls.shape != (400, 20) or not np.all(lengths == multiplier_lengths + 1):
        raise ValueError("expected 400 successful Jiang--Fu CVP20 labels with matched grid/multiplier intervals")
    base_grid = np.linspace(0.0, 5.0, 21)
    if not all(np.max(np.min(np.abs(grids[i, :lengths[i], None] - base_grid), axis=0)) < 1e-6 for i in range(400)):
        raise ValueError("a final Jiang grid omits a ZOH control switch")
    path_mask = np.arange(grids.shape[1] - 1)[None, :] < multiplier_lengths[:, None]
    mean, std = initial[:, :2].mean(axis=0), np.maximum(initial[:, :2].std(axis=0), 1e-8)
    tensor = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype, device=device)
    if kkt_info is not None:
        rebuilt = np.load(kkt_info)
        required_kkt = {"path_multipliers", "lower_bound_multipliers", "upper_bound_multipliers", "stationarity_rms"}
        if not required_kkt.issubset(rebuilt.files):
            raise ValueError(f"missing reconstructed KKT fields: {sorted(required_kkt.difference(rebuilt.files))}")
        if rebuilt["path_multipliers"].shape != (400, grids.shape[1] - 1) or rebuilt["lower_bound_multipliers"].shape != (400, 20):
            raise ValueError("reconstructed KKT information does not match CVP20 labels")
        source_path = rebuilt["path_multipliers"].astype(np.float32)
        source_lower = rebuilt["lower_bound_multipliers"].astype(np.float32)
        source_upper = rebuilt["upper_bound_multipliers"].astype(np.float32)
    else:
        source_path = np.nan_to_num(d["path_multipliers_padded"].astype(np.float32), nan=0.0)
        source_lower = d["lower_bound_multipliers"].astype(np.float32)
        source_upper = d["upper_bound_multipliers"].astype(np.float32)
    return {"initial": tensor(initial), "controls": tensor(controls), "input": tensor((initial[:, :2] - mean) / std),
            "grids": tensor(np.nan_to_num(grids, nan=0.0)), "grid_lengths": tensor(lengths, torch.long),
            "path_multipliers": tensor(source_path), "path_mask": tensor(path_mask.astype(np.float32)),
            "lower_multipliers": tensor(source_lower), "upper_multipliers": tensor(source_upper),
            "input_mean": tensor(mean), "input_std": tensor(std)}


def train_method(name: str, model: Policy, data: dict[str, torch.Tensor], cfg: Config, projection_fraction: float) -> tuple[list[dict], dict]:
    history: list[dict] = []
    start = time.perf_counter()
    schedules = [("supervised", cfg.supervised_epochs, cfg.supervised_lr)]
    if name == "S-u":
        schedules.append(("supervised_decay", cfg.continuation_epochs, cfg.continuation_lr))
    anchor = None
    for stage, epochs, lr in schedules:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(1, epochs + 1):
            sup, controls = supervised_loss(data, model, cfg)
            if stage.startswith("supervised"):
                grad = projected_step(model, opt, sup, None, 0.0)
                row = {"stage": stage, "epoch": epoch, "loss": float(sup.detach()), "control_mse": float(sup.detach()), "gradient_norm": grad}
            else:
                raise AssertionError(stage)
            history.append(row)
            if epoch == 1 or epoch == epochs or epoch % 20 == 0:
                print(f"[{name}] {stage} {epoch}/{epochs} loss={row['loss']:.3e}", flush=True)
    if name != "S-u":
        with torch.no_grad():
            anchor = model(data["input"]).detach()
        opt = torch.optim.Adam(model.parameters(), lr=cfg.continuation_lr)
        for epoch in range(1, cfg.continuation_epochs + 1):
            sup, controls = supervised_loss(data, model, cfg)
            anchor_mse = nn.functional.mse_loss((controls - cfg.u_min) / (cfg.u_max - cfg.u_min), (anchor - cfg.u_min) / (cfg.u_max - cfg.u_min))
            kkt = kkt_terms(data, model, cfg)
            base = sup + cfg.anchor_weight * anchor_mse
            scaled_kkt = cfg.kkt_weight * kkt["raw"]
            grad = projected_step(model, opt, base, scaled_kkt, projection_fraction if name == "S-u+K-processed" else 0.0)
            row = {"stage": "continuation", "epoch": epoch, "loss": float((base + scaled_kkt).detach()),
                   "control_mse": float(sup.detach()), "anchor_mse": float(anchor_mse.detach()),
                   "kkt_raw": float(kkt["raw"].detach()), "stationarity": float(kkt["stationarity"].detach()),
                   "primal": float(kkt["primal"].detach()), "complementarity": float(kkt["complementarity"].detach()), "gradient_norm": grad}
            history.append(row)
            print(f"[{name}] continuation {epoch}/{cfg.continuation_epochs} loss={row['loss']:.3e} kkt={row['kkt_raw']:.3e}", flush=True)
    return history, {"train_seconds": time.perf_counter() - start, "epochs_completed": len(history), "completed": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "kkt_collocation/results/exploratory_vdp_cvp20_jiang_strict_v1/converted_train_native_grid/native_grid_labels.npz")
    parser.add_argument("--kkt-info", type=Path, default=None, help="NNLS-reconstructed multipliers from build_vdp_cvp20_jiang_native_kkt_info.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260771)
    parser.add_argument("--methods", nargs="+", choices=("S-u", "S-u+K", "S-u+K-processed"), default=("S-u", "S-u+K", "S-u+K-processed"))
    parser.add_argument("--projection-fraction", type=float, default=1.0)
    parser.add_argument("--kkt-weight", type=float, default=Config.kkt_weight)
    parser.add_argument("--anchor-weight", type=float, default=Config.anchor_weight)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    if not 0.0 <= args.projection_fraction <= 1.0:
        raise ValueError("projection fraction must be in [0,1]")
    if args.kkt_weight < 0.0 or args.anchor_weight < 0.0:
        raise ValueError("KKT and anchor weights must be nonnegative")
    cfg = (Config(supervised_epochs=2, continuation_epochs=1, kkt_weight=args.kkt_weight, anchor_weight=args.anchor_weight)
           if args.smoke else Config(kkt_weight=args.kkt_weight, anchor_weight=args.anchor_weight))
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data, device, args.kkt_info)
    args.output.mkdir(parents=True)
    prototype = Policy(cfg).to(device)
    base = copy.deepcopy(prototype.state_dict())
    summary = {"formal_protocol": False, "nonformal_reason": "Native-grid KKT is a differentiable RK4 reproduction of the Jiang--Fu upper-bound functional on frozen adaptive grids; it is not a MATLAB ode45 bitwise identity.",
               "seed": args.seed, "device": str(device), "label_source": str(args.data), "kkt_information_source": str(args.kkt_info) if args.kkt_info else "raw MATLAB solver multipliers (not NNLS reconstructed)", "methods": {}, "config": asdict(cfg),
               "multiplier_statement": "Finite-dimensional multipliers paired with their own Jiang--Fu upper-bound intervals; not continuous-time multipliers."}
    for name in args.methods:
        print(f"Training {name}", flush=True)
        model = Policy(cfg).to(device)
        model.load_state_dict(base)
        history, result = train_method(name, model, data, cfg, args.projection_fraction)
        torch.save({"model": model.state_dict(), "input_mean": data["input_mean"].cpu(), "input_std": data["input_std"].cpu(), "config": asdict(cfg), "method": name}, args.output / f"{name}.pth")
        (args.output / f"{name}_training_log.json").write_text(json.dumps({"training": result, "history": history}, indent=2), encoding="utf-8")
        summary["methods"][name] = result
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
