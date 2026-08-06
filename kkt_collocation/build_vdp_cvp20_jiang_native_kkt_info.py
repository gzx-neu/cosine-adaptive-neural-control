"""Reconstruct finite-dimensional CVP20 KKT multipliers on the native grid.

For each already-solved Jiang--Fu VDP label, this script evaluates the same
frozen native-grid differentiable transcription used at training time, detects
active path/control-bound constraints, and solves the nonnegative least-squares
stationarity reconstruction.  The output is label-side KKT information only;
it never uses a neural policy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kkt_collocation.run_vdp_cvp20_jiang_native_grid_ablation import Config, native_upper_bound_constraints  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-tolerance", type=float, default=2e-3)
    parser.add_argument("--bound-tolerance", type=float, default=2e-5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    if args.active_tolerance <= 0 or args.bound_tolerance <= 0:
        raise ValueError("activity tolerances must be positive")
    raw = np.load(args.labels)
    initial = raw["initial_state"].astype(np.float64)
    controls = raw["optimal_controls"].astype(np.float64)
    grids = np.nan_to_num(raw["final_constraint_grid_padded"].astype(np.float64), nan=0.0)
    lengths = raw["final_constraint_grid_length"].astype(np.int64)
    n, max_grid = grids.shape
    if initial.shape != (400, 3) or controls.shape != (400, 20) or n != 400:
        raise ValueError("Expected 400 VDP CVP20 labels")
    cfg = Config()
    path_mu = np.zeros((n, max_grid - 1), dtype=np.float64)
    lower_mu = np.zeros((n, cfg.zoh_steps), dtype=np.float64)
    upper_mu = np.zeros((n, cfg.zoh_steps), dtype=np.float64)
    stationarity_rms = np.empty(n, dtype=np.float64)
    active_path_count = np.empty(n, dtype=np.int64)
    active_bound_count = np.empty(n, dtype=np.int64)
    max_constraint = np.empty(n, dtype=np.float64)

    for row in range(n):
        u = torch.tensor(controls[row : row + 1], dtype=torch.float64, requires_grad=True)
        x0 = torch.tensor(initial[row : row + 1], dtype=torch.float64)
        grid = torch.tensor(grids[row : row + 1], dtype=torch.float64)
        length = torch.tensor(lengths[row : row + 1], dtype=torch.long)
        objective, constraints = native_upper_bound_constraints(x0, u, grid, length, cfg)
        count = int(lengths[row] - 1)
        c = constraints[0, :count]
        grad_j = torch.autograd.grad(objective.sum(), u, retain_graph=True)[0].detach().cpu().numpy().reshape(-1)
        c_np = c.detach().cpu().numpy()
        active_path = np.flatnonzero(c_np >= -args.active_tolerance)
        lower_gap = controls[row] - cfg.u_min
        upper_gap = cfg.u_max - controls[row]
        active_lower = np.flatnonzero(lower_gap <= args.bound_tolerance)
        active_upper = np.flatnonzero(upper_gap <= args.bound_tolerance)
        gradients: list[np.ndarray] = []
        placement: list[tuple[str, int]] = []
        for index in active_path:
            gradient = torch.autograd.grad(c[index], u, retain_graph=True)[0].detach().cpu().numpy().reshape(-1)
            gradients.append(gradient)
            placement.append(("path", int(index)))
        for index in active_lower:
            gradient = np.zeros(cfg.zoh_steps)
            gradient[index] = -1.0  # u_min-u <= 0
            gradients.append(gradient)
            placement.append(("lower", int(index)))
        for index in active_upper:
            gradient = np.zeros(cfg.zoh_steps)
            gradient[index] = 1.0  # u-u_max <= 0
            gradients.append(gradient)
            placement.append(("upper", int(index)))
        if gradients:
            multipliers, _ = nnls(np.stack(gradients, axis=1), -grad_j)
            residual = grad_j + np.stack(gradients, axis=0).T @ multipliers
            for (kind, index), value in zip(placement, multipliers):
                if kind == "path": path_mu[row, index] = value
                elif kind == "lower": lower_mu[row, index] = value
                else: upper_mu[row, index] = value
        else:
            residual = grad_j
        stationarity_rms[row] = float(np.sqrt(np.mean(np.square(residual))))
        active_path_count[row] = len(active_path)
        active_bound_count[row] = len(active_lower) + len(active_upper)
        max_constraint[row] = float(c_np.max())
        if (row + 1) % 25 == 0 or row == 0:
            print(f"reconstructed {row + 1}/{n}; rms={stationarity_rms[row]:.3e}; active_path={active_path_count[row]}", flush=True)

    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "native_grid_nnls_kkt.npz",
        path_multipliers=path_mu,
        lower_bound_multipliers=lower_mu,
        upper_bound_multipliers=upper_mu,
        stationarity_rms=stationarity_rms,
        active_path_count=active_path_count,
        active_bound_count=active_bound_count,
        max_native_constraint=max_constraint,
    )
    report = {
        "source_labels": str(args.labels),
        "records": int(n),
        "method": "active-set nonnegative least-squares reconstruction on the frozen Jiang--Fu CVP20 native upper-bound grid",
        "multipliers": "finite-dimensional transcription multipliers; not continuous-time multipliers",
        "active_path_tolerance": args.active_tolerance,
        "bound_tolerance": args.bound_tolerance,
        "stationarity_rms": {"mean": float(stationarity_rms.mean()), "max": float(stationarity_rms.max()), "median": float(np.median(stationarity_rms))},
        "active_path_constraints": {"mean": float(active_path_count.mean()), "max": int(active_path_count.max())},
        "active_bound_constraints": {"mean": float(active_bound_count.mean()), "max": int(active_bound_count.max())},
        "max_native_constraint": {"mean": float(max_constraint.mean()), "max": float(max_constraint.max())},
    }
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
