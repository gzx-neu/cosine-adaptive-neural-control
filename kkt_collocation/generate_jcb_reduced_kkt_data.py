"""Generate two-dimensional JCB CVP labels through its exact convex QP.

The original JCB dynamics are linear, its running cost is quadratic, and the
path constraint is affine in the state.  With a ZOH control-vector
parameterization, the finite-dimensional transcription is therefore a convex
box-constrained quadratic program.  This script constructs that QP exactly
for a chosen CVP resolution and solves each initial condition independently.

The default is CVP50 on the existing two-dimensional domain.  ``path_duals``
and ``bound_duals`` are finite-dimensional KKT quantities of this direct
transcription only; they are not continuous-time multiplier functions.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, minimize, nnls

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class JCBConfig:
    horizon: float = 1.0
    zoh_steps: int = 50
    # Number of exact constraint subnodes within each ZOH interval.  One means
    # the interval endpoint; the left endpoint of every interval is retained.
    substeps_per_zoh: int = 1
    control_bounds: tuple[float, float] = (-15.0, 15.0)
    x1_initial_range: tuple[float, float] = (-0.1, 0.1)
    x2_initial_range: tuple[float, float] = (-1.1, -0.9)
    solver_maxiter: int = 300
    node_margin: float = 0.0

    @property
    def zoh_duration(self) -> float:
        return self.horizon / self.zoh_steps

    @property
    def dt(self) -> float:
        return self.zoh_duration / self.substeps_per_zoh


class ReducedJCBQP:
    """Exact-flow CVP transcription, represented as a dense convex QP."""

    def __init__(self, cfg: JCBConfig) -> None:
        self.cfg = cfg
        self.path_count = cfg.zoh_steps * (cfg.substeps_per_zoh + 1)
        self._build_static_matrices()

    @staticmethod
    def _flow(base1: float, base2: float, map1: np.ndarray, map2: np.ndarray,
              control_index: int, tau: float) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Exact affine ZOH flow at time ``tau`` from an affine segment start."""
        e = np.exp(-tau)
        unit = np.zeros_like(map1); unit[control_index] = 1.0
        x2_base = e * base2
        x2_map = e * map2 + (1.0 - e) * unit
        x1_base = base1 + (1.0 - e) * base2
        x1_map = map1 + (1.0 - e) * map2 + (tau - 1.0 + e) * unit
        return x1_base, x2_base, x1_map, x2_map

    def _build_static_matrices(self) -> None:
        """Build H and A; their values do not depend on the initial state."""
        n, h, cfg = self.cfg.zoh_steps, self.cfg.zoh_duration, self.cfg
        xi, wi = np.polynomial.legendre.leggauss(5)
        qnodes = 0.5 * h * (xi + 1.0)
        H = np.zeros((n, n))
        A_rows: list[np.ndarray] = []

        # Initial-state-independent affine control maps.  The scalar base
        # placeholders are zero here; their p-dependent counterparts are
        # assembled in _linear_terms using the same propagation.
        b1 = b2 = 0.0
        m1 = np.zeros(n); m2 = np.zeros(n)
        for k in range(n):
            # g(t_k) = x2(t_k) - 8(t_k-.5)^2 + .5
            A_rows.append(m2.copy())
            for q, w in zip(qnodes, wi):
                _, _, qm1, qm2 = self._flow(b1, b2, m1, m2, k, float(q))
                H += cfg.zoh_duration * float(w) * (np.outer(qm1, qm1) + np.outer(qm2, qm2))
                H[k, k] += cfg.zoh_duration * float(w) * 0.005
            for node in range(1, cfg.substeps_per_zoh + 1):
                _, _, _, nm2 = self._flow(b1, b2, m1, m2, k, node * cfg.dt)
                A_rows.append(nm2)
            b1, b2, m1, m2 = self._flow(b1, b2, m1, m2, k, h)

        # Objective is u^T H u + c^T u + const, hence grad=2Hu+c.
        self.H = H
        self.A = np.asarray(A_rows)

    def _linear_terms(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Return objective linear coefficient c, path offset b, and constant."""
        n, h, cfg = self.cfg.zoh_steps, self.cfg.zoh_duration, self.cfg
        xi, wi = np.polynomial.legendre.leggauss(5)
        qnodes = 0.5 * h * (xi + 1.0)
        c = np.zeros(n); b_rows: list[float] = []; constant = 0.0
        b1, b2 = float(p[0]), float(p[1])
        m1 = np.zeros(n); m2 = np.zeros(n)
        for k in range(n):
            current_time = k * h
            b_rows.append(b2 - 8.0 * (current_time - 0.5) ** 2 + 0.5 + cfg.node_margin)
            for q, w in zip(qnodes, wi):
                qb1, qb2, qm1, qm2 = self._flow(b1, b2, m1, m2, k, float(q))
                scale = 0.5 * h * float(w)
                c += 2.0 * scale * (qb1 * qm1 + qb2 * qm2)
                constant += scale * (qb1 * qb1 + qb2 * qb2)
            for node in range(1, cfg.substeps_per_zoh + 1):
                tau = node * cfg.dt
                _, nb2, _, _ = self._flow(b1, b2, m1, m2, k, tau)
                t = current_time + tau
                b_rows.append(nb2 - 8.0 * (t - 0.5) ** 2 + 0.5 + cfg.node_margin)
            b1, b2, m1, m2 = self._flow(b1, b2, m1, m2, k, h)
        return c, np.asarray(b_rows), float(constant)

    def _reconstruct_multipliers(self, controls: np.ndarray, c: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
        grad = 2.0 * self.H @ controls + c
        g = self.A @ controls + b
        active_tol = 2.0e-4
        active_path = np.flatnonzero(g >= -active_tol)
        lower, upper = self.cfg.control_bounds
        active_lower = np.flatnonzero(controls - lower <= active_tol)
        active_upper = np.flatnonzero(upper - controls <= active_tol)
        blocks: list[np.ndarray] = []
        if active_path.size:
            blocks.append(self.A[active_path].T)
        if active_lower.size:
            blocks.append(-np.eye(self.cfg.zoh_steps)[:, active_lower])
        if active_upper.size:
            blocks.append(np.eye(self.cfg.zoh_steps)[:, active_upper])
        path_duals = np.zeros(self.path_count)
        bound_duals = np.zeros((2, self.cfg.zoh_steps))
        if not blocks:
            return path_duals, bound_duals, float(np.linalg.norm(grad)), 0
        design = np.column_stack(blocks)
        scale = np.maximum(np.linalg.norm(design, axis=0), 1.0e-12)
        multipliers, _ = nnls(design / scale, -grad, maxiter=10000)
        multipliers /= scale
        offset = 0
        if active_path.size:
            path_duals[active_path] = multipliers[offset:offset + active_path.size]
            offset += active_path.size
        if active_lower.size:
            bound_duals[0, active_lower] = multipliers[offset:offset + active_lower.size]
            offset += active_lower.size
        if active_upper.size:
            bound_duals[1, active_upper] = multipliers[offset:offset + active_upper.size]
        stationarity = grad + self.A.T @ path_duals - bound_duals[0] + bound_duals[1]
        return path_duals, bound_duals, float(np.linalg.norm(stationarity)), int(active_path.size)

    def solve(self, p: np.ndarray) -> dict[str, Any]:
        p = np.asarray(p, dtype=float).reshape(2)
        c, b, constant = self._linear_terms(p)
        fun = lambda u: float(u @ self.H @ u + c @ u + constant)
        jac = lambda u: 2.0 * self.H @ u + c
        started = time.perf_counter()
        constraint = LinearConstraint(self.A, -np.inf * np.ones(self.path_count), -b)
        result = minimize(
            fun, np.zeros(self.cfg.zoh_steps), jac=jac, method="SLSQP",
            bounds=[self.cfg.control_bounds] * self.cfg.zoh_steps, constraints=constraint,
            options={"maxiter": self.cfg.solver_maxiter, "ftol": 1.0e-11, "disp": False},
        )
        if not result.success:
            raise RuntimeError(result.message)
        controls = np.asarray(result.x, dtype=float)
        path_duals, bound_duals, stationarity, active_count = self._reconstruct_multipliers(controls, c, b)
        return {
            "initial_state_parameter": p.tolist(),
            "initial_state": [float(p[0]), float(p[1]), 0.0, 0.0],
            "controls": controls.tolist(), "objective": fun(controls),
            "path_duals": path_duals.tolist(), "bound_duals": bound_duals.tolist(),
            "kkt_stationarity_norm": stationarity,
            "active_discretized_path_constraints": active_count,
            "discretized_path_max_g": float(np.max(self.A @ controls + b - self.cfg.node_margin)),
            "solve_seconds": time.perf_counter() - started,
            "dual_source": "active-constraint NNLS reconstruction of the exact convex CVP quadratic program",
        }


_WORKER: ReducedJCBQP | None = None


def _worker_init(config: dict[str, Any]) -> None:
    global _WORKER
    _WORKER = ReducedJCBQP(JCBConfig(**config))


def _worker_solve(payload: tuple[int, list[float]]) -> dict[str, Any]:
    if _WORKER is None:
        raise RuntimeError("Worker is not initialized")
    index, state = payload
    record: dict[str, Any] = {"index": index, "initial_state_parameter": state}
    try:
        record.update(_WORKER.solve(np.asarray(state, dtype=float)))
        record["success"] = True
    except Exception as exc:
        record.update({"success": False, "error": f"{type(exc).__name__}: {exc}"})
    return record


def _read_existing(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    return {int(row["index"]): row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--zoh-steps", type=int, default=50)
    parser.add_argument("--substeps-per-zoh", type=int, default=1)
    parser.add_argument("--node-margin", type=float, default=0.0)
    parser.add_argument("--solver-maxiter", type=int, default=300)
    parser.add_argument("--x1-min", type=float, default=-0.1)
    parser.add_argument("--x1-max", type=float, default=0.1)
    parser.add_argument("--x2-min", type=float, default=-1.1)
    parser.add_argument("--x2-max", type=float, default=-0.9)
    parser.add_argument("--limit", type=int, default=None, help="Solve only the first N grid points (smoke test).")
    parser.add_argument("--output", type=Path, default=ROOT / "kkt_collocation" / "results" / "jcb2d_qp_cvp50_30x30")
    args = parser.parse_args()
    if args.grid_size < 2 or args.workers < 1 or args.zoh_steps < 1 or args.substeps_per_zoh < 1:
        raise ValueError("grid size must be >=2; worker, CVP, and node counts must be positive")
    cfg = JCBConfig(zoh_steps=args.zoh_steps, substeps_per_zoh=args.substeps_per_zoh,
                    node_margin=args.node_margin, solver_maxiter=args.solver_maxiter,
                    x1_initial_range=(args.x1_min, args.x1_max), x2_initial_range=(args.x2_min, args.x2_max))
    x1 = np.linspace(*cfg.x1_initial_range, args.grid_size)
    x2 = np.linspace(*cfg.x2_initial_range, args.grid_size)
    states = np.asarray([[a, b] for a in x1 for b in x2], dtype=float)
    if args.limit is not None:
        states = states[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    records_path = args.output / "records.jsonl"
    records = _read_existing(records_path)
    pending = [(i, state.tolist()) for i, state in enumerate(states) if i not in records]
    with records_path.open("a", encoding="utf-8") as handle:
        with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init, initargs=(asdict(cfg),)) as pool:
            for record in pool.imap_unordered(_worker_solve, pending, chunksize=1):
                records[int(record["index"])] = record
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                print(f"{len(records)}/{len(states)} success={record['success']}", flush=True)
    ordered = [records[i] for i in range(len(states))]
    records_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in ordered), encoding="utf-8")
    successful = [row for row in ordered if row.get("success")]
    summary = {
        "purpose": "Two-dimensional JCB CVP labels from an exact convex quadratic-program transcription.",
        "config": asdict(cfg), "grid_shape": [args.grid_size, args.grid_size],
        "labels_requested": len(states), "labels_successful": len(successful), "labels_failed": len(states) - len(successful),
        "mean_solve_seconds_successful": float(np.mean([row["solve_seconds"] for row in successful])) if successful else None,
        "multiplier_interpretation": "Finite-dimensional KKT quantities of the discrete exact-flow CVP quadratic program, not continuous-time multiplier functions.",
        "cold_start_protocol": f"Each label uses the fixed zero CVP{cfg.zoh_steps} control vector; no neighbour warm starts are used.",
        "continuous_time_audit": "Not applied while generating labels; HDS event-located auditing and correction remain deployment-stage procedures.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
