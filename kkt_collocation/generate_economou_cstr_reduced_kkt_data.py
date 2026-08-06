"""Parallel reduced-space KKT-label generation for the finite-horizon Economou CSTR.

The state trajectory is eliminated by an RK4 map, as in the penicillin KKT
label generator.  Thus the nonlinear program has only the 20 physical ZOH
controls as decision variables, while the path multipliers remain finite-
dimensional quantities associated with the 10-substep RK4 transcription.
Every label is an independent cold start; neighbouring labels are never used
as primal warm starts.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kkt_collocation.screen_economou_cstr_30x30 import (  # noqa: E402
    EconomouScreenConfig,
    _casadi,
    economou_ode,
    initial_grid,
)


class ReducedEconomouCSTR:
    """Control-only RK4 transcription and finite-dimensional KKT labels."""

    def __init__(self, cfg: EconomouScreenConfig) -> None:
        self.cfg = cfg
        self.ca = _casadi()
        self.control_size = 2 * cfg.zoh_steps
        self.path_count = 2 * cfg.zoh_steps * cfg.substeps_per_zoh
        self._build()

    def _f(self, x: Any, u: Any) -> Any:
        c, cfg = self.ca, self.cfg
        ca_, cb_, temperature = x[0], x[1], x[2]
        ti, flow = u[0], u[1]
        k1 = cfg.c1_s_inv * c.exp(-cfg.e1_cal_mol / (cfg.gas_constant_cal_mol_K * temperature))
        k2 = cfg.c2_s_inv * c.exp(-cfg.e2_cal_mol / (cfg.gas_constant_cal_mol_K * temperature))
        rate = k1 * ca_ - k2 * cb_
        dilution = flow / cfg.residence_time_s
        heat_scale = cfg.minus_delta_h_cal_mol / (cfg.density_kg_L * cfg.heat_capacity_cal_kg_K)
        return c.vertcat(
            dilution * (cfg.ca_feed - ca_) - rate,
            dilution * (cfg.cb_feed - cb_) + rate,
            dilution * (ti - temperature) + heat_scale * rate,
        )

    def _rk4(self, x: Any, u: Any) -> Any:
        h = self.cfg.dt
        k1 = self._f(x, u)
        k2 = self._f(x + 0.5 * h * k1, u)
        k3 = self._f(x + 0.5 * h * k2, u)
        k4 = self._f(x + h * k3, u)
        return x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def _build(self) -> None:
        c, cfg = self.ca, self.cfg
        u_flat = c.SX.sym("u", self.control_size)
        p = c.SX.sym("p", 3)
        u = c.reshape(u_flat, 2, cfg.zoh_steps)
        x = p
        running_cost = 0
        paths: list[Any] = []
        for j in range(cfg.zoh_steps * cfg.substeps_per_zoh):
            current_u = u[:, j // cfg.substeps_per_zoh]
            running_cost += cfg.dt * (
                -current_u[1] - 2.009 * x[1] + (1.657e-3 * current_u[0]) ** 2
            )
            x = self._rk4(x, current_u)
            # Initial feasibility is provided by the operating-domain guard;
            # the same fixed interior margin used by the earlier screen is
            # imposed after the first dynamics step.
            paths.extend((x[0] - (cfg.ca_max - cfg.node_margin),
                          x[2] - (cfg.temperature_max_K - cfg.node_margin)))
        objective = running_cost / cfg.horizon_s
        g = c.vertcat(*paths)
        self.objective = c.Function("economou_reduced_objective", [u_flat, p], [objective])
        self.gradient = c.Function("economou_reduced_gradient", [u_flat, p], [c.gradient(objective, u_flat)])
        self.constraints = c.Function("economou_reduced_constraints", [u_flat, p], [g])
        self.jacobian = c.Function("economou_reduced_jacobian", [u_flat, p], [c.jacobian(g, u_flat)])
        low = np.tile(np.array([cfg.ti_bounds_K[0], cfg.flow_bounds[0]], dtype=float), cfg.zoh_steps)
        high = np.tile(np.array([cfg.ti_bounds_K[1], cfg.flow_bounds[1]], dtype=float), cfg.zoh_steps)
        self.bounds = list(zip(low, high))
        self.lower, self.upper = low, high

    def initial_guess(self, controls: np.ndarray | None = None) -> np.ndarray:
        if controls is None:
            controls = np.tile(np.array([400.0, 0.20], dtype=float), self.cfg.zoh_steps)
        controls = np.asarray(controls, dtype=float).reshape(-1)
        if controls.size != self.control_size:
            raise ValueError("Initial control guess has the wrong size")
        return controls

    def _reconstruct_multipliers(self, controls: np.ndarray, p: np.ndarray,
                                 active_tolerance: float = 2e-4) -> tuple[np.ndarray, np.ndarray, float, int]:
        """NNLS reconstruction of discrete path and bound KKT multipliers.

        This follows the previous CSTR treatment: the primal SLSQP result is
        retained and nonnegative multipliers are reconstructed only for active
        inequalities.  They are labels for the transcribed NLP, not
        continuous-time multiplier functions.
        """
        gradient = np.asarray(self.gradient(controls, p), float).reshape(-1)
        g = np.asarray(self.constraints(controls, p), float).reshape(-1)
        jacobian = np.asarray(self.jacobian(controls, p), float)
        path_active = np.flatnonzero(g >= -active_tolerance)
        lower_active = np.flatnonzero(controls - self.lower <= active_tolerance)
        upper_active = np.flatnonzero(self.upper - controls <= active_tolerance)
        blocks: list[np.ndarray] = []
        if path_active.size:
            blocks.append(jacobian[path_active].T)
        if lower_active.size:
            blocks.append(-np.eye(self.control_size)[:, lower_active])
        if upper_active.size:
            blocks.append(np.eye(self.control_size)[:, upper_active])
        path_duals = np.zeros(self.path_count)
        bound_duals = np.zeros((2, self.control_size))
        if not blocks:
            return path_duals, bound_duals, float(np.linalg.norm(gradient)), 0
        design = np.column_stack(blocks)
        scale = np.maximum(np.linalg.norm(design, axis=0), 1e-12)
        coefficients, _ = nnls(design / scale, -gradient, maxiter=10000)
        multipliers = coefficients / scale
        offset = 0
        if path_active.size:
            path_duals[path_active] = multipliers[offset: offset + path_active.size]
            offset += path_active.size
        if lower_active.size:
            bound_duals[0, lower_active] = multipliers[offset: offset + lower_active.size]
            offset += lower_active.size
        if upper_active.size:
            bound_duals[1, upper_active] = multipliers[offset: offset + upper_active.size]
        stationarity = gradient + jacobian.T @ path_duals - bound_duals[0] + bound_duals[1]
        return path_duals, bound_duals, float(np.linalg.norm(stationarity)), int(path_active.size)

    def solve(self, initial_state: np.ndarray, controls_guess: np.ndarray | None = None) -> dict[str, Any]:
        p = np.asarray(initial_state, dtype=float).reshape(3)
        objective = lambda u: float(self.objective(u, p))
        gradient = lambda u: np.asarray(self.gradient(u, p), float).reshape(-1)
        constraints = lambda u: np.asarray(self.constraints(u, p), float).reshape(-1)
        jacobian = lambda u: np.asarray(self.jacobian(u, p), float)
        started = time.perf_counter()
        result = minimize(
            objective, self.initial_guess(controls_guess), method="SLSQP", jac=gradient, bounds=self.bounds,
            constraints=[{"type": "ineq", "fun": lambda u: -constraints(u),
                          "jac": lambda u: -jacobian(u)}],
            options={"maxiter": self.cfg.solver_maxiter, "ftol": 1e-9, "disp": False},
        )
        if not result.success:
            raise RuntimeError(result.message)
        controls = np.asarray(result.x, dtype=float).reshape(self.cfg.zoh_steps, 2)
        flat = controls.reshape(-1)
        path_duals, bound_duals, residual, active_count = self._reconstruct_multipliers(flat, p)
        hds_peak = event_located_peak(p, controls, self.cfg)
        return {
            "initial_state": p.tolist(),
            "controls": controls.tolist(),
            "objective": objective(flat),
            "path_duals": path_duals.reshape(-1, 2).tolist(),
            "bound_duals": bound_duals.tolist(),
            "kkt_stationarity_norm": residual,
            "active_discretized_path_constraints": active_count,
            "discretized_path_max_g": float(np.max(constraints(flat))),
            "event_located_max_g": hds_peak,
            "solve_seconds": time.perf_counter() - started,
            "dual_source": "active-constraint NNLS reconstruction of the reduced RK4 transcription",
        }


def _segment_peak(state: np.ndarray, control: np.ndarray, cfg: EconomouScreenConfig) -> tuple[np.ndarray, np.ndarray]:
    duration = cfg.zoh_duration_s

    def ca_event(time: float, x: np.ndarray) -> float:
        return float(economou_ode(time, x, control, cfg)[0])

    def temp_event(time: float, x: np.ndarray) -> float:
        return float(economou_ode(time, x, control, cfg)[2])

    ca_event.direction = temp_event.direction = -1
    ca_event.terminal = temp_event.terminal = False
    solution = solve_ivp(
        lambda t, x: economou_ode(t, x, control, cfg), (0.0, duration), state,
        dense_output=True, events=(ca_event, temp_event), method="DOP853",
        rtol=1e-10, atol=1e-12, max_step=duration / 20,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    endpoints = np.array([0.0, duration])
    ca_times = np.unique(np.r_[endpoints, solution.t_events[0]])
    temp_times = np.unique(np.r_[endpoints, solution.t_events[1]])
    peaks = np.array([
        np.max(solution.sol(ca_times)[0] - cfg.ca_max),
        np.max(solution.sol(temp_times)[2] - cfg.temperature_max_K),
    ])
    return peaks, solution.y[:, -1]


def event_located_peak(initial_state: np.ndarray, controls: np.ndarray, cfg: EconomouScreenConfig) -> float:
    state = np.asarray(initial_state, dtype=float).copy()
    maximum = -np.inf
    for control in controls:
        peaks, state = _segment_peak(state, np.asarray(control, dtype=float), cfg)
        maximum = max(maximum, float(np.max(peaks)))
    return maximum


_WORKER: ReducedEconomouCSTR | None = None


def _worker_init(config: dict[str, Any]) -> None:
    global _WORKER
    _WORKER = ReducedEconomouCSTR(EconomouScreenConfig(**config))


def _worker_solve(payload: tuple[int, list[float]]) -> dict[str, Any]:
    if _WORKER is None:
        raise RuntimeError("Worker was not initialized")
    index, initial_state = payload
    record: dict[str, Any] = {"index": index, "initial_state": initial_state}
    try:
        try:
            record.update(_WORKER.solve(np.asarray(initial_state, dtype=float)))
            record["cold_start_attempt"] = "fixed_[400K,0.20]"
        except RuntimeError as primary_error:
            # This remains an independent cold start: the fallback is a fixed
            # physical sequence, never a neighbouring-label warm start.
            secondary = np.tile(np.array([420.0, 1.0]), _WORKER.cfg.zoh_steps)
            record.update(_WORKER.solve(np.asarray(initial_state, dtype=float), secondary))
            record["cold_start_attempt"] = "fixed_[420K,1.0]_after_primary_failure"
            record["primary_attempt_error"] = str(primary_error)
        # The label is defined by the discrete NLP.  The event-located peak is
        # saved as a diagnostic, not used as an additional acceptance
        # threshold; this keeps direct-transcription label generation separate
        # from the later VALC audit/correction stage.
        record["success"] = True
        record["event_peak_nonpositive"] = bool(record["event_located_max_g"] <= 0.0)
    except Exception as exc:
        record.update({"success": False, "error": f"{type(exc).__name__}: {exc}"})
    return record


def _read_existing(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            # Upgrade checkpoints created by the earlier version, which
            # incorrectly marked a solved discrete NLP as failed whenever the
            # report-only continuous-time diagnostic had a positive peak.
            if (row.get("error") == "Node-feasible reduced NLP failed the event-located continuous-time audit"
                    and "controls" in row and "objective" in row):
                row.pop("error", None)
                row["success"] = True
                row["event_peak_nonpositive"] = bool(row["event_located_max_g"] <= 0.0)
            records[int(row["index"])] = row
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--cell-centers", action="store_true",
                        help="Use centers of an equally spaced grid rather than grid endpoints (useful for an independent test cohort).")
    parser.add_argument("--lhs", action="store_true",
                        help="Use an independent Latin-hypercube design with grid-size**2 points.")
    parser.add_argument("--seed", type=int, default=20260802,
                        help="Random seed for --lhs (stored in summary.json).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--zoh-steps", type=int, default=10)
    parser.add_argument("--substeps-per-zoh", type=int, default=10)
    parser.add_argument("--node-margin", type=float, default=None,
                        help="Optional inward tightening of the discrete C_A and T constraints; use 0 for the original bounds.")
    parser.add_argument("--solver-maxiter", type=int, default=500)
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry recorded failed grid points and compact the checkpoint on completion.")
    parser.add_argument("--ca-min", type=float, default=0.30)
    parser.add_argument("--ca-max", type=float, default=0.50)
    parser.add_argument("--temperature-min", type=float, default=405.0)
    parser.add_argument("--temperature-max", type=float, default=425.0)
    parser.add_argument("--limit", type=int, default=None, help="Solve only the first N grid points.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "economou_cstr_reduced_kkt10_30x30")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.cell_centers and args.lhs:
        raise ValueError("--cell-centers and --lhs select different designs; use only one.")
    if args.ca_min > args.ca_max or args.temperature_min > args.temperature_max:
        raise ValueError("Each declared initial-state interval must have lower <= upper.")
    if args.zoh_steps < 1 or args.substeps_per_zoh < 1:
        raise ValueError("--zoh-steps and --substeps-per-zoh must be positive.")
    cfg = replace(
        EconomouScreenConfig(),
        zoh_steps=args.zoh_steps,
        substeps_per_zoh=args.substeps_per_zoh,
        solver_maxiter=args.solver_maxiter,
        ca_initial_range=(args.ca_min, args.ca_max),
        temperature_initial_range_K=(args.temperature_min, args.temperature_max),
        node_margin=EconomouScreenConfig().node_margin if args.node_margin is None else args.node_margin,
    )
    if args.lhs:
        # The same independently stratified construction used for the VDP
        # test cohort: each marginal interval is sampled exactly once, while
        # the coordinate pairing is randomized.  Thus it uniformly covers
        # the two-dimensional operating domain without being a regular grid.
        count = args.grid_size ** 2
        rng = np.random.default_rng(args.seed)
        unit = np.empty((count, 2), dtype=float)
        for dimension in range(2):
            unit[:, dimension] = (rng.permutation(count) + rng.random(count)) / count
        ca_values = cfg.ca_initial_range[0] + unit[:, 0] * (cfg.ca_initial_range[1] - cfg.ca_initial_range[0])
        temp_values = cfg.temperature_initial_range_K[0] + unit[:, 1] * (cfg.temperature_initial_range_K[1] - cfg.temperature_initial_range_K[0])
        states = np.column_stack((ca_values, 1.0 - ca_values, temp_values))
        design_label = "independent Latin-hypercube sample"
    elif args.cell_centers:
        ca_edges = np.linspace(*cfg.ca_initial_range, args.grid_size + 1)
        temp_edges = np.linspace(*cfg.temperature_initial_range_K, args.grid_size + 1)
        ca_values = 0.5 * (ca_edges[:-1] + ca_edges[1:])
        temp_values = 0.5 * (temp_edges[:-1] + temp_edges[1:])
        states = np.array([[ca, 1.0 - ca, temp] for ca in ca_values for temp in temp_values], float)
        design_label = "cell centers"
    else:
        states = initial_grid(cfg, args.grid_size)
        design_label = "including endpoints"
    if args.limit is not None:
        states = states[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "initial_states.npy", states)
    records_path = args.output / "records.jsonl"
    existing = _read_existing(records_path)
    pending = [
        (i, state.tolist()) for i, state in enumerate(states)
        if i not in existing or (args.retry_failed and not existing[i].get("success", False))
    ]
    context = mp.get_context("spawn")
    with records_path.open("a", encoding="utf-8") as handle:
        with context.Pool(args.workers, initializer=_worker_init, initargs=(asdict(cfg),)) as pool:
            for record in pool.imap_unordered(_worker_solve, pending, chunksize=1):
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                existing[int(record["index"])] = record
                print(f"{record['index'] + 1}/{len(states)} success={record['success']}", flush=True)
    ordered = [existing[i] for i in range(len(states))]
    # A retry replaces the earlier failed row instead of leaving duplicate
    # indices that would make downstream label loading ambiguous.
    records_path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in ordered), encoding="utf-8"
    )
    successful = [row for row in ordered if row.get("success")]
    summary = {
        "purpose": "Independent cold-start reduced-space RK4 KKT labels for the Economou CSTR.",
        "config": asdict(cfg),
        "grid_size": args.grid_size,
        "initial_state_layout": design_label,
        "lhs_seed": args.seed if args.lhs else None,
        "labels_requested": len(states),
        "labels_successful": len(successful),
        "labels_failed": len(states) - len(successful),
        "event_peak_nonpositive_count": int(sum(bool(row.get("event_peak_nonpositive", row.get("event_located_max_g", np.inf) <= 0.0)) for row in successful)),
        "mean_solve_seconds_successful": float(np.mean([row["solve_seconds"] for row in successful])) if successful else None,
        "multiplier_interpretation": "Finite-dimensional multipliers of the 10-substep reduced RK4 transcription only; not continuous-time multiplier functions.",
        "cold_start_protocol": "Every initial condition first uses the fixed [T_i,F]=[400 K,0.20] sequence. A failed primary solve may retry the fixed [420 K,1.0] sequence; no neighbouring label is ever used as a warm start.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
