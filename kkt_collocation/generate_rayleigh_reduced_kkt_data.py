"""Generate direct-transcription Rayleigh labels on a 2-D initial-state grid.

The Jiang supplementary Rayleigh problem already specifies 20 ZOH controls on
``[0, 4.5]``.  This generator retains those dynamics, objective, bounds, and
path constraint, but extends the published fixed initial state ``(-5,-5)`` to
a declared rectangular training domain.  It uses the same reduced-space RK4
label protocol as the CSTR data generator: controls are the only NLP decision
variables, states are eliminated by forward propagation, and the saved KKT
multipliers are finite-dimensional quantities of this transcription.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, nnls

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RayleighConfig:
    horizon: float = 4.5
    zoh_steps: int = 20
    substeps_per_zoh: int = 10
    control_bounds: tuple[float, float] = (-4.0, 4.0)
    x1_initial_range: tuple[float, float] = (-5.5, -4.5)
    x2_initial_range: tuple[float, float] = (-5.5, -4.5)
    solver_maxiter: int = 700
    # RK10 nodal constraints are tightened slightly so that the independent
    # event-located check also accepts the offline reference sequence.
    node_margin: float = 1.0e-5

    @property
    def zoh_duration(self) -> float:
        return self.horizon / self.zoh_steps

    @property
    def dt(self) -> float:
        return self.zoh_duration / self.substeps_per_zoh


def _casadi():
    package = ROOT / "third_party" / "casadi"
    binary = package / "casadi"
    if str(package) not in sys.path:
        sys.path.append(str(package))
    if binary.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(binary))
    import casadi as ca
    return ca


def rayleigh_ode(_time: float, x: np.ndarray, control: float) -> np.ndarray:
    x1, x2, _cost = map(float, x)
    return np.array((
        x2,
        4.0 * control - x1 - x2 * (7.0 * x2 * x2 / 50.0 - 7.0 / 5.0),
        x1 * x1 + control * control,
    ))


def path_constraint(x: np.ndarray, control: float) -> float:
    return float(control + float(x[0]) / 6.0)


class ReducedRayleigh:
    def __init__(self, cfg: RayleighConfig) -> None:
        self.cfg, self.ca = cfg, _casadi()
        self.path_count = cfg.zoh_steps * (cfg.substeps_per_zoh + 1)
        self._build()

    def _f(self, x: Any, u: Any):
        c = self.ca
        return c.vertcat(
            x[1],
            4.0 * u - x[0] - x[1] * (7.0 * x[1] ** 2 / 50.0 - 7.0 / 5.0),
            x[0] ** 2 + u ** 2,
        )

    def _rk4(self, x: Any, u: Any):
        h = self.cfg.dt
        k1 = self._f(x, u)
        k2 = self._f(x + 0.5 * h * k1, u)
        k3 = self._f(x + 0.5 * h * k2, u)
        k4 = self._f(x + h * k3, u)
        return x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def _build(self) -> None:
        c, cfg = self.ca, self.cfg
        u = c.SX.sym("u", cfg.zoh_steps)
        p = c.SX.sym("p", 2)
        x = c.vertcat(p[0], p[1], 0.0)
        constraints: list[Any] = []
        for k in range(cfg.zoh_steps):
            # This is t_k^+: its inclusion is essential because u jumps at a
            # ZOH switch while x remains continuous.
            constraints.append(u[k] + x[0] / 6.0 + cfg.node_margin)
            for _ in range(cfg.substeps_per_zoh):
                x = self._rk4(x, u[k])
                constraints.append(u[k] + x[0] / 6.0 + cfg.node_margin)
        g = c.vertcat(*constraints)
        self.objective = c.Function("rayleigh_objective", [u, p], [x[2]])
        self.gradient = c.Function("rayleigh_gradient", [u, p], [c.gradient(x[2], u)])
        self.constraints = c.Function("rayleigh_constraints", [u, p], [g])
        self.jacobian = c.Function("rayleigh_jacobian", [u, p], [c.jacobian(g, u)])

    def _reconstruct_multipliers(self, controls: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
        gradient = np.asarray(self.gradient(controls, p), dtype=float).reshape(-1)
        g = np.asarray(self.constraints(controls, p), dtype=float).reshape(-1)
        jac = np.asarray(self.jacobian(controls, p), dtype=float)
        active_tolerance = 2.0e-4
        path_active = np.flatnonzero(g >= -active_tolerance)
        lower, upper = self.cfg.control_bounds
        lower_active = np.flatnonzero(controls - lower <= active_tolerance)
        upper_active = np.flatnonzero(upper - controls <= active_tolerance)
        blocks: list[np.ndarray] = []
        if path_active.size:
            blocks.append(jac[path_active].T)
        if lower_active.size:
            blocks.append(-np.eye(self.cfg.zoh_steps)[:, lower_active])
        if upper_active.size:
            blocks.append(np.eye(self.cfg.zoh_steps)[:, upper_active])
        path_duals = np.zeros(self.path_count)
        bound_duals = np.zeros((2, self.cfg.zoh_steps))
        if not blocks:
            return path_duals, bound_duals, float(np.linalg.norm(gradient)), 0
        design = np.column_stack(blocks)
        scale = np.maximum(np.linalg.norm(design, axis=0), 1.0e-12)
        multipliers, _ = nnls(design / scale, -gradient, maxiter=10000)
        multipliers /= scale
        offset = 0
        if path_active.size:
            path_duals[path_active] = multipliers[offset:offset + path_active.size]
            offset += path_active.size
        if lower_active.size:
            bound_duals[0, lower_active] = multipliers[offset:offset + lower_active.size]
            offset += lower_active.size
        if upper_active.size:
            bound_duals[1, upper_active] = multipliers[offset:offset + upper_active.size]
        stationarity = gradient + jac.T @ path_duals - bound_duals[0] + bound_duals[1]
        return path_duals, bound_duals, float(np.linalg.norm(stationarity)), int(path_active.size)

    def solve(self, p: np.ndarray) -> dict[str, Any]:
        p = np.asarray(p, dtype=float).reshape(2)
        fun = lambda u: float(self.objective(u, p))
        jac = lambda u: np.asarray(self.gradient(u, p), dtype=float).reshape(-1)
        con = lambda u: np.asarray(self.constraints(u, p), dtype=float).reshape(-1)
        cjac = lambda u: np.asarray(self.jacobian(u, p), dtype=float)
        start = time.perf_counter()
        solution = minimize(
            fun, np.zeros(self.cfg.zoh_steps), jac=jac, method="SLSQP",
            bounds=[self.cfg.control_bounds] * self.cfg.zoh_steps,
            constraints={"type": "ineq", "fun": lambda u: -con(u), "jac": lambda u: -cjac(u)},
            options={"maxiter": self.cfg.solver_maxiter, "ftol": 1.0e-10, "disp": False},
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        controls = np.asarray(solution.x, dtype=float)
        duals, bound_duals, stationarity, active_count = self._reconstruct_multipliers(controls, p)
        return {
            "initial_state": [float(p[0]), float(p[1]), 0.0],
            "controls": controls.tolist(),
            "objective": fun(controls),
            "path_duals": duals.tolist(),
            "bound_duals": bound_duals.tolist(),
            "kkt_stationarity_norm": stationarity,
            "active_discretized_path_constraints": active_count,
            "discretized_path_max_g": float(np.max(con(controls) - self.cfg.node_margin)),
            "event_located_max_g": event_located_peak(p, controls, self.cfg),
            "solve_seconds": time.perf_counter() - start,
            "dual_source": "active-constraint NNLS reconstruction of the reduced RK10 transcription",
        }


def segment_peak(state: np.ndarray, control: float, cfg: RayleighConfig) -> tuple[float, np.ndarray]:
    # Within one ZOH segment d/dt(u+x1/6)=x2/6.  Endpoints and all x2=0
    # events therefore contain the exact continuous-time maximum candidates.
    def stationary_event(_time: float, x: np.ndarray) -> float:
        return float(x[1])
    stationary_event.direction = 0
    stationary_event.terminal = False
    solution = solve_ivp(
        lambda t, x: rayleigh_ode(t, x, control), (0.0, cfg.zoh_duration), state,
        method="DOP853", rtol=1.0e-10, atol=1.0e-12, max_step=cfg.zoh_duration / 100.0,
        dense_output=True, events=stationary_event,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    times = np.unique(np.r_[0.0, solution.t_events[0], cfg.zoh_duration])
    peak = max(path_constraint(solution.sol(t), control) for t in times)
    return float(peak), solution.y[:, -1].copy()


def event_located_peak(p: np.ndarray, controls: np.ndarray, cfg: RayleighConfig) -> float:
    state = np.array([float(p[0]), float(p[1]), 0.0])
    peak = -np.inf
    for control in controls:
        local, state = segment_peak(state, float(control), cfg)
        peak = max(peak, local)
    return float(peak)


_WORKER: ReducedRayleigh | None = None


def _worker_init(config: dict[str, Any]) -> None:
    global _WORKER
    _WORKER = ReducedRayleigh(RayleighConfig(**config))


def _worker_solve(payload: tuple[int, list[float]]) -> dict[str, Any]:
    if _WORKER is None:
        raise RuntimeError("Worker is not initialized")
    index, initial = payload
    record: dict[str, Any] = {"index": index, "initial_state_parameter": initial}
    try:
        record.update(_WORKER.solve(np.asarray(initial, dtype=float)))
        record["success"] = bool(record["event_located_max_g"] <= 1.0e-8)
        if not record["success"]:
            record["error"] = "Node-feasible direct NLP failed the event-located continuous-time audit"
    except Exception as exc:
        record.update({"success": False, "error": f"{type(exc).__name__}: {exc}"})
    return record


def _existing(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    return {int(row["index"]): row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--zoh-steps", type=int, default=20)
    parser.add_argument("--substeps-per-zoh", type=int, default=10)
    parser.add_argument("--node-margin", type=float, default=1.0e-5)
    parser.add_argument("--x1-min", type=float, default=-5.5)
    parser.add_argument("--x1-max", type=float, default=-4.5)
    parser.add_argument("--x2-min", type=float, default=-5.5)
    parser.add_argument("--x2-max", type=float, default=-4.5)
    parser.add_argument("--output", type=Path, default=ROOT / "kkt_collocation" / "results" / "rayleigh_reduced_kkt_n20_rk10_30x30")
    args = parser.parse_args()
    if args.grid_size < 2 or args.workers < 1 or args.zoh_steps < 1 or args.substeps_per_zoh < 1:
        raise ValueError("grid size, worker count, ZOH steps, and RK substeps must be positive (grid >= 2).")
    cfg = RayleighConfig(zoh_steps=args.zoh_steps, substeps_per_zoh=args.substeps_per_zoh,
                         node_margin=args.node_margin,
                         x1_initial_range=(args.x1_min, args.x1_max), x2_initial_range=(args.x2_min, args.x2_max))
    x1 = np.linspace(*cfg.x1_initial_range, args.grid_size)
    x2 = np.linspace(*cfg.x2_initial_range, args.grid_size)
    states = np.array([[a, b] for a in x1 for b in x2], dtype=float)
    args.output.mkdir(parents=True, exist_ok=True)
    records_file = args.output / "records.jsonl"
    records = _existing(records_file)
    pending = [(i, state.tolist()) for i, state in enumerate(states) if i not in records]
    with records_file.open("a", encoding="utf-8") as handle:
        with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init, initargs=(asdict(cfg),)) as pool:
            for record in pool.imap_unordered(_worker_solve, pending, chunksize=1):
                records[int(record["index"])] = record
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                print(f"{len(records)}/{len(states)} success={record['success']}", flush=True)
    ordered = [records[i] for i in range(len(states))]
    records_file.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in ordered), encoding="utf-8")
    success = [row for row in ordered if row.get("success")]
    summary = {
        "purpose": "Direct-RK4 reduced-space Rayleigh labels on a two-dimensional initial-condition domain.",
        "source_problem": "Jiang supplementary Rayleigh problem; the initial-condition domain is an extension introduced here.",
        "config": asdict(cfg), "grid_shape": [args.grid_size, args.grid_size],
        "labels_requested": len(states), "labels_successful": len(success), "labels_failed": len(states) - len(success),
        "mean_solve_seconds_successful": float(np.mean([row["solve_seconds"] for row in success])) if success else None,
        "multiplier_interpretation": "Finite-dimensional KKT quantities of the reduced RK10 transcription, not continuous-time multiplier functions.",
        "cold_start_protocol": "Every independent label uses the fixed zero 20-vector specified in the supplied Rayleigh formulation; no neighbour warm starts are used.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
