"""Generate VDP direct-transcription labels with path-constraint multipliers.

This is an independent data-generation branch.  It does not overwrite any of
the existing successful VDP scripts or lookup tables.  The model, initial-state
domain, ZOH horizon, input bounds, and path constraint match the original VDP
experiments:

    y1(0) in [-0.1, 0.1], y2(0) in [0.9, 1.1], y3(0)=0,
    T=5, N=10 ZOH controls, u in [-0.3, 1.0], y1(t)>=-0.4.

The direct transcription uses RK4 defects on ``substeps_per_zoh`` nodes per
ZOH segment.  SciPy's trust-constr solves the NLP using exact CasADi automatic
derivatives and returns multipliers for every discretized path constraint.
Those multipliers are labels for the *discretized NLP*; an independent HDS-like
event audit verifies the resulting continuous-time path constraint.

Examples
--------
Single-point solver check:
    python kkt_collocation/generate_vdp_kkt_data.py --max-points 1

Complete 20 x 20 data set (can be slow):
    python kkt_collocation/generate_vdp_kkt_data.py --grid-size 20
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import BFGS, NonlinearConstraint, minimize

ROOT = Path(__file__).resolve().parents[1]
_WORKER_PROBLEM: "VDPDirectTranscription | None" = None


def import_casadi():
    """Import the project-local CasADi package without shadowing SciPy's NumPy."""
    package_root = ROOT / "third_party" / "casadi"
    package_dll_dir = package_root / "casadi"
    if package_root.exists() and str(package_root) not in sys.path:
        # Append, rather than insert, so the workspace's compatible NumPy stays
        # ahead of the NumPy wheel bundled in the local target directory.
        sys.path.append(str(package_root))
    if package_dll_dir.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(package_dll_dir))
    try:
        import casadi as ca
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "CasADi is required for symbolic derivatives. Install it under "
            "third_party/casadi before running this generator."
        ) from error
    return ca


@dataclass(frozen=True)
class VDPTranscriptionConfig:
    horizon: float = 5.0
    zoh_steps: int = 10
    substeps_per_zoh: int = 5
    y1_min: float = -0.4
    u_min: float = -0.3
    u_max: float = 1.0
    y1_range: tuple[float, float] = (-0.1, 0.1)
    y2_range: tuple[float, float] = (0.9, 1.1)
    collocation_safety_margin: float = 1e-4
    solver_maxiter: int = 500
    gtol: float = 1e-7
    hds_tolerance: float = 1e-7

    @property
    def total_substeps(self) -> int:
        return self.zoh_steps * self.substeps_per_zoh

    @property
    def node_count(self) -> int:
        return self.total_substeps + 1

    @property
    def substep_duration(self) -> float:
        return self.horizon / self.total_substeps

    @property
    def zoh_duration(self) -> float:
        return self.horizon / self.zoh_steps


class VDPDirectTranscription:
    """CasADi-differentiated direct RK4 transcription solved by trust-constr."""

    def __init__(self, config: VDPTranscriptionConfig) -> None:
        self.config = config
        self.ca = import_casadi()
        self.state_size = 3
        self.control_offset = self.state_size * config.node_count
        self.decision_size = self.control_offset + config.zoh_steps
        self.equality_count = self.state_size * (config.total_substeps + 1)
        self.path_count = config.node_count
        self._build_symbolic_functions()

    def _dynamics(self, x: Any, u: Any) -> Any:
        ca = self.ca
        y1, y2, _ = x[0], x[1], x[2]
        return ca.vertcat(
            (1.0 - y2**2) * y1 - y2 + u,
            y1,
            y1**2 + y2**2 + u**2,
        )

    def _rk4(self, x: Any, u: Any) -> Any:
        h = self.config.substep_duration
        k1 = self._dynamics(x, u)
        k2 = self._dynamics(x + 0.5 * h * k1, u)
        k3 = self._dynamics(x + 0.5 * h * k2, u)
        k4 = self._dynamics(x + h * k3, u)
        return x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def _build_symbolic_functions(self) -> None:
        ca = self.ca
        z = ca.SX.sym("z", self.decision_size)
        initial_state = ca.SX.sym("p", self.state_size)
        X = ca.reshape(z[: self.control_offset], self.state_size, self.config.node_count)
        U = z[self.control_offset :]

        equalities = [X[:, 0] - initial_state]
        for substep in range(self.config.total_substeps):
            control_index = substep // self.config.substeps_per_zoh
            equalities.append(X[:, substep + 1] - self._rk4(X[:, substep], U[control_index]))
        # The small interior margin compensates for extrema located between
        # finite transcription nodes. HDS remains the final acceptance test.
        path_constraints = [
            self.config.y1_min + self.config.collocation_safety_margin - X[0, node]
            for node in range(self.config.node_count)
        ]
        constraints = ca.vertcat(*(equalities + path_constraints))
        objective = X[2, -1]

        self.objective_fun = ca.Function("objective", [z, initial_state], [objective])
        self.objective_grad_fun = ca.Function(
            "objective_grad", [z, initial_state], [ca.gradient(objective, z)]
        )
        self.constraint_fun = ca.Function("constraints", [z, initial_state], [constraints])
        self.constraint_jac_fun = ca.Function(
            "constraints_jac", [z, initial_state], [ca.jacobian(constraints, z)]
        )

        self.constraint_lower = np.concatenate(
            (np.zeros(self.equality_count), np.full(self.path_count, -np.inf))
        )
        self.constraint_upper = np.concatenate(
            (np.zeros(self.equality_count), np.zeros(self.path_count))
        )
        self.lower_bounds = np.full(self.decision_size, -np.inf)
        self.upper_bounds = np.full(self.decision_size, np.inf)
        self.lower_bounds[self.control_offset :] = self.config.u_min
        self.upper_bounds[self.control_offset :] = self.config.u_max

    def _integrate_rk4_nodes(self, initial_state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        nodes = np.empty((self.config.node_count, self.state_size), dtype=float)
        nodes[0] = initial_state
        h = self.config.substep_duration
        state = initial_state.astype(float).copy()
        for substep in range(self.config.total_substeps):
            u = float(controls[substep // self.config.substeps_per_zoh])

            def f(x: np.ndarray) -> np.ndarray:
                y1, y2, _ = x
                return np.array([(1.0 - y2**2) * y1 - y2 + u, y1, y1**2 + y2**2 + u**2])

            k1 = f(state)
            k2 = f(state + 0.5 * h * k1)
            k3 = f(state + 0.5 * h * k2)
            k4 = f(state + h * k3)
            state = state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            nodes[substep + 1] = state
        return nodes

    def initial_guess(self, initial_state: np.ndarray, controls_guess: np.ndarray | None = None) -> np.ndarray:
        # A moderate positive feed/control is a much better feasible guess than
        # zero control for the lower y1 path constraint.
        controls = (np.full(self.config.zoh_steps, 0.5, dtype=float) if controls_guess is None
                    else np.asarray(controls_guess, dtype=float).reshape(self.config.zoh_steps))
        if np.any(controls < self.config.u_min) or np.any(controls > self.config.u_max):
            raise ValueError("warm-start controls violate transcription bounds")
        nodes = self._integrate_rk4_nodes(initial_state, controls)
        return np.concatenate((nodes.T.reshape(-1, order="F"), controls))

    def solve(self, initial_state: np.ndarray, controls_guess: np.ndarray | None = None) -> dict[str, Any]:
        p = np.asarray(initial_state, dtype=float).reshape(self.state_size)

        def objective(z: np.ndarray) -> float:
            return float(self.objective_fun(z, p))

        def objective_grad(z: np.ndarray) -> np.ndarray:
            return np.asarray(self.objective_grad_fun(z, p), dtype=float).reshape(-1)

        def constraints(z: np.ndarray) -> np.ndarray:
            return np.asarray(self.constraint_fun(z, p), dtype=float).reshape(-1)

        def constraints_jac(z: np.ndarray) -> np.ndarray:
            return np.asarray(self.constraint_jac_fun(z, p), dtype=float)

        nonlinear_constraint = NonlinearConstraint(
            constraints, self.constraint_lower, self.constraint_upper, jac=constraints_jac
        )
        start = time.perf_counter()
        result = minimize(
            objective,
            self.initial_guess(p, controls_guess),
            method="trust-constr",
            jac=objective_grad,
            hess=BFGS(),
            bounds=list(zip(self.lower_bounds, self.upper_bounds)),
            constraints=[nonlinear_constraint],
            options={
                "maxiter": self.config.solver_maxiter,
                "gtol": self.config.gtol,
                "xtol": 1e-10,
                "barrier_tol": 1e-10,
                "verbose": 0,
            },
        )
        elapsed = time.perf_counter() - start
        if not result.success:
            raise RuntimeError(f"trust-constr failed: {result.message}")

        z = np.asarray(result.x, dtype=float)
        nodes = z[: self.control_offset].reshape(
            (self.state_size, self.config.node_count), order="F"
        ).T
        controls = z[self.control_offset :].copy()
        if not result.v:
            raise RuntimeError("trust-constr did not return constraint multipliers")
        raw_constraint_duals = np.asarray(result.v[0], dtype=float).reshape(-1)
        # The path constraints are represented as y1_min-y1 <= 0.  Their
        # upper-bound KKT multipliers are nonnegative up to numerical noise.
        path_duals = np.maximum(raw_constraint_duals[self.equality_count :], 0.0)
        hds_max_g, hds_time = hds_max_violation(initial_state, controls, self.config)
        return {
            "initial_state": p,
            "optimal_controls": controls,
            "objective": float(result.fun),
            "collocation_states": nodes,
            "path_duals": path_duals,
            "raw_constraint_duals": raw_constraint_duals,
            "hds_max_g": hds_max_g,
            "hds_peak_time": hds_time,
            "solve_seconds": elapsed,
            "solver_status": int(result.status),
            "solver_message": str(result.message),
            "optimality": float(getattr(result, "optimality", np.nan)),
            "constr_violation": float(getattr(result, "constr_violation", np.nan)),
        }


def hds_max_violation(
    initial_state: np.ndarray, controls: np.ndarray, config: VDPTranscriptionConfig
) -> tuple[float, float]:
    """Event-locate extrema of g=y1_min-y1 over the complete ZOH sequence."""
    state = np.asarray(initial_state, dtype=float).copy()
    largest_g = config.y1_min - state[0]
    peak_time = 0.0
    current_time = 0.0
    for control in controls:
        def ode(_t: float, y: np.ndarray) -> np.ndarray:
            y1, y2, _ = y
            return np.array([(1.0 - y2**2) * y1 - y2 + control, y1, y1**2 + y2**2 + control**2])

        def stationary_event(_t: float, y: np.ndarray) -> float:
            # g_dot=-y1_dot; the zero is shared with y1_dot.
            return ode(_t, y)[0]

        stationary_event.direction = 0
        stationary_event.terminal = False
        solution = solve_ivp(
            ode,
            (0.0, config.zoh_duration),
            state,
            method="DOP853",
            rtol=1e-10,
            atol=1e-12,
            max_step=config.zoh_duration / 200.0,
            dense_output=True,
            events=stationary_event,
        )
        candidate_times = np.concatenate(([0.0, config.zoh_duration], solution.t_events[0]))
        candidate_states = solution.sol(candidate_times)
        candidate_g = config.y1_min - candidate_states[0]
        local_index = int(np.argmax(candidate_g))
        if candidate_g[local_index] > largest_g:
            largest_g = float(candidate_g[local_index])
            peak_time = current_time + float(candidate_times[local_index])
        state = solution.y[:, -1]
        current_time += config.zoh_duration
    return float(largest_g), float(peak_time)


def grid_initial_states(config: VDPTranscriptionConfig, grid_size: int) -> np.ndarray:
    y1 = np.linspace(*config.y1_range, grid_size)
    y2 = np.linspace(*config.y2_range, grid_size)
    return np.asarray([[a, b, 0.0] for a in y1 for b in y2], dtype=float)


def save_dataset(path: Path, records: list[dict[str, Any]], config: VDPTranscriptionConfig) -> None:
    keys = ["initial_state", "optimal_controls", "objective", "path_duals", "hds_max_g", "hds_peak_time", "solve_seconds"]
    dataset = {key: np.asarray([record[key] for record in records]) for key in keys}
    dataset["config"] = asdict(config)
    dataset["description"] = (
        "VDP direct RK4 transcription labels with trust-constr path-constraint duals; "
        "continuous-time HDS audit stored separately."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(dataset, handle)


def _initialise_worker(config: VDPTranscriptionConfig) -> None:
    """Create one CasADi/SciPy problem per spawned process, not per sample."""
    global _WORKER_PROBLEM
    _WORKER_PROBLEM = VDPDirectTranscription(config)


def _solve_worker(task: tuple[np.ndarray, np.ndarray | None]) -> dict[str, Any]:
    if _WORKER_PROBLEM is None:
        raise RuntimeError("VDP transcription worker was not initialized")
    state, controls_guess = task
    return _WORKER_PROBLEM.solve(np.asarray(state, dtype=float), controls_guess)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--substeps-per-zoh", type=int, default=5)
    parser.add_argument("--collocation-safety-margin", type=float, default=1e-4,
                        help="Interior nodal margin used before the independent HDS acceptance audit.")
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent NLP worker processes (default: 1).")
    parser.add_argument("--states-npy", type=Path, default=None,
                        help="Optional (n,3) initial-state array.  It replaces the training grid and is used for solver-labelled test subsets.")
    parser.add_argument("--warm-start-lookup", type=Path,
                        help="Optional legacy lookup containing init_states and optimal_us. If --states-npy is omitted, its state order is used.")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Save completed labels every N points; set 0 only when checkpointing is unwanted.")
    parser.add_argument("--resume", action="store_true", help="Resume from a compatible partial --output file.")
    parser.add_argument("--output", type=Path, default=ROOT / "kkt_collocation" / "data" / "vdp_kkt_grid.pkl")
    args = parser.parse_args()
    config = VDPTranscriptionConfig(substeps_per_zoh=args.substeps_per_zoh,
                                    collocation_safety_margin=args.collocation_safety_margin)
    problem = VDPDirectTranscription(config)
    warm_controls: np.ndarray | None = None
    if args.warm_start_lookup is not None:
        with args.warm_start_lookup.open("rb") as handle:
            warm_lookup = pickle.load(handle)
        warm_states = np.asarray(warm_lookup["init_states"], dtype=float)
        warm_controls = np.asarray(warm_lookup["optimal_us"], dtype=float)
        if warm_states.ndim != 2 or warm_states.shape[1] != 3 or warm_controls.shape != (len(warm_states), config.zoh_steps):
            raise ValueError("--warm-start-lookup must contain (n,3) init_states and (n,10) optimal_us")
        states = warm_states if args.states_npy is None else np.asarray(np.load(args.states_npy), dtype=float)
        # Match arbitrary requested states to legacy controls by their exact
        # Cartesian-grid coordinates, rather than assuming a loop ordering.
        seed_map = {tuple(np.round(state, 12)): control for state, control in zip(warm_states, warm_controls)}
        try:
            warm_controls = np.asarray([seed_map[tuple(np.round(state, 12))] for state in states], dtype=float)
        except KeyError as error:
            raise ValueError("a requested state has no warm-start control in --warm-start-lookup") from error
    else:
        states = grid_initial_states(config, args.grid_size) if args.states_npy is None else np.load(args.states_npy)
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("--states-npy must contain an (n,3) array")
    if args.max_points is not None:
        states = states[: args.max_points]
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    records: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        with args.output.open("rb") as handle:
            partial = pickle.load(handle)
        completed = np.asarray(partial["initial_state"], dtype=float)
        if len(completed) > len(states) or not np.allclose(completed, states[:len(completed)], atol=1e-12):
            raise ValueError("partial output is not a prefix of the requested state sequence")
        keys = ["initial_state", "optimal_controls", "objective", "path_duals", "hds_max_g", "hds_peak_time", "solve_seconds"]
        records = [{key: partial[key][index] for key in keys} for index in range(len(completed))]
        states = states[len(completed):]
        warm_controls = None if warm_controls is None else warm_controls[len(completed):]
        print(f"Resuming from {len(records)} completed labels.")
    solve_iterator = None
    executor = None
    if args.workers == 1:
        guesses = [None] * len(states) if warm_controls is None else warm_controls
        solve_iterator = (problem.solve(state, guess) for state, guess in zip(states, guesses))
    else:
        # ``map`` preserves the prescribed grid order, so a parallel run has
        # identical sample ordering to a serial run.
        executor = ProcessPoolExecutor(
            max_workers=args.workers, initializer=_initialise_worker, initargs=(config,)
        )
        guesses = [None] * len(states) if warm_controls is None else warm_controls
        solve_iterator = executor.map(_solve_worker, zip(states, guesses))
    try:
        total_count = len(records) + len(states)
        for local_index, record in enumerate(solve_iterator, start=1):
            index = len(records) + 1
            if record["hds_max_g"] > config.hds_tolerance:
                raise RuntimeError(
                    f"continuous-time audit failed at point {index}: g_max={record['hds_max_g']:.3e}"
                )
            records.append(record)
            print(
                f"{index}/{total_count} J={record['objective']:.6f} "
                f"g_max={record['hds_max_g']:.2e} max_dual={record['path_duals'].max():.2e} "
                f"time={record['solve_seconds']:.2f}s"
            )
            if args.checkpoint_every and index % args.checkpoint_every == 0:
                save_dataset(args.output, records, config)
                print(f"Checkpoint saved: {index}/{total_count} labels.")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    save_dataset(args.output, records, config)
    print(f"Saved {len(records)} KKT-labelled VDP solutions to {args.output}")


if __name__ == "__main__":
    main()
