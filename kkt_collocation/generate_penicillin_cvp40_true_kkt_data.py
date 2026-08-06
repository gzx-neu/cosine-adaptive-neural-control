"""Generate independent CVP40 penicillin labels with active-set KKT-root duals.

The primal problem eliminates states through the RK4 map.  SLSQP obtains a
control-only solution and the active-set KKT equations are then polished with
CasADi exact derivatives.  These are finite-dimensional transcription duals,
not continuous-time multipliers.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, root

import generate_penicillin_kkt_data as pen


class CVP40PenicillinProblem(pen.ReducedPenicillinProblem):
    """CVP40 KKT polish including any active control bounds.

    The historic penicillin data set only needed path multipliers.  At CVP40
    some optima have no active path node and instead use a control bound, so a
    path-only KKT root would be incomplete.  This class retains the original
    active-set root construction but includes both kinds of finite-dimensional
    inequality in stationarity and complementarity.
    """

    def solve(self, x2: float, dual_mode: str = "kkt-root") -> dict:
        if dual_mode != "kkt-root":
            raise ValueError("CVP40 generation only supports KKT-root duals")
        p = np.array([x2], dtype=float)
        started = pen.time.perf_counter()
        # A fixed moderate feed avoids the nonphysical x2<0 branch reached by
        # the historic all-0.8 CVP10 initializer when the control vector has
        # forty degrees of freedom.  It is still a shared cold start.
        guess = np.full(self.config.zoh_steps, 0.4)
        primal = minimize(
            lambda u: float(self.obj(u, p)), guess, method="SLSQP",
            jac=lambda u: np.asarray(self.obj_grad(u, p), dtype=float).ravel(),
            bounds=[(self.config.u_min, self.config.u_max)] * self.config.zoh_steps,
            constraints=[{
                "type": "ineq",
                "fun": lambda u: -np.asarray(self.g(u, p), dtype=float).ravel(),
                "jac": lambda u: -np.asarray(self.g_jac(u, p), dtype=float),
            }],
            options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
        )
        if not primal.success:
            raise RuntimeError(f"SLSQP failed: {primal.message}")
        controls = np.asarray(primal.x, dtype=float)
        path = np.asarray(self.g(controls, p), dtype=float).ravel()
        path_active = np.flatnonzero(path >= -self.config.active_tolerance)
        lower_active = np.flatnonzero(controls - self.config.u_min <= self.config.active_tolerance)
        upper_active = np.flatnonzero(self.config.u_max - controls <= self.config.active_tolerance)

        active: list[tuple[str, int]] = (
            [("path", int(i)) for i in path_active]
            + [("lower", int(i)) for i in lower_active]
            + [("upper", int(i)) for i in upper_active]
        )

        def active_jacobian(u: np.ndarray, active_set: list[tuple[str, int]]) -> np.ndarray:
            path_jac = np.asarray(self.g_jac(u, p), dtype=float)
            result = []
            for kind, index in active_set:
                if kind == "path":
                    result.append(path_jac[index])
                elif kind == "lower":
                    row = np.zeros(self.config.zoh_steps); row[index] = -1.0; result.append(row)
                else:
                    row = np.zeros(self.config.zoh_steps); row[index] = 1.0; result.append(row)
            return np.asarray(result, dtype=float)

        root_result = None
        kkt_residual = np.asarray(self.obj_grad(controls, p), dtype=float).ravel()
        raw_active_dual = np.empty(0, dtype=float)
        for _ in range(10):
            if not active:
                break
            a0 = active_jacobian(controls, active)
            gradient0 = np.asarray(self.obj_grad(controls, p), dtype=float).ravel()
            initial_dual = np.linalg.lstsq(a0.T, -gradient0, rcond=None)[0]

            def equations(z: np.ndarray) -> np.ndarray:
                u, dual = z[:self.config.zoh_steps], z[self.config.zoh_steps:]
                jac = active_jacobian(u, active)
                values = np.asarray(self.g(u, p), dtype=float).ravel()
                constraints = []
                for kind, index in active:
                    constraints.append(values[index] if kind == "path" else
                                       self.config.u_min - u[index] if kind == "lower" else u[index] - self.config.u_max)
                grad = np.asarray(self.obj_grad(u, p), dtype=float).ravel()
                return np.r_[grad + jac.T @ dual, constraints]

            def jacobian(z: np.ndarray) -> np.ndarray:
                u, dual = z[:self.config.zoh_steps], z[self.config.zoh_steps:]
                jac = active_jacobian(u, active)
                path_dual = np.zeros(self.config.node_count)
                for (kind, index), value in zip(active, dual):
                    if kind == "path":
                        path_dual[index] = value
                hessian = (np.asarray(self.obj_hess(u, p), dtype=float)
                           + np.asarray(self.g_lagrangian_hess(u, p, path_dual), dtype=float))
                return np.block([[hessian, jac.T], [jac, np.zeros((len(active), len(active)))]] )

            root_result = root(equations, np.r_[controls, initial_dual], jac=jacobian, method="hybr",
                               options={"xtol": 1e-10, "maxfev": 2000})
            controls = np.asarray(root_result.x[:self.config.zoh_steps], dtype=float)
            raw_active_dual = np.asarray(root_result.x[self.config.zoh_steps:], dtype=float)
            kkt_residual = equations(root_result.x)
            if raw_active_dual.size == 0 or raw_active_dual.min() >= -1e-7:
                break
            active = [item for item, value in zip(active, raw_active_dual) if value >= -1e-7]
        else:
            raise RuntimeError("active-set cleanup did not reach dual feasibility")

        if np.linalg.norm(kkt_residual) > 1e-7:
            message = "unconstrained" if root_result is None else root_result.message
            raise RuntimeError(f"complete active-set KKT root failed: {message}; residual={np.linalg.norm(kkt_residual):.2e}")
        if controls.min() < self.config.u_min - 1e-7 or controls.max() > self.config.u_max + 1e-7:
            raise RuntimeError("KKT root left the declared control bounds")
        peak, peak_time = adaptive_hds_peak(x2, controls, self.config)
        if peak > self.config.hds_tolerance:
            raise RuntimeError(f"KKT-polished node-feasible but HDS-infeasible: {peak:.3e}")
        path_after = np.asarray(self.g(controls, p), dtype=float).ravel()
        if path_after.max() > 1e-7:
            raise RuntimeError(f"KKT root is not node feasible: {path_after.max():.3e}")
        path_duals = np.zeros_like(path_after)
        bound_duals = np.zeros((2, self.config.zoh_steps))
        for (kind, index), value in zip(active, raw_active_dual):
            if kind == "path": path_duals[index] = max(0.0, value)
            elif kind == "lower": bound_duals[0, index] = max(0.0, value)
            else: bound_duals[1, index] = max(0.0, value)
        stationarity = (np.asarray(self.obj_grad(controls, p), dtype=float).ravel()
                        + np.asarray(self.g_jac(controls, p), dtype=float).T @ path_duals
                        - bound_duals[0] + bound_duals[1])
        return {
            "initial_state": np.array([1.0, x2, 0.001, 250.0]), "optimal_controls": controls,
            "objective": float(self.obj(controls, p)), "path_duals": path_duals,
            "raw_path_duals": path_duals.copy(), "bound_duals": bound_duals,
            "hds_max_g": peak, "hds_peak_time": peak_time,
            "solve_seconds": pen.time.perf_counter() - started,
            "dual_source": "complete_active_set_kkt_root_path_and_control_bounds",
            "solver_success": bool(primal.success), "kkt_certificate_accepted": True,
            "optimality": float(np.linalg.norm(kkt_residual)),
            "constr_violation": float(max(0.0, path_after.max())),
            "active_count": int(sum(kind == "path" for kind, _ in active)),
            "active_bound_count": int(sum(kind != "path" for kind, _ in active)),
            "stationarity_norm": float(np.linalg.norm(stationarity)),
        }


_WORKER: CVP40PenicillinProblem | None = None


def adaptive_hds_peak(x2: float, controls: np.ndarray, config: pen.PenicillinConfig) -> tuple[float, float]:
    """DOP853 adaptive integration plus exact stationary-point event checks."""
    state = np.array([1.0, x2, 0.001, 250.0], dtype=float)
    greatest, when, offset = state[1] - config.x2_limit, 0.0, 0.0
    for u in np.asarray(controls, dtype=float):
        def ode(_time: float, x: np.ndarray) -> np.ndarray:
            return pen.PenicillinTranscription._f_numpy(x, float(u))
        def stationary(_time: float, x: np.ndarray) -> float:
            return float(ode(_time, x)[1])
        stationary.direction, stationary.terminal = 0, False
        sol = solve_ivp(ode, (0.0, config.segment_duration), state, method="DOP853",
                        rtol=1e-10, atol=1e-12, dense_output=True, events=stationary)
        if not sol.success:
            raise RuntimeError(sol.message)
        times = np.unique(np.r_[0.0, config.segment_duration, sol.t_events[0]])
        values = sol.sol(times)[1] - config.x2_limit
        local = int(np.argmax(values))
        if values[local] > greatest:
            greatest, when = float(values[local]), offset + float(times[local])
        state = sol.y[:, -1]
        offset += config.segment_duration
    return float(greatest), float(when)


def _init(cfg_dict: dict) -> None:
    global _WORKER
    # The imported generator is left unchanged.  This replacement applies only
    # inside this independent CVP40 worker process.
    pen.hds_peak = adaptive_hds_peak
    cfg = pen.PenicillinConfig(**cfg_dict)
    _WORKER = CVP40PenicillinProblem(cfg, np.empty(0), np.empty((0, cfg.zoh_steps)), use_warm_start=False)


def _solve(x2: float) -> dict:
    if _WORKER is None:
        raise RuntimeError("penicillin worker was not initialized")
    return _WORKER.solve(float(x2), dual_mode="kkt-root")


def _json_row(index: int, row: dict) -> dict:
    return {
        "index": index, "success": True,
        "initial_state": np.asarray(row["initial_state"]).tolist(),
        "controls": np.asarray(row["optimal_controls"]).tolist(),
        "objective": float(row["objective"]), "path_duals": np.asarray(row["path_duals"]).tolist(),
        "bound_duals": np.asarray(row["bound_duals"]).tolist(),
        "event_located_max_g": float(row["hds_max_g"]),
        "event_peak_time": float(row["hds_peak_time"]), "solve_seconds": float(row["solve_seconds"]),
        "kkt_root_residual": float(row["optimality"]),
        "stationarity_norm": float(row["stationarity_norm"]),
        "active_discretized_path_constraints": int(row["active_count"]),
        "active_control_bound_constraints": int(row.get("active_bound_count", 0)),
        "dual_source": str(row["dual_source"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=400)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.points < 1 or args.workers < 1:
        raise ValueError("--points and --workers must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    # Preserve penicillin's original 1e-3 interior transcription margin, with
    # CVP40 and ten RK4 substeps per ZOH segment.
    cfg = replace(pen.PenicillinConfig(), zoh_steps=40, substeps_per_zoh=10,
                  active_tolerance=2e-3)
    x2 = np.linspace(*cfg.x2_range, args.points)
    args.output.mkdir(parents=True)
    np.save(args.output / "initial_x2.npy", x2)
    context = mp.get_context("spawn")
    with context.Pool(args.workers, initializer=_init, initargs=(asdict(cfg),)) as pool:
        rows = list(pool.imap(_solve, x2, chunksize=1))
    data = {key: np.asarray([row[key] for row in rows]) for key in
            ("initial_state", "optimal_controls", "objective", "path_duals", "raw_path_duals",
             "bound_duals", "hds_max_g", "hds_peak_time", "solve_seconds", "dual_source",
             "solver_success", "kkt_certificate_accepted", "optimality", "constr_violation",
             "active_count", "active_bound_count", "stationarity_norm")}
    data["config"] = asdict(cfg)
    data["description"] = (
        "Penicillin CVP40 reduced direct-RK4 NLP labels with active-set KKT-root path multipliers and "
        "adaptive event-located HDS audit. The multipliers belong to the finite-dimensional transcription, "
        "not the continuous-time problem."
    )
    data["initialization"] = "fixed moderate control u=0.4 (independent cold start); no neighbour or policy warm start"
    with (args.output / "labels.pkl").open("wb") as handle:
        pickle.dump(data, handle)
    record_rows = [_json_row(i, row) for i, row in enumerate(rows)]
    (args.output / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in record_rows), encoding="utf-8")
    summary = {
        "purpose": "Independent CVP40 penicillin KKT labels.",
        "labels_requested": len(rows), "labels_successful": len(rows), "labels_failed": 0,
        "config": asdict(cfg), "initial_state_layout": f"{args.points} uniformly spaced x2 values",
        "mean_solve_seconds": float(np.mean(data["solve_seconds"])),
        "max_event_located_g": float(np.max(data["hds_max_g"])),
        "mean_kkt_root_residual": float(np.mean(data["optimality"])),
        "max_kkt_root_residual": float(np.max(data["optimality"])),
        "mean_stationarity_norm": float(np.mean(data["stationarity_norm"])),
        "max_stationarity_norm": float(np.max(data["stationarity_norm"])),
        "multiplier_interpretation": "Finite-dimensional path multipliers recovered by active-set KKT-root polishing of the reduced RK4 transcription; not continuous-time multipliers.",
        "cold_start_protocol": data["initialization"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
