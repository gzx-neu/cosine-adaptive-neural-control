"""Reduced-space VDP CVP label generator.

Only the ZOH controls are NLP variables.  All states are eliminated by the
declared RK4 forward rollout, so this is a control-vector (CVP) NLP rather
than the older simultaneous state/control transcription.  Path multipliers
belong only to this finite-dimensional reduced transcription.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import BFGS, Bounds, NonlinearConstraint, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kkt_collocation.generate_vdp_kkt_data import import_casadi  # noqa: E402


@dataclass(frozen=True)
class Config:
    horizon: float = 5.0
    zoh_steps: int = 20
    substeps_per_zoh: int = 10
    y1_min: float = -0.4
    u_min: float = -0.3
    u_max: float = 1.0
    collocation_safety_margin: float = 1e-3
    solver_maxiter: int = 2500
    gtol: float = 1e-7
    rtol: float = 1e-10
    atol: float = 1e-12

    @property
    def dt(self) -> float:
        return self.horizon / (self.zoh_steps * self.substeps_per_zoh)

    @property
    def zoh_duration(self) -> float:
        return self.horizon / self.zoh_steps

    @property
    def node_count(self) -> int:
        return self.zoh_steps * self.substeps_per_zoh + 1


class ReducedVDPCVP:
    def __init__(self, cfg: Config, audit_references: bool = False) -> None:
        self.cfg, self.audit_references, self.ca = cfg, audit_references, import_casadi()
        self._build()

    def _dynamics(self, x: Any, u: Any) -> Any:
        return self.ca.vertcat((1.0 - x[1] ** 2) * x[0] - x[1] + u, x[0], x[0] ** 2 + x[1] ** 2 + u ** 2)

    def _rk4(self, x: Any, u: Any) -> Any:
        h = self.cfg.dt
        k1 = self._dynamics(x, u)
        k2 = self._dynamics(x + .5 * h * k1, u)
        k3 = self._dynamics(x + .5 * h * k2, u)
        k4 = self._dynamics(x + h * k3, u)
        return x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def _build(self) -> None:
        ca, cfg = self.ca, self.cfg
        u = ca.SX.sym("u", cfg.zoh_steps)
        p = ca.SX.sym("p", 3)
        x = p
        states = [x]
        for step in range(cfg.zoh_steps * cfg.substeps_per_zoh):
            x = self._rk4(x, u[step // cfg.substeps_per_zoh])
            states.append(x)
        X = ca.horzcat(*states)
        g = cfg.y1_min + cfg.collocation_safety_margin - X[0, :].T
        f = x[2]
        self.f = ca.Function("vdp_cvp_f", [u, p], [f])
        self.grad = ca.Function("vdp_cvp_grad", [u, p], [ca.gradient(f, u)])
        self.g = ca.Function("vdp_cvp_g", [u, p], [g])
        self.jac = ca.Function("vdp_cvp_jac", [u, p], [ca.jacobian(g, u)])

    def solve(self, state: np.ndarray, guess: np.ndarray | None = None) -> dict[str, Any]:
        cfg, p = self.cfg, np.asarray(state, float).reshape(3)
        u0 = np.full(cfg.zoh_steps, .5) if guess is None else np.asarray(guess, float).reshape(cfg.zoh_steps)
        def f(u): return float(self.f(u, p))
        def grad(u): return np.asarray(self.grad(u, p), float).reshape(-1)
        def g(u): return np.asarray(self.g(u, p), float).reshape(-1)
        def jac(u): return np.asarray(self.jac(u, p), float)
        constraint = NonlinearConstraint(g, -np.inf * np.ones(cfg.node_count), np.zeros(cfg.node_count), jac=jac)
        # Each attempt is an independently declared, fixed cold initial
        # control sequence.  In particular, no neighbouring solved point or
        # learned policy is ever used as a warm start.
        guesses = (u0, np.full(cfg.zoh_steps, .2), np.full(cfg.zoh_steps, .8))
        started, result, attempts = time.perf_counter(), None, 0
        for candidate in guesses:
            attempts += 1
            result = minimize(f, candidate, method="trust-constr", jac=grad, hess=BFGS(),
                              bounds=Bounds(cfg.u_min, cfg.u_max), constraints=[constraint],
                              options={"maxiter": cfg.solver_maxiter, "gtol": cfg.gtol, "xtol": 1e-10, "barrier_tol": 1e-10, "verbose": 0})
            if result.success:
                break
        elapsed = time.perf_counter() - started
        if result is None or not result.success:
            raise RuntimeError(str(result.message if result is not None else "no solver attempt"))
        raw_duals = np.asarray(result.v[0], float).reshape(-1)
        controls = np.asarray(result.x, float)
        hds_gmax = audit(state, controls, cfg) if self.audit_references else np.nan
        return {"initial_state": p, "optimal_controls": controls, "objective": float(result.fun),
                "path_duals": np.maximum(raw_duals, 0.0), "raw_path_duals": raw_duals,
                "hds_gmax": hds_gmax, "solve_seconds": elapsed,
                "cold_start_attempts": attempts, "optimality": float(result.optimality),
                "constraint_violation": float(result.constr_violation)}


def audit(state: np.ndarray, controls: np.ndarray, cfg: Config) -> float:
    """Adaptive DOP853 plus stationary-event peak checks; no fixed-grid scan."""
    x, largest = np.asarray(state, float).copy(), cfg.y1_min - float(state[0])
    for u in controls:
        def ode(_t, y):
            return np.array([(1 - y[1] ** 2) * y[0] - y[1] + u, y[0], y[0] ** 2 + y[1] ** 2 + u ** 2])
        def stationary(_t, y): return ode(_t, y)[0]
        stationary.direction, stationary.terminal = 0, False
        sol = solve_ivp(ode, (0., cfg.zoh_duration), x, method="DOP853", rtol=cfg.rtol, atol=cfg.atol,
                        dense_output=True, events=stationary)
        candidates = np.concatenate(([0., cfg.zoh_duration], sol.t_events[0]))
        largest = max(largest, float(np.max(cfg.y1_min - sol.sol(candidates)[0])))
        x = sol.y[:, -1]
    return float(largest)


_WORKER: ReducedVDPCVP | None = None
def _init(cfg: Config, audit_references: bool) -> None:
    global _WORKER
    _WORKER = ReducedVDPCVP(cfg, audit_references)
def _solve(state: np.ndarray) -> dict[str, Any]:
    if _WORKER is None: raise RuntimeError("worker not initialized")
    return _WORKER.solve(state)


def grid_states(size: int) -> np.ndarray:
    return np.asarray([[a, b, 0.] for a in np.linspace(-.1, .1, size) for b in np.linspace(.9, 1.1, size)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states-npy", type=Path)
    ap.add_argument("--grid-size", type=int, default=20)
    ap.add_argument("--max-points", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--audit-references", action="store_true",
                    help="Optional diagnostic only; disabled by default so cold-reference timing is NLP-only.")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists(): raise FileExistsError(f"Refusing to overwrite: {args.output}")
    cfg = Config()
    states = np.asarray(np.load(args.states_npy), float) if args.states_npy else grid_states(args.grid_size)
    if args.max_points is not None: states = states[:args.max_points]
    if states.ndim != 2 or states.shape[1] != 3: raise ValueError("states must have shape (n,3)")
    if args.workers < 1: raise ValueError("workers must be positive")
    if args.workers == 1:
        problem = ReducedVDPCVP(cfg, args.audit_references); records = [problem.solve(x) for x in states]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(cfg, args.audit_references)) as pool:
            records = list(pool.map(_solve, states, chunksize=1))
    if args.audit_references and any(row["hds_gmax"] > 1e-8 for row in records):
        raise RuntimeError("A CVP reference failed the declared continuous-time audit")
    out = {key: np.asarray([row[key] for row in records]) for key in ("initial_state", "optimal_controls", "objective", "path_duals", "hds_gmax", "solve_seconds", "cold_start_attempts", "optimality", "constraint_violation")}
    out["config"], out["description"] = asdict(cfg), "VDP CVP20 reduced-space RK4 transcription; controls are the only NLP variables."
    out["reference_audit_performed"] = bool(args.audit_references)
    out["cold_start_protocol"] = "fixed all-0.5 controls, then fixed all-0.2 and all-0.8 retries only; no neighbour or policy warm start"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f: pickle.dump(out, f)
    print(f"saved {len(records)} CVP{cfg.zoh_steps} reduced-space references; mean solve={np.mean(out['solve_seconds']):.4f}s")


if __name__ == "__main__":
    main()
