"""Screen a finite-horizon Economou CSTR adaptation on a 30 x 30 initial-state grid.

This is a *screening* study, not a manuscript experiment.  It keeps the
reversible-reaction parameters and nominal residence time from Economou,
Morari, and Palsson (1986), and the economic criterion and state limits used
by the later active-constraint-control adaptation.  The finite horizon,
initial-state rectangle, and bounded implementation range for the inlet
temperature are explicitly declared protocol choices below.

The script only creates/assesses offline labels.  It deliberately does not
train a policy, so an unsuitable operating region is detected before a costly
learning run.
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
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_WORKER_TRANSCRIPTION = None
_WORKER_CONFIG = None


@dataclass(frozen=True)
class EconomouScreenConfig:
    # Economou, Morari, and Palsson (1986), Table I.
    residence_time_s: float = 60.0
    c1_s_inv: float = 5.0e3
    c2_s_inv: float = 1.0e6
    e1_cal_mol: float = 1.0e4
    e2_cal_mol: float = 1.5e4
    gas_constant_cal_mol_K: float = 1.987
    minus_delta_h_cal_mol: float = 5.0e3
    density_kg_L: float = 1.0
    heat_capacity_cal_kg_K: float = 1.0e3
    ca_feed: float = 1.0
    cb_feed: float = 0.0

    # Finite-horizon VALC adaptation: declared, rather than attributed to the
    # steady-state source.  F is normalized by the cited nominal F*=1.
    horizon_s: float = 120.0
    zoh_steps: int = 10
    substeps_per_zoh: int = 3
    ti_bounds_K: tuple[float, float] = (380.0, 450.0)
    flow_bounds: tuple[float, float] = (0.0, 1.0)
    ca_max: float = 0.5
    temperature_max_K: float = 425.0
    node_margin: float = 1.0e-4

    # The requested two-dimensional initially safe screening domain.  C_B(0)
    # is determined by the conserved A+B total used in the original example.
    ca_initial_range: tuple[float, float] = (0.30, 0.50)
    temperature_initial_range_K: tuple[float, float] = (405.0, 425.0)

    solver_maxiter: int = 500
    seed: int = 20260723

    @property
    def dt(self) -> float:
        return self.horizon_s / (self.zoh_steps * self.substeps_per_zoh)

    @property
    def zoh_duration_s(self) -> float:
        return self.horizon_s / self.zoh_steps

    @property
    def node_count(self) -> int:
        return self.zoh_steps * self.substeps_per_zoh + 1


def _casadi():
    package = ROOT / "third_party" / "casadi"
    library = package / "casadi"
    if str(package) not in sys.path:
        sys.path.append(str(package))
    if library.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(library))
    import casadi as ca
    return ca


def reaction_rate(x: np.ndarray, cfg: EconomouScreenConfig) -> float:
    ca, cb, temperature = map(float, x)
    t = max(temperature, 1.0)
    k1 = cfg.c1_s_inv * np.exp(-cfg.e1_cal_mol / (cfg.gas_constant_cal_mol_K * t))
    k2 = cfg.c2_s_inv * np.exp(-cfg.e2_cal_mol / (cfg.gas_constant_cal_mol_K * t))
    return float(k1 * ca - k2 * cb)


def economou_ode(_time: float, x: np.ndarray, u: np.ndarray, cfg: EconomouScreenConfig) -> np.ndarray:
    ca, cb, temperature = map(float, x)
    ti, flow = map(float, u)
    rate = reaction_rate(x, cfg)
    dilution = flow / cfg.residence_time_s
    heat_scale = cfg.minus_delta_h_cal_mol / (cfg.density_kg_L * cfg.heat_capacity_cal_kg_K)
    return np.array([
        dilution * (cfg.ca_feed - ca) - rate,
        dilution * (cfg.cb_feed - cb) + rate,
        dilution * (ti - temperature) + heat_scale * rate,
    ])


def stage_cost(x: np.ndarray, u: np.ndarray) -> float:
    """Economou active-constraint adaptation of the cited steady-state cost."""
    ti, flow = map(float, u)
    return float(-flow - 2.009 * float(x[1]) + (1.657e-3 * ti) ** 2)


class EconomouTranscription:
    def __init__(self, cfg: EconomouScreenConfig) -> None:
        self.cfg = cfg
        self.ca = _casadi()
        self.state_size, self.input_size = 3, 2
        self.control_offset = self.state_size * cfg.node_count
        self.decision_size = self.control_offset + self.input_size * cfg.zoh_steps
        self.equality_count = self.state_size * cfg.node_count
        self._build()

    def _f(self, x: Any, u: Any):
        c, cfg = self.ca, self.cfg
        ca_, cb_, temp = x[0], x[1], x[2]
        ti, flow = u[0], u[1]
        k1 = cfg.c1_s_inv * c.exp(-cfg.e1_cal_mol / (cfg.gas_constant_cal_mol_K * temp))
        k2 = cfg.c2_s_inv * c.exp(-cfg.e2_cal_mol / (cfg.gas_constant_cal_mol_K * temp))
        rate = k1 * ca_ - k2 * cb_
        dilution = flow / cfg.residence_time_s
        heat_scale = cfg.minus_delta_h_cal_mol / (cfg.density_kg_L * cfg.heat_capacity_cal_kg_K)
        return c.vertcat(
            dilution * (cfg.ca_feed - ca_) - rate,
            dilution * (cfg.cb_feed - cb_) + rate,
            dilution * (ti - temp) + heat_scale * rate,
        )

    def _rk4(self, x: Any, u: Any):
        h = self.cfg.dt
        k1 = self._f(x, u); k2 = self._f(x + .5 * h * k1, u)
        k3 = self._f(x + .5 * h * k2, u); k4 = self._f(x + h * k3, u)
        return x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def _build(self) -> None:
        c, cfg = self.ca, self.cfg
        z = c.SX.sym("z", self.decision_size)
        p = c.SX.sym("p", self.state_size)
        X = c.reshape(z[:self.control_offset], self.state_size, cfg.node_count)
        U = c.reshape(z[self.control_offset:], self.input_size, cfg.zoh_steps)
        equalities = [X[:, 0] - p]
        running_cost = 0
        for j in range(cfg.zoh_steps * cfg.substeps_per_zoh):
            u = U[:, j // cfg.substeps_per_zoh]
            equalities.append(X[:, j + 1] - self._rk4(X[:, j], u))
            running_cost += cfg.dt * (-u[1] - 2.009 * X[1, j] + (1.657e-3 * u[0]) ** 2)
        # The initial point is supplied by the declared safety guard.  A small
        # transcription margin applies only after control has had one step to
        # act; this avoids excluding safe boundary initial conditions.
        inequalities = []
        for j in range(1, cfg.node_count):
            inequalities += [X[0, j] - (cfg.ca_max - cfg.node_margin),
                             X[2, j] - (cfg.temperature_max_K - cfg.node_margin)]
        all_constraints = c.vertcat(*(equalities + inequalities))
        self.objective = c.Function("economou_screen_objective", [z, p], [running_cost / cfg.horizon_s])
        self.gradient = c.Function("economou_screen_gradient", [z, p], [c.gradient(running_cost / cfg.horizon_s, z)])
        self.constraints = c.Function("economou_screen_constraints", [z, p], [all_constraints])
        self.jacobian = c.Function("economou_screen_jacobian", [z, p], [c.jacobian(all_constraints, z)])
        self.bounds_lo = np.full(self.decision_size, -np.inf)
        self.bounds_hi = np.full(self.decision_size, np.inf)
        controls = np.arange(self.control_offset, self.decision_size).reshape(cfg.zoh_steps, self.input_size)
        self.bounds_lo[controls[:, 0]] = cfg.ti_bounds_K[0]
        self.bounds_hi[controls[:, 0]] = cfg.ti_bounds_K[1]
        self.bounds_lo[controls[:, 1]] = cfg.flow_bounds[0]
        self.bounds_hi[controls[:, 1]] = cfg.flow_bounds[1]
        self.path_count = len(inequalities)

    def _propagate_rk4(self, p: np.ndarray, controls: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        state = np.asarray(p, float).copy()
        nodes = [state.copy()]
        for j in range(cfg.zoh_steps * cfg.substeps_per_zoh):
            u = controls[j // cfg.substeps_per_zoh]
            h = cfg.dt
            k1 = economou_ode(0, state, u, cfg)
            k2 = economou_ode(0, state + .5 * h * k1, u, cfg)
            k3 = economou_ode(0, state + .5 * h * k2, u, cfg)
            k4 = economou_ode(0, state + h * k3, u, cfg)
            state = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            nodes.append(state.copy())
        return np.asarray(nodes)

    def initial_guess(self, p: np.ndarray, warm: np.ndarray | None = None) -> np.ndarray:
        cfg = self.cfg
        if warm is None:
            # A conservative, physically bounded warm start.  It is not an
            # asserted optimum and merely makes the direct NLP reliable.
            controls = np.tile(np.array([400.0, 0.20]), (cfg.zoh_steps, 1))
        else:
            controls = np.asarray(warm, float).reshape(cfg.zoh_steps, self.input_size).copy()
            controls[:, 0] = np.clip(controls[:, 0], *cfg.ti_bounds_K)
            controls[:, 1] = np.clip(controls[:, 1], *cfg.flow_bounds)
        nodes = self._propagate_rk4(p, controls)
        return np.r_[nodes.T.reshape(-1, order="F"), controls.reshape(-1)]

    def solve(self, p: np.ndarray, warm: np.ndarray | None = None) -> dict[str, Any]:
        point = np.asarray(p, float)
        fun = lambda z: float(self.objective(z, point))
        jac = lambda z: np.asarray(self.gradient(z, point), float).reshape(-1)
        con = lambda z: np.asarray(self.constraints(z, point), float).reshape(-1)
        cjac = lambda z: np.asarray(self.jacobian(z, point), float)
        start = time.perf_counter()
        result = minimize(
            fun, self.initial_guess(point, warm), method="SLSQP", jac=jac,
            bounds=list(zip(self.bounds_lo, self.bounds_hi)),
            constraints=[
                {"type": "eq", "fun": lambda z: con(z)[:self.equality_count],
                 "jac": lambda z: cjac(z)[:self.equality_count]},
                {"type": "ineq", "fun": lambda z: -con(z)[self.equality_count:],
                 "jac": lambda z: -cjac(z)[self.equality_count:]},
            ],
            options={"maxiter": self.cfg.solver_maxiter, "ftol": 1e-9, "disp": False},
        )
        if not result.success:
            raise RuntimeError(result.message)
        controls = np.asarray(result.x[self.control_offset:], float).reshape(self.cfg.zoh_steps, self.input_size)
        return {
            "controls": controls,
            "objective": fun(result.x),
            "solve_seconds": time.perf_counter() - start,
        }


def initial_grid(cfg: EconomouScreenConfig, size: int) -> np.ndarray:
    ca_values = np.linspace(*cfg.ca_initial_range, size)
    t_values = np.linspace(*cfg.temperature_initial_range_K, size)
    return np.array([[ca, 1.0 - ca, temp] for ca in ca_values for temp in t_values], float)


def dense_audit(p: np.ndarray, controls: np.ndarray, cfg: EconomouScreenConfig) -> tuple[float, float]:
    """Dense numerical screen only; the later VALC audit remains event-located."""
    state = np.asarray(p, float).copy()
    max_ca, max_temp = state[0], state[2]
    for control in controls:
        local = np.linspace(0.0, cfg.zoh_duration_s, 101)
        solution = solve_ivp(
            lambda t, x: economou_ode(t, x, control, cfg),
            (0.0, cfg.zoh_duration_s), state, t_eval=local,
            method="DOP853", rtol=1e-10, atol=1e-12,
            max_step=cfg.zoh_duration_s / 150,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        max_ca = max(max_ca, float(np.max(solution.y[0])))
        max_temp = max(max_temp, float(np.max(solution.y[2])))
        state = solution.y[:, -1]
    return max_ca - cfg.ca_max, max_temp - cfg.temperature_max_K


def _worker_initialize(cfg: EconomouScreenConfig) -> None:
    """Construct one CasADi transcription per spawned worker."""
    global _WORKER_TRANSCRIPTION, _WORKER_CONFIG
    _WORKER_CONFIG = cfg
    _WORKER_TRANSCRIPTION = EconomouTranscription(cfg)


def _solve_one(payload: tuple[int, list[float]]) -> dict[str, Any]:
    """Independent cold-start label solve suitable for multiprocessing."""
    global _WORKER_TRANSCRIPTION, _WORKER_CONFIG
    index, point_list = payload
    point = np.asarray(point_list, float)
    record: dict[str, Any] = {"index": index, "initial_state": point.tolist()}
    try:
        solved = _WORKER_TRANSCRIPTION.solve(point, warm=None)
        peak_ca, peak_temp = dense_audit(point, solved["controls"], _WORKER_CONFIG)
        record.update({
            "success": True,
            "objective": solved["objective"],
            "solve_seconds": solved["solve_seconds"],
            "controls": solved["controls"].tolist(),
            "dense_peak_ca_minus_limit": peak_ca,
            "dense_peak_temperature_minus_limit_K": peak_temp,
        })
    except Exception as exc:  # preserve every failure in the checkpoint audit trail
        record.update({"success": False, "error": str(exc)})
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N dry run; preserve resume data.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent cold-start worker processes (use 1 for sequential warm starts).")
    parser.add_argument("--warm-start", action="store_true",
                        help="Reuse the preceding label only in sequential mode; disabled for parallel solves.")
    parser.add_argument("--output", type=Path, default=ROOT / "kkt_collocation" / "results" / "economou_cstr_screen_30x30")
    args = parser.parse_args()
    cfg = EconomouScreenConfig()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    states = initial_grid(cfg, args.grid_size)
    if args.limit is not None:
        states = states[:args.limit]
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.workers > 1 and args.warm_start:
        raise ValueError("Parallel labels are independent cold starts; --warm-start is incompatible with --workers > 1.")
    output_path = out / "screen_records.jsonl"
    completed: list[dict[str, Any]] = []
    if output_path.exists():
        completed = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed_indices = {int(record["index"]) for record in completed}
    if len(completed_indices) != len(completed):
        raise RuntimeError("Checkpoint contains duplicate label indices.")
    if completed_indices and (min(completed_indices) < 0 or max(completed_indices) >= len(states)):
        raise RuntimeError("Existing records exceed the requested screen scope; choose a new output directory.")
    remaining = [(index, p.tolist()) for index, p in enumerate(states) if index not in completed_indices]
    with output_path.open("a", encoding="utf-8") as handle:
        if args.workers == 1:
            transcription = EconomouTranscription(cfg)
            warm: np.ndarray | None = None
            if args.warm_start and completed and completed[-1]["success"]:
                warm = np.asarray(completed[-1]["controls"], float)
            for index, point_list in remaining:
                point = np.asarray(point_list, float)
                record: dict[str, Any] = {"index": index, "initial_state": point.tolist()}
                try:
                    solved = transcription.solve(point, warm if args.warm_start else None)
                    warm = solved["controls"] if args.warm_start else None
                    peak_ca, peak_temp = dense_audit(point, solved["controls"], cfg)
                    record.update({"success": True, "objective": solved["objective"],
                                   "solve_seconds": solved["solve_seconds"], "controls": solved["controls"].tolist(),
                                   "dense_peak_ca_minus_limit": peak_ca,
                                   "dense_peak_temperature_minus_limit_K": peak_temp})
                except Exception as exc:
                    record.update({"success": False, "error": str(exc)}); warm = None
                handle.write(json.dumps(record) + "\n"); handle.flush()
                print(f"{index + 1}/{len(states)} {'ok' if record['success'] else 'failed'}", flush=True)
        else:
            context = mp.get_context("spawn")
            with context.Pool(args.workers, initializer=_worker_initialize, initargs=(cfg,)) as pool:
                for record in pool.imap_unordered(_solve_one, remaining, chunksize=1):
                    handle.write(json.dumps(record) + "\n"); handle.flush()
                    print(f"{record['index'] + 1}/{len(states)} {'ok' if record['success'] else 'failed'}", flush=True)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok = [r for r in rows if r["success"]]
    summary = {
        "purpose": "Pre-training, finite-horizon Economou-CSTR screening; not a manuscript result.",
        "source_parameters": "Economou, Morari, and Palsson (1986), Table I; active-constraint cost/limits from the supplied Example 2 document.",
        "declared_adaptation_choices": {
            "fixed_feed_concentration": cfg.ca_feed,
            "two_initial_coordinates": ["C_A(0)", "T(0)"],
            "C_B(0)": "1-C_A(0)",
            "initial_grid": f"{args.grid_size} x {args.grid_size}",
            "bounded_inlet_temperature_range_K": cfg.ti_bounds_K,
            "horizon_s": cfg.horizon_s,
            "label_solution_protocol": "independent cold-start direct-transcription NLP" if args.workers > 1 else
                ("sequential warm-start direct-transcription NLP" if args.warm_start else "independent cold-start direct-transcription NLP"),
            "worker_processes": args.workers,
        },
        "config": asdict(cfg),
        "requested_instances": len(states),
        "completed_instances": len(rows),
        "successful_instances": len(ok),
        "success_rate_percent": 100.0 * len(ok) / len(rows) if rows else 0.0,
    }
    if ok:
        values = np.asarray([r["objective"] for r in ok], float)
        ca_peaks = np.asarray([r["dense_peak_ca_minus_limit"] for r in ok], float)
        temp_peaks = np.asarray([r["dense_peak_temperature_minus_limit_K"] for r in ok], float)
        summary.update({
            "objective_range": [float(values.min()), float(values.max())],
            "mean_label_solve_seconds": float(np.mean([r["solve_seconds"] for r in ok])),
            "dense_audit_ca_violation_rate_percent": float(100.0 * np.mean(ca_peaks > 1e-8)),
            "dense_audit_temperature_violation_rate_percent": float(100.0 * np.mean(temp_peaks > 1e-8)),
            "worst_dense_ca_peak_minus_limit": float(ca_peaks.max()),
            "worst_dense_temperature_peak_minus_limit_K": float(temp_peaks.max()),
            "near_active_ca_rate_percent": float(100.0 * np.mean(ca_peaks > -1e-3)),
            "near_active_temperature_rate_percent": float(100.0 * np.mean(temp_peaks > -1e-2)),
        })
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
