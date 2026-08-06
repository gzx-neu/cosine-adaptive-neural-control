"""Small, isolated Rayleigh VALC feasibility pilot.

This script deliberately validates only the learning + pre-execution audit and
lambda-ray correction parts of VALC.  It keeps the Rayleigh dynamics,
objective, control bounds, and path constraint from Jiang's supplementary
code, but defines a small two-dimensional initial-condition domain around the
published fixed initial condition.  It is not a replacement for a final,
KKT-labelled benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
# CasADi and PyTorch may load separate OpenMP runtimes on this Windows setup.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    horizon: float = 4.5
    zoh_steps: int = 20
    substeps: int = 20
    control_min: float = -4.0
    control_max: float = 4.0
    x1_bounds: tuple[float, float] = (-5.5, -3.5)
    x2_bounds: tuple[float, float] = (-5.5, -4.5)
    grid_size: int = 5
    test_samples: int = 20
    epochs: int = 300
    seed: int = 20260781
    safety_margin: float = 1e-7

    @property
    def segment_duration(self) -> float:
        return self.horizon / self.zoh_steps

    @property
    def dt(self) -> float:
        return self.segment_duration / self.substeps


def ode(_t: float, x: np.ndarray, u: float) -> np.ndarray:
    x1, x2, _cost = x
    return np.array((x2, 4.0 * u - x1 - x2 * (7.0 * x2 * x2 / 50.0 - 7.0 / 5.0), u * u + x1 * x1))


def g(x: np.ndarray, u: float) -> float:
    return float(u + x[0] / 6.0)


def rk4_rollout(initial: np.ndarray, controls: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Return substep states and the corresponding sampled g values."""
    x = np.asarray(initial, float).copy()
    states = [x.copy()]
    values: list[float] = []
    for u in np.asarray(controls, float):
        values.append(g(x, float(u)))
        for _ in range(cfg.substeps):
            k1 = ode(0.0, x, float(u)); k2 = ode(0.0, x + .5 * cfg.dt * k1, float(u))
            k3 = ode(0.0, x + .5 * cfg.dt * k2, float(u)); k4 = ode(0.0, x + cfg.dt * k3, float(u))
            x = x + cfg.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            states.append(x.copy())
            values.append(g(x, float(u)))
    return np.asarray(states), np.asarray(values)


class RayleighTranscription:
    """Small direct-RK4 NLP with CasADi exact first derivatives."""
    def __init__(self, cfg: Config) -> None:
        import sys
        package = ROOT / "third_party" / "casadi"
        binary = package / "casadi"
        if str(package) not in sys.path:
            sys.path.append(str(package))
        if binary.exists() and hasattr(__import__("os"), "add_dll_directory"):
            __import__("os").add_dll_directory(str(binary))
        import casadi as ca
        self.ca, self.cfg = ca, cfg
        u = ca.SX.sym("u", cfg.zoh_steps)
        p = ca.SX.sym("p", 2)
        x = ca.vertcat(p[0], p[1], 0.0)
        gs = []
        def f(q, v):
            return ca.vertcat(q[1], 4.0 * v - q[0] - q[1] * (7.0 * q[1] ** 2 / 50.0 - 7.0 / 5.0), v ** 2 + q[0] ** 2)
        for k in range(cfg.zoh_steps):
            v = u[k]
            gs.append(v + x[0] / 6.0)
            for _ in range(cfg.substeps):
                k1 = f(x, v); k2 = f(x + .5 * cfg.dt * k1, v)
                k3 = f(x + .5 * cfg.dt * k2, v); k4 = f(x + cfg.dt * k3, v)
                x = x + cfg.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
                gs.append(v + x[0] / 6.0)
        objective = x[2]
        constraints = ca.vertcat(*gs)
        self.objective = ca.Function("rayleigh_objective", [u, p], [objective])
        self.gradient = ca.Function("rayleigh_gradient", [u, p], [ca.gradient(objective, u)])
        self.constraints = ca.Function("rayleigh_constraints", [u, p], [constraints])
        self.jacobian = ca.Function("rayleigh_jacobian", [u, p], [ca.jacobian(constraints, u)])


def solve_label(initial: np.ndarray, cfg: Config, transcription: RayleighTranscription) -> dict:
    """Cold-start direct-RK4 NLP used only for this compact screening pilot."""
    start = time.perf_counter()
    u0 = np.zeros(cfg.zoh_steps)

    def objective(u: np.ndarray) -> float:
        return float(transcription.objective(u, initial[:2]))

    def objective_jac(u: np.ndarray) -> np.ndarray:
        return np.asarray(transcription.gradient(u, initial[:2]), float).reshape(-1)

    def constraints(u: np.ndarray) -> np.ndarray:
        return -np.asarray(transcription.constraints(u, initial[:2]), float).reshape(-1) - cfg.safety_margin

    def constraints_jac(u: np.ndarray) -> np.ndarray:
        return -np.asarray(transcription.jacobian(u, initial[:2]), float)

    result = minimize(
        objective, u0, jac=objective_jac, method="SLSQP", bounds=[(cfg.control_min, cfg.control_max)] * cfg.zoh_steps,
        constraints={"type": "ineq", "fun": constraints, "jac": constraints_jac}, options={"maxiter": 700, "ftol": 1e-10, "disp": False},
    )
    states, values = rk4_rollout(initial, result.x, cfg)
    feasible = bool(values.max() <= 5e-6)
    if not result.success or not feasible:
        raise RuntimeError(f"Rayleigh NLP failed at {initial.tolist()}: success={result.success}, max_g={values.max():.3e}, {result.message}")
    return {"initial": np.asarray(initial, float), "controls": result.x.copy(), "objective": float(states[-1, 2]),
            "max_sampled_g": float(values.max()), "seconds": time.perf_counter() - start}


def segment_peak(state: np.ndarray, control: float, duration: float) -> tuple[float, np.ndarray]:
    # On a ZOH interval, d/dt [u+x1/6] = x2/6.  The event locates all
    # interior extrema; endpoints are evaluated as well.
    def event(_t: float, x: np.ndarray) -> float:
        return float(x[1] / 6.0)
    event.direction = 0
    event.terminal = False
    sol = solve_ivp(lambda t, x: ode(t, x, control), (0.0, duration), state, method="DOP853",
                    rtol=1e-10, atol=1e-12, max_step=duration / 150.0, dense_output=True, events=event)
    times = np.r_[0.0, sol.t_events[0], duration]
    return float(max(g(sol.sol(t), control) for t in times)), sol.y[:, -1].copy()


def audit_and_correct(initial: np.ndarray, nominal: np.ndarray, cfg: Config) -> dict:
    state = np.asarray(initial, float).copy()
    corrected: list[float] = []
    raw_peaks: list[float] = []
    count = 0
    candidates = 0
    for u in np.asarray(nominal, float):
        raw_peak, terminal = segment_peak(state, float(u), cfg.segment_duration)
        raw_peaks.append(raw_peak)
        if raw_peak <= -cfg.safety_margin:
            corrected.append(float(u)); state = terminal; continue
        # Same nonnegative scalar ray used by VALC.  The closest input-feasible
        # scale to one that passes the event-located audit is selected.
        if abs(u) < 1e-14:
            return {"accepted": False, "raw_peak": float(max(raw_peaks)), "controls": None, "corrected_segments": count, "candidates": candidates}
        maximum = max(abs(cfg.control_min / u), abs(cfg.control_max / u), 1.0)
        scales = np.unique(np.r_[np.linspace(0.0, maximum, 31), 1.0])
        scales = scales[(scales * u >= cfg.control_min - 1e-12) & (scales * u <= cfg.control_max + 1e-12)]
        scales = scales[np.argsort(np.abs(scales - 1.0))]
        chosen = None
        for lam in scales:
            if abs(lam - 1.0) < 1e-14:
                continue
            candidates += 1
            peak, next_state = segment_peak(state, float(lam * u), cfg.segment_duration)
            if peak <= -cfg.safety_margin:
                chosen = (float(lam * u), next_state); break
        if chosen is None:
            return {"accepted": False, "raw_peak": float(max(raw_peaks)), "controls": None, "corrected_segments": count, "candidates": candidates}
        control, state = chosen
        corrected.append(control); count += 1
    # Re-audit the complete changed sequence, as a correction affects later states.
    state = np.asarray(initial, float).copy(); peaks = []
    for u in corrected:
        peak, state = segment_peak(state, u, cfg.segment_duration); peaks.append(peak)
    return {"accepted": bool(max(peaks) <= -cfg.safety_margin), "raw_peak": float(max(raw_peaks)),
            "applied_peak": float(max(peaks)), "controls": np.asarray(corrected), "corrected_segments": count, "candidates": candidates}


def high_accuracy_objective(initial: np.ndarray, controls: np.ndarray, cfg: Config) -> float:
    state = np.asarray(initial, float).copy()
    for u in controls:
        sol = solve_ivp(lambda t, x: ode(t, x, float(u)), (0.0, cfg.segment_duration), state, method="DOP853",
                        rtol=1e-11, atol=1e-13, max_step=cfg.segment_duration / 200.0)
        state = sol.y[:, -1]
    return float(state[2])


class Policy(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.value = nn.Linear(128, 1)
        self.control = nn.Linear(128, cfg.zoh_steps)
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.body(x)
        return self.value(z), 4.0 * torch.tanh(self.control(z))


def lhs(cfg: Config, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); out = np.empty((n, 2))
    for j, bounds in enumerate((cfg.x1_bounds, cfg.x2_bounds)):
        out[:, j] = bounds[0] + (rng.permutation(n) + rng.random(n)) / n * (bounds[1] - bounds[0])
    return out


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, default=ROOT / "kkt_collocation" / "results" / "rayleigh_valc_supervised_hds_pilot")
    p.add_argument("--grid-size", type=int, default=5); p.add_argument("--test-samples", type=int, default=20); p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--x1-min", type=float, default=-5.5); p.add_argument("--x1-max", type=float, default=-3.5)
    p.add_argument("--x2-min", type=float, default=-5.5); p.add_argument("--x2-max", type=float, default=-4.5)
    args = p.parse_args(); cfg = Config(grid_size=args.grid_size, test_samples=args.test_samples, epochs=args.epochs,
                                        x1_bounds=(args.x1_min, args.x1_max), x2_bounds=(args.x2_min, args.x2_max)); args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    x1 = np.linspace(*cfg.x1_bounds, cfg.grid_size); x2 = np.linspace(*cfg.x2_bounds, cfg.grid_size)
    train_states = np.asarray([[a, b, 0.0] for a in x1 for b in x2])
    print(f"Generating {len(train_states)} cold-start Rayleigh labels.")
    transcription = RayleighTranscription(cfg)
    labels = [solve_label(x, cfg, transcription) for x in train_states]
    train_u = np.asarray([r["controls"] for r in labels]); train_j = np.asarray([r["objective"] for r in labels])
    x = torch.tensor(train_states[:, :2], dtype=torch.float32); u = torch.tensor(train_u, dtype=torch.float32); j = torch.tensor(train_j[:, None], dtype=torch.float32)
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6); jm, js = j.mean(), j.std(unbiased=False).clamp_min(1e-6)
    model = Policy(cfg); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(cfg.epochs):
        predj, predu = model((x - mean) / std)
        loss = nn.functional.mse_loss(predu, u) + .1 * nn.functional.mse_loss(predj, (j - jm) / js)
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 100 == 0: print(f"epoch={epoch+1}, supervised_loss={loss.item():.4e}")
    test2 = lhs(cfg, cfg.test_samples, cfg.seed + 2); test_states = np.c_[test2, np.zeros(len(test2))]
    print(f"Generating {len(test_states)} frozen cold-start references.")
    refs = [solve_label(x0, cfg, transcription) for x0 in test_states]
    with torch.no_grad():
        _predj, nominal = model((torch.tensor(test2, dtype=torch.float32) - mean) / std)
    rows = []
    for index, (x0, ref, uhat) in enumerate(zip(test_states, refs, nominal.numpy())):
        raw_j = high_accuracy_objective(x0, uhat, cfg); start = time.perf_counter(); outcome = audit_and_correct(x0, uhat, cfg); audit_s = time.perf_counter() - start
        if outcome["accepted"]:
            final_j = high_accuracy_objective(x0, outcome["controls"], cfg); gap = 100.0 * (final_j - ref["objective"]) / max(abs(ref["objective"]), 1e-12)
        else:
            final_j = np.nan; gap = np.nan
        rows.append({"sample_index": index, "x1_0": x0[0], "x2_0": x0[1], "reference_objective": ref["objective"],
                     "nominal_objective": raw_j, "hds_objective": final_j, "hds_relative_gap_percent": gap,
                     "raw_hds_max_g": outcome["raw_peak"], "hds_max_g": outcome.get("applied_peak", np.nan),
                     "accepted": outcome["accepted"], "fallback": not outcome["accepted"], "corrected_segments": outcome["corrected_segments"],
                     "audit_seconds": audit_s, "reference_seconds": ref["seconds"]})
    with (args.output / "per_sample.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    accepted = np.asarray([r["accepted"] for r in rows], bool); gaps = np.asarray([r["hds_relative_gap_percent"] for r in rows], float)
    summary = {"purpose": "screening pilot: supervised policy plus event-located audit/lambda correction; no KKT continuation",
               "config": asdict(cfg), "labels": len(labels), "label_mean_seconds": float(np.mean([r["seconds"] for r in labels])),
               "test": {"samples": len(rows), "accepted_rate_percent": float(100 * accepted.mean()), "fallback_rate_percent": float(100 * (1 - accepted.mean())),
                        "nominal_violation_rate_percent": float(100 * np.mean([r["raw_hds_max_g"] > 0 for r in rows])),
                        "mean_corrected_segments": float(np.mean([r["corrected_segments"] for r in rows])),
                        "mean_hds_relative_gap_percent": float(np.nanmean(gaps)), "std_hds_relative_gap_percent": float(np.nanstd(gaps, ddof=1)),
                        "mean_audit_seconds": float(np.mean([r["audit_seconds"] for r in rows])), "mean_reference_seconds": float(np.mean([r["reference_seconds"] for r in rows]))}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "normalization": {"mean": mean, "std": std, "jm": jm, "js": js}, "config": asdict(cfg)}, args.output / "policy.pth")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
