"""Fast single-seed VALC screen on the 900 Economou-CSTR direct-NLP labels.

This preliminary driver is intentionally limited to supervised policy-value
training plus the deployment HDS--lambda stage.  KKT continuation is added
only after finite-dimensional multiplier labels have been reconstructed from
the direct transcription; it is not silently approximated here.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kkt_collocation.screen_economou_cstr_30x30 import EconomouScreenConfig, economou_ode, stage_cost


class PolicyValue(nn.Module):
    def __init__(self, cfg: EconomouScreenConfig) -> None:
        super().__init__(); self.cfg = cfg
        self.body = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.value = nn.Linear(128, 1)
        self.control = nn.Linear(128, 2 * cfg.zoh_steps)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.body(inputs)
        normalized = torch.sigmoid(self.control(latent)).view(-1, self.cfg.zoh_steps, 2)
        low = torch.tensor([self.cfg.ti_bounds_K[0], self.cfg.flow_bounds[0]], dtype=inputs.dtype, device=inputs.device)
        span = torch.tensor([self.cfg.ti_bounds_K[1] - self.cfg.ti_bounds_K[0], self.cfg.flow_bounds[1] - self.cfg.flow_bounds[0]], dtype=inputs.dtype, device=inputs.device)
        return self.value(latent), low + normalized * span


def load_labels(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 900 or not all(row["success"] for row in rows):
        raise ValueError("Expected 900 successful direct-transcription labels.")
    rows.sort(key=lambda row: row["index"])
    states = np.asarray([row["initial_state"] for row in rows], float)
    controls = np.asarray([row["controls"] for row in rows], float)
    objectives = np.asarray([row["objective"] for row in rows], float)
    return states, controls, objectives


def lhs_states(cfg: EconomouScreenConfig, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ca = cfg.ca_initial_range[0] + (rng.permutation(n) + rng.random(n)) / n * (cfg.ca_initial_range[1] - cfg.ca_initial_range[0])
    temp = cfg.temperature_initial_range_K[0] + (rng.permutation(n) + rng.random(n)) / n * (cfg.temperature_initial_range_K[1] - cfg.temperature_initial_range_K[0])
    return np.c_[ca, 1.0 - ca, temp]


def segment_audit(state: np.ndarray, control: np.ndarray, cfg: EconomouScreenConfig) -> tuple[np.ndarray, np.ndarray]:
    """Locate numerical peaks from DOP853 dense output on one ZOH segment."""
    duration = cfg.zoh_duration_s
    sol = solve_ivp(lambda t, x: economou_ode(t, x, control, cfg), (0.0, duration), state,
                    dense_output=True, method="DOP853", rtol=1e-10, atol=1e-12, max_step=duration / 160)
    if not sol.success:
        raise RuntimeError(sol.message)
    def g(time: float) -> np.ndarray:
        x = sol.sol(time)
        return np.array([x[0] - cfg.ca_max, x[2] - cfg.temperature_max_K])
    mesh = np.linspace(0.0, duration, 41); values = np.asarray([g(t) for t in mesh])
    peaks = values.max(axis=0)
    # Every local sampled maximum is refined with the solver's dense output.
    for constraint in range(2):
        candidates = [0, len(mesh) - 1] + [j for j in range(1, len(mesh) - 1)
                 if values[j, constraint] >= values[j - 1, constraint] and values[j, constraint] >= values[j + 1, constraint]]
        for j in candidates:
            lo, hi = mesh[max(0, j - 1)], mesh[min(len(mesh) - 1, j + 1)]
            if hi > lo:
                result = minimize_scalar(lambda t: -g(t)[constraint], bounds=(lo, hi), method="bounded",
                                         options={"xatol": 1e-10})
                peaks[constraint] = max(peaks[constraint], -result.fun)
    return peaks, sol.y[:, -1]


def segment_audit_event_located(state: np.ndarray, control: np.ndarray,
                                cfg: EconomouScreenConfig) -> tuple[np.ndarray, np.ndarray]:
    r"""Fast event-located peak audit for this CSTR's two state constraints.

    For ``g_1=C_A-C_{A,\max}`` and ``g_2=T-T_{\max}``, an interior local
    maximum can occur only where ``\dot C_A=0`` or ``\dot T=0``.  DOP853
    locates these descending stationary crossings directly; the endpoints are
    included explicitly.  The original :func:`segment_audit` remains the
    conservative reference implementation used for regression comparisons.
    """
    duration = cfg.zoh_duration_s

    def ca_maximum_event(time: float, x: np.ndarray) -> float:
        return float(economou_ode(time, x, control, cfg)[0])

    def temperature_maximum_event(time: float, x: np.ndarray) -> float:
        return float(economou_ode(time, x, control, cfg)[2])

    # Only a positive-to-negative crossing is a local maximum.  Endpoints are
    # checked separately, so monotone segments require no artificial events.
    ca_maximum_event.direction = -1
    ca_maximum_event.terminal = False
    temperature_maximum_event.direction = -1
    temperature_maximum_event.terminal = False
    sol = solve_ivp(
        lambda t, x: economou_ode(t, x, control, cfg), (0.0, duration), state,
        dense_output=True, events=(ca_maximum_event, temperature_maximum_event),
        method="DOP853", rtol=1e-10, atol=1e-12, max_step=duration / 20,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    endpoint_times = np.array([0.0, duration])
    ca_times = np.unique(np.r_[endpoint_times, sol.t_events[0]])
    temperature_times = np.unique(np.r_[endpoint_times, sol.t_events[1]])
    ca_values = sol.sol(ca_times)[0] - cfg.ca_max
    temperature_values = sol.sol(temperature_times)[2] - cfg.temperature_max_K
    return np.array([float(np.max(ca_values)), float(np.max(temperature_values))]), sol.y[:, -1]


def hds_correct(state0: np.ndarray, controls: np.ndarray, cfg: EconomouScreenConfig, grid_size: int = 31) -> dict:
    """Sequential two-input ray correction in normalized input coordinates."""
    state = np.asarray(state0, float).copy(); output = np.asarray(controls, float).copy()
    low = np.array([cfg.ti_bounds_K[0], cfg.flow_bounds[0]])
    span = np.array([cfg.ti_bounds_K[1] - cfg.ti_bounds_K[0], cfg.flow_bounds[1] - cfg.flow_bounds[0]])
    raw_peak = -np.inf; corrected = 0
    for k in range(cfg.zoh_steps):
        peaks, _ = segment_audit(state, output[k], cfg); raw_peak = max(raw_peak, float(peaks.max()))
        if np.max(peaks) <= 1e-8:
            _, state = segment_audit(state, output[k], cfg); continue
        normalized = np.clip((output[k] - low) / span, 0.0, 1.0)
        positive = normalized > 1e-10
        lambda_max = float(np.min(1.0 / normalized[positive])) if np.any(positive) else 1.0
        candidates = np.unique(np.r_[np.linspace(0.0, lambda_max, grid_size), 1.0])
        candidates = candidates[(candidates >= 0.0) & (candidates <= lambda_max + 1e-12)]
        accepted: list[tuple[float, np.ndarray, np.ndarray]] = []
        for lam in candidates:
            candidate = low + np.clip(lam * normalized, 0.0, 1.0) * span
            candidate_peaks, next_state = segment_audit(state, candidate, cfg)
            if np.max(candidate_peaks) <= 1e-8:
                accepted.append((abs(lam - 1.0), candidate, next_state))
        if not accepted:
            return {"accepted": False, "raw_peak": raw_peak, "final_peak": np.nan, "controls": output, "corrected_segments": corrected}
        _, output[k], state = min(accepted, key=lambda item: item[0]); corrected += 1
    # Re-audit the final sequence from the original initial condition.
    state = np.asarray(state0, float).copy(); final_peak = -np.inf
    for control in output:
        peaks, state = segment_audit(state, control, cfg); final_peak = max(final_peak, float(peaks.max()))
    return {"accepted": final_peak <= 1e-8, "raw_peak": raw_peak, "final_peak": final_peak,
            "controls": output, "corrected_segments": corrected}


def hds_correct_event_located(state0: np.ndarray, controls: np.ndarray,
                              cfg: EconomouScreenConfig, grid_size: int = 31) -> dict:
    """Equivalent discrete lambda-ray policy using event-located peak checks.

    Candidates are evaluated in increasing ``|lambda-1|`` order.  Thus the
    first accepted candidate is exactly the selected member of the *same*
    finite candidate set, without requiring any monotonicity assumption.
    """
    state = np.asarray(state0, float).copy()
    output = np.asarray(controls, float).copy()
    low = np.array([cfg.ti_bounds_K[0], cfg.flow_bounds[0]])
    span = np.array([cfg.ti_bounds_K[1] - cfg.ti_bounds_K[0],
                     cfg.flow_bounds[1] - cfg.flow_bounds[0]])
    raw_peak = -np.inf
    corrected = 0
    for k in range(cfg.zoh_steps):
        peaks, next_state = segment_audit_event_located(state, output[k], cfg)
        raw_peak = max(raw_peak, float(peaks.max()))
        if np.max(peaks) <= 1e-8:
            # Unlike the reference implementation, retain the already
            # propagated state instead of integrating this safe segment twice.
            state = next_state
            continue
        normalized = np.clip((output[k] - low) / span, 0.0, 1.0)
        positive = normalized > 1e-10
        lambda_max = float(np.min(1.0 / normalized[positive])) if np.any(positive) else 1.0
        candidates = np.unique(np.r_[np.linspace(0.0, lambda_max, grid_size), 1.0])
        candidates = candidates[(candidates >= 0.0) & (candidates <= lambda_max + 1e-12)]
        # lambda=1 was just audited and is known to violate; trying closer
        # candidates first preserves the original minimum-intervention rule.
        for lam in sorted(candidates, key=lambda value: abs(float(value) - 1.0)):
            if np.isclose(lam, 1.0, rtol=0.0, atol=1e-12):
                continue
            candidate = low + np.clip(lam * normalized, 0.0, 1.0) * span
            candidate_peaks, candidate_next = segment_audit_event_located(state, candidate, cfg)
            if np.max(candidate_peaks) <= 1e-8:
                output[k] = candidate
                state = candidate_next
                corrected += 1
                break
        else:
            return {"accepted": False, "raw_peak": raw_peak, "final_peak": np.nan,
                    "controls": output, "corrected_segments": corrected}

    state = np.asarray(state0, float).copy()
    final_peak = -np.inf
    for control in output:
        peaks, state = segment_audit_event_located(state, control, cfg)
        final_peak = max(final_peak, float(peaks.max()))
    return {"accepted": final_peak <= 1e-8, "raw_peak": raw_peak, "final_peak": final_peak,
            "controls": output, "corrected_segments": corrected}


def trajectory_objective(state0: np.ndarray, controls: np.ndarray, cfg: EconomouScreenConfig) -> float:
    state = np.asarray(state0, float).copy(); total = 0.0
    for control in controls:
        sol = solve_ivp(lambda t, x: economou_ode(t, x, control, cfg), (0.0, cfg.zoh_duration_s), state,
                        t_eval=np.linspace(0.0, cfg.zoh_duration_s, 101), method="DOP853", rtol=1e-10, atol=1e-12)
        total += np.trapz([stage_cost(sol.y[:, j], control) for j in range(sol.y.shape[1])], sol.t)
        state = sol.y[:, -1]
    return float(total / cfg.horizon_s)


def _audit_worker(payload: tuple[int, list[float], list[list[float]], dict]) -> dict:
    index, state, controls, config = payload
    cfg = EconomouScreenConfig(**config)
    # The event implementation has been regression-checked against the
    # original dense-scan reference on frozen test instances.
    result = hds_correct_event_located(np.asarray(state, float), np.asarray(controls, float), cfg)
    nominal = trajectory_objective(np.asarray(state, float), np.asarray(controls, float), cfg)
    applied = np.nan if not result["accepted"] else trajectory_objective(np.asarray(state, float), result["controls"], cfg)
    return {"index": index, "initial_state": state, "nominal_hds_max_g": result["raw_peak"],
            "applied_hds_max_g": result["final_peak"], "nominal_objective": nominal,
            "applied_objective": applied, "accepted": result["accepted"], "fallback": not result["accepted"],
            "corrected_segments": result["corrected_segments"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=ROOT / "kkt_collocation" / "results" / "economou_cstr_direct_30x30" / "screen_records.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "kkt_collocation" / "results" / "economou_cstr_valc_preliminary")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--test-points", type=int, default=100)
    parser.add_argument("--hds-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260781)
    args = parser.parse_args(); cfg = EconomouScreenConfig(); args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    states, ref_controls, ref_objective = load_labels(args.labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(states[:, [0, 2]], dtype=torch.float32, device=device)
    uref = torch.tensor(ref_controls, dtype=torch.float32, device=device)
    jref = torch.tensor(ref_objective[:, None], dtype=torch.float32, device=device)
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    jmean, jstd = jref.mean(), jref.std(unbiased=False).clamp_min(1e-6)
    policy = PolicyValue(cfg).to(device); optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    low = torch.tensor([cfg.ti_bounds_K[0], cfg.flow_bounds[0]], device=device)
    span = torch.tensor([cfg.ti_bounds_K[1] - cfg.ti_bounds_K[0], cfg.flow_bounds[1] - cfg.flow_bounds[0]], device=device)
    for epoch in range(args.epochs):
        predj, predu = policy((x - mean) / std)
        loss_u = nn.functional.mse_loss((predu - low) / span, (uref - low) / span)
        loss_j = nn.functional.mse_loss(predj, (jref - jmean) / jstd)
        loss = loss_u + 0.1 * loss_j
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); optimizer.step()
        if (epoch + 1) % 50 == 0: print(f"epoch {epoch + 1}/{args.epochs}: loss={loss.item():.6e}", flush=True)
    torch.save({"model": policy.cpu().state_dict(), "state_mean": mean.cpu().numpy(), "state_std": std.cpu().numpy(),
                "objective_mean": float(jmean.cpu()), "objective_std": float(jstd.cpu()), "config": asdict(cfg)}, args.output / "supervised_policy.pth")
    test = lhs_states(cfg, args.test_points, args.seed + 1)
    with torch.no_grad(): _, predicted = policy(torch.tensor((test[:, [0, 2]] - mean.cpu().numpy()) / std.cpu().numpy(), dtype=torch.float32))
    controls = predicted.numpy()
    payload = [(i, test[i].tolist(), controls[i].tolist(), asdict(cfg)) for i in range(len(test))]
    context = mp.get_context("spawn")
    with context.Pool(args.hds_workers) as pool:
        rows = list(pool.imap_unordered(_audit_worker, payload, chunksize=1))
    rows.sort(key=lambda row: row["index"])
    (args.output / "per_sample.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    values = lambda key: np.asarray([row[key] for row in rows], float)
    accepted = values("accepted").astype(bool)
    summary = {"status": "preliminary supervised-plus-HDS screen; KKT continuation not yet included",
               "labels": len(states), "training_epochs": args.epochs, "test_points": len(rows),
               "nominal_violation_rate_percent": float(100 * np.mean(values("nominal_hds_max_g") > 1e-8)),
               "nominal_max_g": float(values("nominal_hds_max_g").max()), "acceptance_rate_percent": float(100 * accepted.mean()),
               "fallback_rate_percent": float(100 * (1 - accepted.mean())),
               "final_max_g": float(np.nanmax(values("applied_hds_max_g"))),
               "mean_corrected_segments": float(values("corrected_segments").mean()),
               "nominal_objective": float(values("nominal_objective").mean()), "applied_objective": float(np.nanmean(values("applied_objective"))),
               "mean_objective_change": float(np.nanmean(values("applied_objective") - values("nominal_objective"))),
               "hds_workers": args.hds_workers}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
