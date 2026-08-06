"""Cache and render complete three-seed CSTR population trajectories.

Run once per training seed to create a compact trajectory cache, then run
without ``--seed`` to render the complete 1200-trajectory temperature and
heat-removal figures.  All plotted observations are retained.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kkt_collocation.run_cstr_full_simulation import CSTRConfig, Policy, cstr_ode, lhs_states, make_corrector, predict  # noqa: E402

TRAIN_SEEDS = (20260718, 20260725, 20260726)
TEST_SEEDS = (20260724, 20260825, 20260826)
RESULTS = ROOT / "kkt_collocation" / "results" / "cstr_multiseed_test1200"


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def trajectory(state0: np.ndarray, controls: np.ndarray, cfg: CSTRConfig, grid: np.ndarray) -> np.ndarray:
    state = np.asarray(state0, dtype=float).copy(); times: list[float] = []; values: list[float] = []; offset = 0.0
    for control in controls:
        local = np.linspace(0, cfg.zoh_duration, 31)
        sol = solve_ivp(lambda t, x: cstr_ode(t, x, float(control), cfg), (0, cfg.zoh_duration), state,
                        t_eval=local, method="DOP853", rtol=1e-10, atol=1e-12)
        times.extend(offset + local if not times else offset + local[1:])
        values.extend(sol.y[1] if not values else sol.y[1, 1:])
        state = sol.y[:, -1]; offset += cfg.zoh_duration
    return np.interp(grid, np.asarray(times), np.asarray(values))


def cache_seed(seed: int) -> None:
    index = TRAIN_SEEDS.index(seed); test_seed = TEST_SEEDS[index]
    checkpoint = load_checkpoint(RESULTS / f"cstr_seed{seed}.pth")
    cfg = CSTRConfig(**checkpoint["config"])
    model = Policy(cfg); model.load_state_dict(checkpoint["model"])
    states = lhs_states(cfg, 400, test_seed)
    nominal, _ = predict(model, np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"]), states)
    with (RESULTS / "per_sample.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["training_seed"]) == seed]
    if len(rows) != 400:
        raise RuntimeError(f"expected 400 stored rows for seed {seed}, found {len(rows)}")
    corrected = nominal.copy(); corrector = make_corrector(cfg)
    time_grid = np.linspace(0, cfg.horizon, 301); raw_temp = np.empty((400, len(time_grid))); applied_temp = np.empty_like(raw_temp)
    for i, (state, raw_control, row) in enumerate(zip(states, nominal, rows)):
        raw_temp[i] = trajectory(state, raw_control, cfg, time_grid)
        if float(row["nominal_hds_max_g"]) > cfg.hds_tolerance:
            outcome = corrector.correct(state, raw_control, cfg.zoh_duration)
            if not outcome.accepted:
                raise RuntimeError(f"plot cache correction rejected at seed {seed}, sample {i}")
            corrected[i] = outcome.controls
            applied_temp[i] = trajectory(state, outcome.controls, cfg, time_grid)
        else:
            applied_temp[i] = raw_temp[i]
        if (i + 1) % 50 == 0:
            print(f"seed {seed}: cached {i + 1}/400", flush=True)
    np.savez_compressed(RESULTS / f"population_seed{seed}.npz", time=time_grid, raw_temperature=raw_temp,
                        applied_temperature=applied_temp, raw_controls=nominal, applied_controls=corrected,
                        raw_violation=np.asarray([float(row["nominal_hds_max_g"]) > cfg.hds_tolerance for row in rows]))


def render() -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"font.family": "Arial", "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.right": False, "axes.spines.top": False})
    caches = [np.load(RESULTS / f"population_seed{seed}.npz") for seed in TRAIN_SEEDS]
    time_grid = caches[0]["time"]; raw_temp = np.vstack([c["raw_temperature"] for c in caches]); applied_temp = np.vstack([c["applied_temperature"] for c in caches])
    raw_controls = np.vstack([c["raw_controls"] for c in caches]); applied_controls = np.vstack([c["applied_controls"] for c in caches]); violations = np.concatenate([c["raw_violation"] for c in caches]).astype(bool)
    cfg = CSTRConfig()
    figures = ROOT / "论文写作" / "figures"; figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3), sharey=True)
    for values, ax, color, title in ((raw_temp, axes[0], "#d9b8b8", "Nominal neural policy"),
                                     (applied_temp, axes[1], "#b8cbe0", "After HDS--$\\lambda$ correction")):
        for curve in values: ax.plot(time_grid, curve, color=color, lw=.38, alpha=.36)
    for index in np.flatnonzero(violations):
        axes[0].plot(time_grid, raw_temp[index], color="#b13a3a", lw=.7, alpha=.62)
        axes[1].plot(time_grid, applied_temp[index], color="#2f6f9f", lw=.7, alpha=.62)
    for ax, title in zip(axes, ("Nominal neural policy", "After HDS--$\\lambda$ correction")):
        ax.axhline(cfg.temperature_max, color="#b13a3a", lw=.9, ls="--"); ax.set(title=title, xlabel="Time (min)", xlim=(0, cfg.horizon)); ax.grid(axis="y", color=".9", lw=.5)
    axes[0].set_ylabel("Reactor temperature $T$ (K)"); fig.tight_layout()
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 600}), (".tiff", {"dpi": 600})):
        fig.savefig(figures / f"cstr_temperature_population{suffix}", bbox_inches="tight", **kwargs)
    fig.savefig(figures / "cstr_temperature_population.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3), sharey=True); step_time = np.linspace(0, cfg.horizon, cfg.zoh_steps + 1)
    for values, ax, color, title in ((raw_controls, axes[0], "#d9b8b8", "Nominal neural policy"),
                                     (applied_controls, axes[1], "#b8cbe0", "After HDS--$\\lambda$ correction")):
        for control in values: ax.step(step_time, np.r_[control, control[-1]], where="post", color=color, lw=.38, alpha=.36)
        ax.axhline(cfg.cooling_min, color=".45", lw=.65, ls=":"); ax.axhline(cfg.cooling_max, color=".45", lw=.65, ls=":")
        ax.set(title=title, xlabel="Time (min)", xlim=(0, cfg.horizon)); ax.grid(axis="y", color=".9", lw=.5)
    axes[0].set_ylabel("Heat-removal rate $q_c$ (K min$^{-1}$)"); fig.tight_layout()
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 600}), (".tiff", {"dpi": 600})):
        fig.savefig(figures / f"cstr_controls_population{suffix}", bbox_inches="tight", **kwargs)
    fig.savefig(figures / "cstr_controls_population.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=TRAIN_SEEDS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    RESULTS = args.results
    if args.seed is None:
        render()
    else:
        cache_seed(args.seed)


if __name__ == "__main__":
    main()
