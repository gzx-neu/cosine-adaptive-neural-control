"""Reconstruct and plot all 30-seed linear-cosine constraint trajectories.

The plotted curves are visualization samples from adaptive DOP853 dense
solutions.  Safety decisions and lambda selection remain those of the frozen
event-located HDS implementations.  No fixed plotting grid is used as an
acceptance test.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kkt_collocation"))

from kkt_collocation.economou_cstr_hds_fast import candidates as cstr_candidates  # noqa: E402
from kkt_collocation.run_economou_cstr_supervised_hds import PolicyValue  # noqa: E402
from kkt_collocation.run_penicillin_ablation import (  # noqa: E402
    DT as PEN_DT,
    Policy as PenicillinPolicy,
    UMAX as PEN_UMAX,
    g as pen_g,
    gdot as pen_gdot,
    ode as pen_ode,
)
from kkt_collocation.run_two_stage_vs_kkt_only_ablation import Problem  # noqa: E402
from kkt_collocation.run_unified_su_suj_sk_konly_ablation import UnifiedConfig  # noqa: E402
from kkt_collocation.run_vdp_ablation import (  # noqa: E402
    constraint as vdp_g,
    constraint_derivative as vdp_gdot,
    vdp_ode,
)
from kkt_collocation.screen_economou_cstr_30x30 import (  # noqa: E402
    EconomouScreenConfig,
    economou_ode,
)
from kkt_collocation.train_vdp_kkt_policy import KKTPolicyValueNetwork, TrainConfig  # noqa: E402
from offline_safe_control.adaptive_event_hds import (  # noqa: E402
    AdaptiveEventHDSConfig,
    AdaptiveEventHDSCorrector,
)


SEEDS = tuple(range(20260771, 20260801))
RESULTS = ROOT / "kkt_collocation" / "results"
OUT = RESULTS / "paper30_linear_cosine_constraint_population_20260803_v1"
CSTR_OUT = RESULTS / "paper30_linear_cosine_constraint_population_margin1e6_20260803_v1"
FIGURES = ROOT / "论文写作" / "figures"
METHOD = "S-u+K"


def _cache_root(benchmark: str) -> Path:
    """Use the conservative -1e-6 CSTR cache and the frozen scalar caches."""
    return CSTR_OUT if benchmark == "cstr" else OUT


def _seed_dir(benchmark: str, seed: int) -> Path:
    if benchmark == "vdp":
        root = (
            "multiseed10_vdp_k10_cuda_20260803_v1"
            if seed <= 20260790
            else "multiseed_vdp_k10_cuda_seeds21_30_20260803_v1"
        )
    elif benchmark == "penicillin":
        root = (
            "multiseed10_penicillin_k10_cpu_20260803_v1"
            if seed <= 20260790
            else "multiseed_penicillin_k10_cpu_seeds21_30_20260803_v1"
        )
    elif benchmark == "cstr":
        root = (
            "multiseed10_cstr_k10_cuda_20260803_v1"
            if seed <= 20260790
            else "multiseed_cstr_k10_cuda_seeds21_30_20260803_v1"
        )
    else:
        raise ValueError(benchmark)
    return RESULTS / root / "linear_cosine" / benchmark / f"seed{seed}"


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 400:
        raise ValueError(f"Expected 400 rows in {path}, found {len(rows)}")
    rows.sort(key=lambda row: int(row.get("sample_index", row.get("index", -1))))
    return rows


def _scalar_segment(
    ode, constraint, derivative, state: np.ndarray, control: float,
    duration: float, local_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    def event(_time: float, x: np.ndarray) -> float:
        return float(derivative(x, float(control)))

    event.direction = 0
    event.terminal = False
    solution = solve_ivp(
        lambda t, x: ode(t, x, float(control)),
        (0.0, duration),
        np.asarray(state, float),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
        events=event,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    event_times = np.unique(np.r_[0.0, duration, solution.t_events[0]])
    event_values = np.asarray([constraint(solution.sol(t)) for t in event_times], float)
    plotted = np.asarray([constraint(solution.sol(t)) for t in local_grid], float)
    return solution.y[:, -1].copy(), plotted, float(event_values.max())


def _scalar_trace(benchmark: str, initial: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if benchmark == "vdp":
        ode, constraint, derivative, duration, per_segment = vdp_ode, vdp_g, vdp_gdot, 0.5, 20
    else:
        ode, constraint, derivative, duration, per_segment = pen_ode, pen_g, pen_gdot, PEN_DT, 40
    local = np.linspace(0.0, duration, per_segment + 1)
    state = np.asarray(initial, float).copy()
    values: list[np.ndarray] = []
    peak = -np.inf
    for index, control in enumerate(np.asarray(controls, float)):
        state, segment, segment_peak = _scalar_segment(
            ode, constraint, derivative, state, float(control), duration, local
        )
        values.append(segment if index == 0 else segment[1:])
        peak = max(peak, segment_peak)
    time = np.linspace(0.0, duration * len(controls), per_segment * len(controls) + 1)
    return time, np.concatenate(values), float(peak)


def _cstr_segment(
    state: np.ndarray, control: np.ndarray, cfg: EconomouScreenConfig, local_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def dca(t: float, x: np.ndarray) -> float:
        return float(economou_ode(t, x, control, cfg)[0])

    def dtemp(t: float, x: np.ndarray) -> float:
        return float(economou_ode(t, x, control, cfg)[2])

    dca.direction = dtemp.direction = 0
    dca.terminal = dtemp.terminal = False
    duration = cfg.zoh_duration_s
    solution = solve_ivp(
        lambda t, x: economou_ode(t, x, control, cfg),
        (0.0, duration),
        np.asarray(state, float),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
        events=(dca, dtemp),
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    event_times = np.unique(np.r_[0.0, duration, *solution.t_events])
    event_states = solution.sol(event_times)
    event_peaks = np.array(
        [event_states[0].max() - cfg.ca_max, event_states[2].max() - cfg.temperature_max_K],
        dtype=float,
    )
    plotted_states = solution.sol(local_grid)
    plotted = np.column_stack(
        (plotted_states[0] - cfg.ca_max, plotted_states[2] - cfg.temperature_max_K)
    )
    return solution.y[:, -1].copy(), plotted, event_peaks


def _cstr_nominal_trace(
    initial: np.ndarray, controls: np.ndarray, cfg: EconomouScreenConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_segment = 5
    local = np.linspace(0.0, cfg.zoh_duration_s, per_segment + 1)
    state = np.asarray(initial, float).copy()
    traces: list[np.ndarray] = []
    peaks = np.full(2, -np.inf)
    for index, control in enumerate(np.asarray(controls, float)):
        state, plotted, segment_peaks = _cstr_segment(state, control, cfg, local)
        traces.append(plotted if index == 0 else plotted[1:])
        peaks = np.maximum(peaks, segment_peaks)
    time = np.linspace(0.0, cfg.horizon_s, per_segment * cfg.zoh_steps + 1)
    return time, np.vstack(traces), peaks


def _cstr_corrected_trace(
    initial: np.ndarray, controls: np.ndarray, cfg: EconomouScreenConfig, grid: int = 31
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    per_segment = 5
    local = np.linspace(0.0, cfg.zoh_duration_s, per_segment + 1)
    state = np.asarray(initial, float).copy()
    traces: list[np.ndarray] = []
    peaks = np.full(2, -np.inf)
    corrected = 0
    for index, nominal in enumerate(np.asarray(controls, float)):
        next_state, plotted, segment_peaks = _cstr_segment(state, nominal, cfg, local)
        if segment_peaks.max() <= 1e-8:
            chosen_state, chosen_plot, chosen_peaks = next_state, plotted, segment_peaks
        else:
            low, span, normalized, order = cstr_candidates(nominal, cfg, grid)
            found = None
            for lam in order:
                if np.isclose(lam, 1.0, rtol=0.0, atol=1e-12):
                    continue
                candidate = low + np.clip(lam * normalized, 0.0, 1.0) * span
                candidate_state, candidate_plot, candidate_peaks = _cstr_segment(
                    state, candidate, cfg, local
                )
                if candidate_peaks.max() <= 1e-8:
                    found = candidate_state, candidate_plot, candidate_peaks
                    break
            if found is None:
                raise RuntimeError(f"No safe CSTR lambda candidate at segment {index}")
            chosen_state, chosen_plot, chosen_peaks = found
            corrected += 1
        state = chosen_state
        traces.append(chosen_plot if index == 0 else chosen_plot[1:])
        peaks = np.maximum(peaks, chosen_peaks)
    time = np.linspace(0.0, cfg.horizon_s, per_segment * cfg.zoh_steps + 1)
    return time, np.vstack(traces), peaks, corrected


def _predict_scalar(benchmark: str, directory: Path, states: np.ndarray) -> np.ndarray:
    checkpoint = _load_checkpoint(directory / f"{METHOD}.pth")
    if benchmark == "vdp":
        model = KKTPolicyValueNetwork(TrainConfig())
        features = states[:, :2]
    else:
        model = PenicillinPolicy()
        features = states[:, None]
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mean = np.asarray(checkpoint["normalization"]["mean"], float)
    std = np.asarray(checkpoint["normalization"]["std"], float)
    inputs = torch.tensor((features - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        _, controls = model(inputs)
    return controls.numpy()


def _cstr_protocol() -> tuple[EconomouScreenConfig, np.ndarray]:
    reference = RESULTS / "economou_cstr_reduced_kkt_n100_test400_lhs_margin0"
    raw = json.loads((reference / "summary.json").read_text(encoding="utf-8"))["config"]
    for key in ("ti_bounds_K", "flow_bounds", "ca_initial_range", "temperature_initial_range_K"):
        raw[key] = tuple(raw[key])
    cfg = EconomouScreenConfig(**raw)
    rows = [
        json.loads(line)
        for line in (reference / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda item: int(item["index"]))
    states = np.asarray([row["initial_state"] for row in rows], float)
    if len(states) != 400 or cfg.zoh_steps != 100 or cfg.substeps_per_zoh != 10:
        raise ValueError("Unexpected frozen CSTR test protocol")
    return cfg, states


def _predict_cstr(directory: Path, cfg: EconomouScreenConfig, states: np.ndarray) -> np.ndarray:
    checkpoint = _load_checkpoint(directory / f"{METHOD}.pth")
    model = PolicyValue(cfg).eval()
    model.load_state_dict(checkpoint["model"])
    mean = np.asarray(checkpoint["state_mean"], float)
    std = np.asarray(checkpoint["state_std"], float)
    inputs = torch.tensor((states[:, [0, 2]] - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        _, controls = model(inputs)
    return controls.numpy()


def cache_seed(benchmark: str, seed: int, force: bool = False) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    output = _cache_root(benchmark) / f"{benchmark}_seed{seed}_trajectories.npz"
    if output.exists() and not force:
        return f"{benchmark} seed {seed}: cached"
    directory = _seed_dir(benchmark, seed)
    if not directory.exists():
        raise FileNotFoundError(directory)

    if benchmark in ("vdp", "penicillin"):
        states = np.load(directory / "test_states.npy")
        rows = _read_rows(directory / f"test_per_sample_{METHOD}.csv")
        controls = _predict_scalar(benchmark, directory, states)
        if benchmark == "vdp":
            initials = states
            corrector = AdaptiveEventHDSCorrector(
                vdp_ode, vdp_g, vdp_gdot, (-0.3, 1.0), AdaptiveEventHDSConfig(grid_size=31)
            )
            duration = 0.5
        else:
            initials = np.c_[np.ones(400), states, np.full(400, 0.001), np.full(400, 250.0)]
            corrector = AdaptiveEventHDSCorrector(
                pen_ode, pen_g, pen_gdot, (0.0, PEN_UMAX), AdaptiveEventHDSConfig(grid_size=31)
            )
            duration = PEN_DT
        nominal_paths: list[np.ndarray] = []
        applied_paths: list[np.ndarray] = []
        nominal_peaks: list[float] = []
        applied_peaks: list[float] = []
        corrected_counts: list[int] = []
        time = None
        for index, (initial, nominal, row) in enumerate(zip(initials, controls, rows)):
            time, nominal_path, nominal_peak = _scalar_trace(benchmark, initial, nominal)
            expected = int(float(row["corrected_segments"]))
            if expected:
                outcome = corrector.correct(initial, nominal, duration)
                if not outcome.accepted:
                    raise RuntimeError(f"{benchmark} seed {seed} sample {index}: unexpected fallback")
                applied_controls = np.asarray(outcome.controls, float)
                observed = sum(int(segment.corrected) for segment in outcome.segments)
                _, applied_path, applied_peak = _scalar_trace(benchmark, initial, applied_controls)
            else:
                applied_path, applied_peak, observed = nominal_path.copy(), nominal_peak, 0
            if observed != expected:
                raise ValueError(
                    f"{benchmark} seed {seed} sample {index}: corrected segments {observed} != {expected}"
                )
            stored_nominal = float(row["nominal_hds_max_g"])
            stored_applied = float(row["applied_hds_max_g"])
            if not np.isclose(nominal_peak, stored_nominal, rtol=0.0, atol=5e-7):
                raise ValueError(
                    f"{benchmark} seed {seed} sample {index}: nominal peak mismatch "
                    f"{nominal_peak} vs {stored_nominal}"
                )
            if not np.isclose(applied_peak, stored_applied, rtol=0.0, atol=5e-7):
                raise ValueError(
                    f"{benchmark} seed {seed} sample {index}: applied peak mismatch "
                    f"{applied_peak} vs {stored_applied}"
                )
            nominal_paths.append(nominal_path)
            applied_paths.append(applied_path)
            nominal_peaks.append(nominal_peak)
            applied_peaks.append(applied_peak)
            corrected_counts.append(observed)
        np.savez_compressed(
            output,
            time=np.asarray(time),
            nominal_g=np.asarray(nominal_paths, dtype=np.float32)[:, :, None],
            applied_g=np.asarray(applied_paths, dtype=np.float32)[:, :, None],
            nominal_event_peak=np.asarray(nominal_peaks),
            applied_event_peak=np.asarray(applied_peaks),
            corrected_segments=np.asarray(corrected_counts, dtype=np.int16),
            sample_index=np.arange(400),
            training_seed=np.full(400, seed),
        )
    else:
        cfg, states = _cstr_protocol()
        rows = _read_rows(directory / "hds_test400" / f"test_per_sample_{METHOD}.csv")
        controls = _predict_cstr(directory, cfg, states)
        nominal_paths: list[np.ndarray] = []
        applied_paths: list[np.ndarray] = []
        nominal_peaks: list[np.ndarray] = []
        applied_peaks: list[np.ndarray] = []
        corrected_counts: list[int] = []
        time = None
        for index, (initial, nominal, row) in enumerate(zip(states, controls, rows)):
            time, nominal_path, nominal_peak = _cstr_nominal_trace(initial, nominal, cfg)
            expected = int(float(row["corrected_segments"]))
            if expected:
                _, applied_path, applied_peak, observed = _cstr_corrected_trace(initial, nominal, cfg)
            else:
                applied_path, applied_peak, observed = nominal_path.copy(), nominal_peak.copy(), 0
            if observed != expected:
                raise ValueError(
                    f"cstr seed {seed} sample {index}: corrected segments {observed} != {expected}"
                )
            stored_nominal = float(row["nominal_max_g"])
            stored_applied = float(row["final_max_g"])
            if not np.isclose(nominal_peak.max(), stored_nominal, rtol=0.0, atol=1e-6):
                raise ValueError(
                    f"cstr seed {seed} sample {index}: nominal peak mismatch "
                    f"{nominal_peak.max()} vs {stored_nominal}"
                )
            if not np.isclose(applied_peak.max(), stored_applied, rtol=0.0, atol=1e-6):
                raise ValueError(
                    f"cstr seed {seed} sample {index}: applied peak mismatch "
                    f"{applied_peak.max()} vs {stored_applied}"
                )
            nominal_paths.append(nominal_path)
            applied_paths.append(applied_path)
            nominal_peaks.append(nominal_peak)
            applied_peaks.append(applied_peak)
            corrected_counts.append(observed)
        np.savez_compressed(
            output,
            time=np.asarray(time),
            nominal_g=np.asarray(nominal_paths, dtype=np.float32),
            applied_g=np.asarray(applied_paths, dtype=np.float32),
            nominal_event_peak=np.asarray(nominal_peaks),
            applied_event_peak=np.asarray(applied_peaks),
            corrected_segments=np.asarray(corrected_counts, dtype=np.int16),
            sample_index=np.arange(400),
            training_seed=np.full(400, seed),
            config_json=json.dumps(asdict(cfg)),
        )
    return f"{benchmark} seed {seed}: wrote {output.name}"


def cache_benchmark(benchmark: str, workers: int, force: bool) -> None:
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(cache_seed, benchmark, seed, force): seed for seed in SEEDS}
        for future in as_completed(jobs):
            print(future.result(), flush=True)


def _load_population(benchmark: str) -> dict[str, np.ndarray]:
    caches = [np.load(_cache_root(benchmark) / f"{benchmark}_seed{seed}_trajectories.npz") for seed in SEEDS]
    if any(len(cache["sample_index"]) != 400 for cache in caches):
        raise ValueError(f"{benchmark}: incomplete seed cache")
    time = np.asarray(caches[0]["time"])
    if any(not np.array_equal(time, cache["time"]) for cache in caches[1:]):
        raise ValueError(f"{benchmark}: inconsistent plotting grids")
    return {
        "time": time,
        "nominal_g": np.vstack([cache["nominal_g"] for cache in caches]),
        "applied_g": np.vstack([cache["applied_g"] for cache in caches]),
        "nominal_event_peak": np.vstack(
            [np.atleast_2d(cache["nominal_event_peak"]).reshape(400, -1) for cache in caches]
        ),
        "applied_event_peak": np.vstack(
            [np.atleast_2d(cache["applied_event_peak"]).reshape(400, -1) for cache in caches]
        ),
        "corrected_segments": np.concatenate([cache["corrected_segments"] for cache in caches]),
        "training_seed": np.concatenate([cache["training_seed"] for cache in caches]),
        "sample_index": np.tile(np.arange(400), len(SEEDS)),
    }


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def _panel(
    ax, time: np.ndarray, values: np.ndarray, *, color: str, title: str,
    ylabel: str, panel: str, ylim: tuple[float, float],
    xticks: tuple[float, ...],
) -> None:
    # Every trajectory is retained.  Rasterization keeps vector exports small
    # while axes, labels, boundary, and population maximum remain editable.
    ax.set_rasterization_zorder(2)
    lines = ax.plot(time, values.T, color=color, lw=0.16, alpha=0.0075, zorder=1)
    for line in lines:
        line.set_rasterized(True)
    maximum = np.max(values, axis=0)
    median = np.median(values, axis=0)
    ax.plot(time, maximum, color=color, lw=0.95, zorder=3, label="Population maximum")
    ax.plot(time, median, color="0.18", lw=0.75, zorder=3, label="Population median")
    ax.axhline(0.0, color="0.45", ls=":", lw=0.8, zorder=4, label="Constraint boundary $g=0$")
    ax.set_xlim(float(time[0]), float(time[-1]))
    ax.set_xticks(xticks)
    ax.set_ylim(*ylim)
    ax.set_title(title, pad=3)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.text(-0.18, 1.04, panel, transform=ax.transAxes, fontweight="bold", va="bottom")
    ax.grid(axis="y", color="0.92", lw=0.4, zorder=0)
    ax.set_box_aspect(0.72)
    ax.set_anchor("N")


def render() -> None:
    import matplotlib.pyplot as plt

    _style()
    populations = {name: _load_population(name) for name in ("vdp", "penicillin", "cstr")}
    entries = [
        ("VDP", populations["vdp"], 0, r"$g=-0.4-y_1$", (0, 1, 2, 3, 4, 5)),
        ("Penicillin", populations["penicillin"], 0, r"$g=x_2-0.5$", (0, 10, 20, 30, 40)),
        ("CSTR", populations["cstr"], 0, r"$g_1=C_A-0.5$", (0, 30, 60, 90, 120)),
        ("CSTR", populations["cstr"], 1, r"$g_2=T-425$", (0, 30, 60, 90, 120)),
    ]
    # Match the control-population figure: blue-grey for nominal behavior,
    # muted brick for HDS--lambda changes, charcoal for central tendency.
    colors = {"nominal": "#6E8FA8", "applied": "#C75D44"}
    fig, axes = plt.subplots(2, 4, figsize=(7.25, 3.55), constrained_layout=False)
    for col, (name, data, constraint_index, label, xticks) in enumerate(entries):
        raw = data["nominal_g"][:, :, constraint_index]
        applied = data["applied_g"][:, :, constraint_index]
        lower = float(min(raw.min(), applied.min()))
        upper = float(max(raw.max(), applied.max()))
        span = max(upper - lower, 1e-5)
        ylim = (lower - 0.05 * span, upper + 0.07 * span)
        heading = name if name != "CSTR" else f"CSTR: {label}"
        _panel(
            axes[0, col], data["time"], raw, color=colors["nominal"],
            title=heading, ylabel=label, panel=f"({chr(97 + col)})", ylim=ylim,
            xticks=xticks,
        )
        _panel(
            axes[1, col], data["time"], applied, color=colors["applied"],
            title=heading, ylabel=label, panel=f"({chr(101 + col)})", ylim=ylim,
            xticks=xticks,
        )
    fig.text(.012, .685, "Nominal policy", rotation=90, ha="center", va="center",
             fontsize=7.5, color=colors["nominal"])
    fig.text(.012, .275, "After HDS--$\\lambda$ correction", rotation=90, ha="center", va="center",
             fontsize=7.5, color=colors["applied"])
    fig.suptitle(
        "Complete path-constraint populations\n"
        "30 training seeds $\\times$ 400 matched test points per benchmark",
        y=0.995, fontsize=8.2,
    )
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="0.55", lw=.45, alpha=.55, label="Trajectories ($n=12{,}000$)"),
        Line2D([0], [0], color=colors["nominal"], lw=1.0, label="Nominal max"),
        Line2D([0], [0], color=colors["applied"], lw=1.0, label="Post-HDS max"),
        Line2D([0], [0], color="0.18", lw=.8, label="Median"),
        Line2D([0], [0], color="0.45", ls=":", lw=.8, label="Boundary $g=0$"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(.54, .925),
               frameon=False, ncol=5, columnspacing=.9, handlelength=1.8)
    fig.subplots_adjust(left=0.100, right=0.995, bottom=0.115, top=0.775, wspace=0.44, hspace=0.62)
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "linear_cosine_constraint_population_30seeds_all12000"
    # Render the dense 12,000-trajectory figure once, then copy the identical
    # bytes to the compatibility locations. ``dpi`` also controls the embedded
    # raster layer in the otherwise-vector SVG/PDF exports.
    svg_target = OUT / f"{stem}.svg"
    pdf_target = OUT / f"{stem}.pdf"
    png_target = OUT / f"{stem}.png"
    tiff_target = OUT / f"{stem}.tiff"
    fig.savefig(svg_target, dpi=1200, bbox_inches="tight")
    fig.savefig(pdf_target, dpi=1200, bbox_inches="tight")
    fig.savefig(png_target, dpi=1200, bbox_inches="tight")
    fig.savefig(tiff_target, dpi=1200, bbox_inches="tight")
    for target in (svg_target, pdf_target, png_target, tiff_target):
        shutil.copy2(target, FIGURES / target.name)
        shutil.copy2(target, CSTR_OUT / target.name)
    plt.close(fig)

    summary = {
        "method": "S-u+K with linear cosine projection, 200+10 epochs",
        "training_seeds": list(SEEDS),
        "test_points_per_seed": 400,
        "trajectories_per_benchmark": 12000,
        "sampling_note": (
            "All trajectories are plotted. Curves are sampled from adaptive DOP853 dense output only for "
            "visualization; safety acceptance and lambda selection use segment endpoints plus all located "
            "constraint-derivative stationary events."
        ),
        "hds_statement": (
            "continuous-time numerical audit evidence under the declared model and numerical settings"
        ),
        "benchmarks": {},
    }
    for name, data in populations.items():
        summary["benchmarks"][name] = {
            "rows": int(len(data["training_seed"])),
            "unique_training_seeds": int(len(np.unique(data["training_seed"]))),
            "unique_test_indices_per_seed": 400,
            "nominal_event_peak_max": np.max(data["nominal_event_peak"], axis=0).tolist(),
            "applied_event_peak_max": np.max(data["applied_event_peak"], axis=0).tolist(),
            "mean_corrected_segments": float(np.mean(data["corrected_segments"])),
        }
    (OUT / "constraint_population_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "README.md").write_text(
        "# 30-seed complete constraint-trajectory figure\n\n"
        "The figure retains all 30 x 400 trajectories for the final linear-cosine policy. "
        "The plotting grid is a visualization of adaptive DOP853 dense output and is not used "
        "for safety acceptance. Event-located HDS statistics are stored in each seed cache and "
        "cross-checked against the frozen per-sample evaluation CSV files.\n",
        encoding="utf-8",
    )
    print(FIGURES / f"{stem}.png", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("vdp", "penicillin", "cstr", "all"), default="all")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if not args.render_only:
        selected = ("vdp", "penicillin", "cstr") if args.benchmark == "all" else (args.benchmark,)
        for benchmark in selected:
            cache_benchmark(benchmark, args.workers, args.force)
    if args.benchmark == "all" or args.render_only:
        render()


if __name__ == "__main__":
    main()
