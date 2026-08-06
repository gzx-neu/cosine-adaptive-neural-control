"""Render manuscript figures only from frozen final artifacts.

The script intentionally does not inspect legacy 20x20/2100-point plotting
folders.  Every plotted value comes from the fixed three-seed summaries, a
frozen checkpoint, or the explicitly labelled single-checkpoint OOD diagnostic.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "论文写作" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector  # noqa: E402
from run_penicillin_ablation import (  # noqa: E402
    DT as PEN_DT, UMAX as PEN_UMAX, Policy as PenicillinPolicy, g as pen_g,
    gdot as pen_gdot, ode as pen_ode,
)
from run_vdp_ablation import constraint as vdp_g, constraint_derivative as vdp_gdot, vdp_ode  # noqa: E402
from kkt_collocation.train_vdp_kkt_policy import KKTPolicyValueNetwork, TrainConfig  # noqa: E402


plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False,
})
COLORS = {"nominal": "#6E8FA8", "corrected": "#C75D44", "VDP": "#6E8FA8", "Penicillin": "#6E8FA8", "CSTR": "#6E8FA8"}
SEED_DIRS = {
    "VDP": [ROOT / "kkt_collocation" / "results" / f"final_multiseed_vdp900_penalty_seed{seed}" for seed in (20260751, 20260752, 20260753)],
    "Penicillin": [ROOT / "kkt_collocation" / "results" / f"final_multiseed_penicillin400_penalty_seed{seed}" for seed in (20260761, 20260762, 20260763)],
}
CONSERVATIVE = ROOT / "kkt_collocation" / "results" / "conservative_margin_1e6"
CONSERVATIVE_SEEDS = {
    "VDP": (20260751, 20260752, 20260753),
    "Penicillin": (20260761, 20260762, 20260763),
    "CSTR": (20260718, 20260725, 20260726),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel_caption(ax: plt.Axes, text: str) -> None:
    """Place a compact (a)/(b) panel label below, rather than over, the data."""
    ax.text(.5, -.27, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, clip_on=False)


def plot_gate_decision() -> None:
    """Three-seed gate evidence as points and mean lines, not bars."""
    summaries = {name: [load(directory / "summary.json") for directory in directories] for name, directories in SEED_DIRS.items()}
    labels = ["VDP\nretain supervised", "Penicillin\nselect KKT-refined"]
    rate = {name: np.asarray([100 * item["adaptive_gate"]["supervised_raw_validation"]["violation_rate"] for item in values]) for name, values in summaries.items()}
    peak = {name: np.asarray([100 * item["adaptive_gate"]["supervised_raw_validation"]["normalized_peak_violation"] for item in values]) for name, values in summaries.items()}
    rate_threshold = 100 * summaries["VDP"][0]["adaptive_gate"]["thresholds"]["allowed_violation_rate"]
    peak_threshold = 100 * summaries["VDP"][0]["adaptive_gate"]["thresholds"]["allowed_normalized_peak_violation"]
    x = np.arange(2); jitter = np.array([-.075, 0, .075])
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.65), sharex=True)
    for ax, values, threshold, title, ylabel in (
        (axes[0], rate, rate_threshold, "Severe-violation rate", "Validation samples (%)"),
        (axes[1], peak, peak_threshold, "Maximum normalized violation", "Constraint scale (%)"),
    ):
        ax.axhline(threshold, color="0.25", ls="--", lw=1, label=f"gate threshold = {threshold:.0f}%")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels)
        ymax = max(*(array.max() for array in values.values()), threshold) * 1.22
        ax.set_ylim(0, ymax)
        for index, name in enumerate(("VDP", "Penicillin")):
            ax.scatter(index + jitter, values[name], s=24, color=COLORS[name], alpha=.32, edgecolors="none")
            ax.hlines(values[name].mean(), index-.18, index+.18, color=COLORS[name], lw=2.1)
            ax.text(index+.21, values[name].mean(), f"mean {values[name].mean():.1f}", va="center", fontsize=7)
        ax.grid(axis="y", alpha=.22)
        ax.legend(frameon=False, loc="upper left")
    save(fig, "gate_validation_decision")


def plot_hds_and_burden() -> None:
    sampling = [
        load(ROOT / "kkt_collocation" / "results" / "final_sampling_vs_hds_vdp_3seeds" / "summary.json")["comparison"]["audits"],
        load(ROOT / "kkt_collocation" / "results" / "final_sampling_vs_hds_penicillin_3seeds" / "summary.json")["comparison"]["audits"],
    ]
    keys = ["zoh_endpoints", "uniform_10", "uniform_100"]
    labels = ["ZOH endpoints", "10 samples", "100 samples"]
    fig, ax = plt.subplots(figsize=(6.7, 3.0))
    x = np.arange(3); width = .34
    for offset, audit, name in ((-width/2, sampling[0], "VDP"), (width/2, sampling[1], "Penicillin")):
        values = [100 * audit[key]["false_safe_rate_vs_hds"] for key in keys]
        bars = ax.bar(x + offset, values, width, label=name, color=COLORS[name])
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, value + 2.2, f"{value:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("False-safe rate relative to HDS (%)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    save(fig, "hds_sampling_false_safe_rate")

    vdp = load(ROOT / "kkt_collocation" / "results" / "final_multiseed_vdp900_penalty_aggregate" / "summary.json")["methods"]
    pen = load(ROOT / "kkt_collocation" / "results" / "final_multiseed_penicillin400_penalty_aggregate" / "summary.json")["methods"]
    names = ["Never-KKT + HDS-lambda", "Constraint-penalty: S+P + HDS-lambda", "Always-KKT + HDS-lambda"]
    labels = ["Supervised", "Penalty-trained", "KKT-refined"]
    fig, ax = plt.subplots(figsize=(6.7, 3.0))
    x = np.arange(3)
    for offset, data, name in ((-.18, vdp, "VDP"), (.18, pen, "Penicillin")):
        means = [data[key]["mean_corrected_segments"]["mean"] for key in names]
        stds = [data[key]["mean_corrected_segments"]["sample_std"] for key in names]
        ax.bar(x + offset, means, .36, yerr=stds, capsize=3, color=COLORS[name], label=name)
    ax.set_ylabel("Corrected ZOH segments per sequence")
    ax.set_xticks(x, labels)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.22)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    save(fig, "correction_burden_ablation")


def _rows(path: Path, method: str) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["method"] == method]


def _adaptive_safe_rows(problem: str) -> list[dict]:
    rows: list[dict] = []
    for seed_index, seed in enumerate(CONSERVATIVE_SEEDS[problem]):
        directory = CONSERVATIVE / f"{problem.lower()}_seed{seed}"
        with (directory / "per_sample.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row["training_seed_index"] = seed_index
                if problem == "VDP":
                    row["nominal_cost"] = row["nominal_objective"]
                    row["applied_cost"] = row["applied_objective"]
                elif problem == "Penicillin":
                    row["raw_hds_max_g"] = row["nominal_hds_max_g"]
                    row["raw_product"] = str(-float(row["nominal_objective"]))
                    row["applied_product"] = str(-float(row["applied_objective"]))
                    row["product_change"] = str(-float(row["objective_change"]))
                rows.append(row)
    return rows


def _binned_mean(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if not len(x):
        return
    edges = np.linspace(x.min(), x.max(), 13)
    centers = (edges[:-1] + edges[1:]) / 2
    means = np.array([y[(x >= lo) & (x < hi if i < len(edges)-2 else x <= hi)].mean()
                      if np.any((x >= lo) & (x < hi if i < len(edges)-2 else x <= hi)) else np.nan
                      for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))])
    ax.plot(centers, means, color=color, lw=2.2, label="Binned mean")


def plot_population_figures() -> None:
    """Full 1,200-point three-seed population views for the Adaptive branch."""
    records = {name: _adaptive_safe_rows(name) for name in ("VDP", "Penicillin")}
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15))
    for row_index, (name, rows) in enumerate(records.items()):
        if name == "VDP":
            raw = np.asarray([float(row["nominal_hds_max_g"]) for row in rows])
            applied = np.asarray([float(row["applied_hds_max_g"]) for row in rows])
            performance_loss = np.asarray([float(row["objective_change"]) for row in rows])
            performance_label = "Cost increase after correction"
        else:
            raw = np.asarray([float(row["raw_hds_max_g"]) for row in rows])
            applied = np.asarray([float(row["applied_hds_max_g"]) for row in rows])
            performance_loss = -np.asarray([float(row["product_change"]) for row in rows])
            performance_label = "Product loss after correction"
        color = COLORS[name]
        ax = axes[row_index, 0]
        ax.scatter(raw, applied, s=9, color=color, alpha=.16, edgecolors="none", label="All test trajectories")
        lo, hi = min(raw.min(), applied.min()), max(raw.max(), applied.max())
        pad = .06 * max(hi-lo, 1e-5); lo -= pad; hi += pad
        ax.plot([lo, hi], [lo, hi], color="0.5", ls=":", lw=1, label="No correction")
        ax.axhline(0, color="0.25", ls="--", lw=.9)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Raw HDS peak $g_{\\max}$")
        ax.set_ylabel("Post-filter HDS peak $g_{\\max}$")
        ax.set_title(f"({chr(97 + row_index*2)}) {name}: continuous-time repair")
        ax.grid(alpha=.20)
        if row_index == 0:
            ax.legend(frameon=False, loc="lower right")
        ax = axes[row_index, 1]
        raw_positive = np.maximum(raw, 0)
        ax.scatter(raw_positive, performance_loss, s=9, color=color, alpha=.16, edgecolors="none", label="All test trajectories")
        if name == "Penicillin":
            cap = .006
            outliers = performance_loss > cap
            if np.any(outliers):
                ax.scatter(raw_positive[outliers], np.full(outliers.sum(), cap), s=20, marker="^", color=color,
                           alpha=.8, edgecolors="none", label=f"{outliers.sum()} values above axis range")
            ax.set_ylim(-.0002, cap)
        ax.axhline(0, color="0.25", ls="--", lw=.9)
        ax.set_xlabel("Raw violation magnitude $[g_{\\max}]_+$")
        ax.set_ylabel(performance_label)
        ax.set_title(f"({chr(98 + row_index*2)}) {name}: repair--performance trade-off")
        ax.grid(alpha=.20)
        if row_index == 0:
            ax.legend(frameon=False, loc="upper left")
    save(fig, "population_repair_tradeoff")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.8), sharey=False)
    method_labels = ["Supervised", "Penalty-trained", "KKT-refined"]
    method_names = {
        "VDP": ["Never-KKT + HDS-lambda", "Constraint-penalty: S+P + HDS-lambda", "Always-KKT + HDS-lambda"],
        "Penicillin": ["Never-KKT + HDS-lambda", "Constraint-penalty: S+P + HDS-lambda", "Always-KKT + HDS-lambda"],
    }
    for ax, (name, directories) in zip(axes, SEED_DIRS.items()):
        all_rows: list[list[dict]] = [[] for _ in method_labels]
        for directory in directories:
            for index, method in enumerate(method_names[name]):
                all_rows[index].extend(_rows(directory / "per_sample.csv", method))
        rng = np.random.default_rng(20260717)
        for index, group in enumerate(all_rows):
            values = np.asarray([float(row["corrected_segments"]) for row in group])
            jitter = rng.uniform(-.22, .22, size=len(values))
            ax.scatter(np.full(len(values), index) + jitter, values, s=7, color=COLORS[name], alpha=.10, edgecolors="none")
            ax.hlines(values.mean(), index-.27, index+.27, color=COLORS["corrected"], lw=2.1)
        ax.set_xticks(range(3), method_labels)
        ax.set_ylabel("Corrected ZOH segments")
        ax.set_title(name)
        ax.grid(axis="y", alpha=.20)
    save(fig, "correction_burden_distribution")


def _checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _vdp_controls(checkpoint: dict, initial: np.ndarray) -> np.ndarray:
    model = KKTPolicyValueNetwork(TrainConfig())
    model.load_state_dict(checkpoint["S"])
    model.eval()
    mean, std = checkpoint["normalization"]["S"]
    x = torch.tensor(((initial[:2] - np.asarray(mean)) / np.asarray(std))[None, :], dtype=torch.float32)
    with torch.no_grad():
        return model(x)[1].numpy()[0]


def _penicillin_controls(checkpoint: dict, x2: float) -> np.ndarray:
    model = PenicillinPolicy()
    model.load_state_dict(checkpoint["true_KKT"])
    model.eval()
    mean, std = checkpoint["normalization"]["true_KKT"]
    x = torch.tensor([[(x2 - float(mean)) / float(std)]], dtype=torch.float32)
    with torch.no_grad():
        return model(x)[1].numpy()[0]


def _rollout(corrector: HDSLambdaCorrector, initial: np.ndarray, controls: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(initial, dtype=float).copy()
    times, states = [], []
    for index, control in enumerate(controls):
        solution = corrector._integrate(state, float(control), duration)
        local_time = np.linspace(index * duration, (index + 1) * duration, 81)
        local_state = solution.sol(local_time - index * duration).T
        if index:
            local_time, local_state = local_time[1:], local_state[1:]
        times.append(local_time); states.append(local_state)
        state = solution.y[:, -1]
    return np.concatenate(times), np.vstack(states)


def _step(ax: plt.Axes, controls: np.ndarray, duration: float, color: str, label: str) -> None:
    t = np.arange(len(controls) + 1) * duration
    ax.step(t, np.r_[controls, controls[-1]], where="post", color=color, lw=1.7, label=label)


def _constraint_path(corrector: HDSLambdaCorrector, initial: np.ndarray, controls: np.ndarray,
                     duration: float, constraint) -> tuple[np.ndarray, np.ndarray]:
    """Dense path-constraint trace using the same continuous model as HDS."""
    state = np.asarray(initial, dtype=float).copy()
    times, values = [], []
    for index, control in enumerate(controls):
        solution = corrector._integrate(state, float(control), duration)
        local_time = np.linspace(index * duration, (index + 1) * duration, 21)
        local_states = solution.sol(local_time - index * duration).T
        local_values = np.asarray([constraint(item) for item in local_states])
        if index:
            local_time, local_values = local_time[1:], local_values[1:]
        times.append(local_time); values.append(local_values)
        state = solution.y[:, -1]
    return np.concatenate(times), np.concatenate(values)


def _constraint_envelope_data() -> dict[str, np.ndarray]:
    """Recreate every frozen Adaptive test path and cache only the plotted traces."""
    cache = CONSERVATIVE / "figure_constraint_envelopes_3seeds.npz"
    if cache.exists():
        with np.load(cache) as data:
            return {key: data[key] for key in data.files}

    paths: dict[str, list[np.ndarray]] = {"vdp_raw": [], "vdp_corrected": [], "pen_raw": [], "pen_corrected": []}
    times: dict[str, np.ndarray] = {}
    for name in ("VDP", "Penicillin"):
        prefix = "vdp" if name == "VDP" else "pen"
        if name == "VDP":
            corrector = HDSLambdaCorrector(vdp_ode, vdp_g, vdp_gdot, (-.3, 1.),
                                            HDSLambdaConfig(grid_size=31, safety_margin=1e-6, max_step_fraction=20.))
            duration, constraint = .5, vdp_g
        else:
            corrector = HDSLambdaCorrector(pen_ode, pen_g, pen_gdot, (0., PEN_UMAX),
                                            HDSLambdaConfig(grid_size=31, safety_margin=1e-6, max_step_fraction=20.))
            duration, constraint = PEN_DT, pen_g
        for seed in CONSERVATIVE_SEEDS[name]:
            with np.load(CONSERVATIVE / f"{name.lower()}_seed{seed}" / "population_controls.npz") as data:
                initial_conditions = data["initial_states"]
                nominal_controls = data["nominal_controls"]
                applied_controls = data["applied_controls"]
            for initial, nominal, applied in zip(initial_conditions, nominal_controls, applied_controls):
                time, raw = _constraint_path(corrector, initial, nominal, duration, constraint)
                _, corrected = _constraint_path(corrector, initial, applied, duration, constraint)
                times[f"{prefix}_time"] = time
                paths[f"{prefix}_raw"].append(raw)
                paths[f"{prefix}_corrected"].append(corrected)
    packed = {key: np.asarray(value) for key, value in paths.items()} | times
    np.savez_compressed(cache, **packed)
    return packed


def _cstr_population_data() -> dict[str, np.ndarray]:
    """Load CSTR paths and verify that they match the conservative controls."""
    directory = ROOT / "kkt_collocation" / "results" / "cstr_multiseed_test1200_900_clean"
    caches = [np.load(directory / f"population_seed{seed}.npz") for seed in (20260718, 20260725, 20260726)]
    old_nominal = np.vstack([cache["raw_controls"] for cache in caches])
    old_applied = np.vstack([cache["applied_controls"] for cache in caches])
    new_caches = [np.load(CONSERVATIVE / f"cstr_seed{seed}" / "population_controls.npz")
                  for seed in CONSERVATIVE_SEEDS["CSTR"]]
    new_nominal = np.vstack([cache["nominal_controls"] for cache in new_caches])
    new_applied = np.vstack([cache["applied_controls"] for cache in new_caches])
    if not np.array_equal(old_nominal, new_nominal) or not np.array_equal(old_applied, new_applied):
        raise RuntimeError("CSTR conservative controls differ from the cached population paths")
    return {
        "time": np.asarray(caches[0]["time"]),
        "raw_constraint": np.vstack([cache["raw_temperature"] for cache in caches]) - 365.0,
        "corrected_constraint": np.vstack([cache["applied_temperature"] for cache in caches]) - 365.0,
        "nominal_controls": new_nominal,
        "corrected_controls": new_applied,
    }


def _adaptive_safe_rows_from_directory(directory: Path) -> list[dict]:
    with (directory / "per_sample.csv").open(encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle)
                if row["method"].startswith("Adaptive (") and row["method"].endswith(" + HDS-lambda")]


def plot_path_constraint_envelopes() -> None:
    """The paper's key population-level before/after safety visual."""
    data = _constraint_envelope_data()
    cstr = _cstr_population_data()
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.15))
    entries = (
        ("VDP", data["vdp_time"], data["vdp_raw"], data["vdp_corrected"]),
        ("Penicillin", data["pen_time"], data["pen_raw"], data["pen_corrected"]),
        ("CSTR", cstr["time"], cstr["raw_constraint"], cstr["corrected_constraint"]),
    )
    for col_index, (name, time, raw, corrected) in enumerate(entries):
        lower = min(raw.min(), corrected.min())
        upper = max(raw.max(), corrected.max())
        margin = max(.06 * (upper - lower), 2e-3)
        zoom = (-.04, .008) if name == "VDP" else ((-.10, .05) if name == "Penicillin" else (-.02, .22))
        # Start the zoom slightly earlier so the approach to the boundary is
        # visible, rather than showing only the near-boundary segment.
        x_zoom = (.20, 2.25) if name == "VDP" else ((1.5, 28.) if name == "Penicillin" else (.90, 1.50))
        for row_index, (series, label, color) in enumerate(((raw, "Nominal policy", COLORS["nominal"]),
                                                             (corrected, "Applied policy", COLORS["corrected"]))):
            ax = axes[row_index, col_index]
            ax.plot(time, series.T, color=color, alpha=.018, lw=.55, zorder=1)
            ax.plot(time, series.max(axis=0), color=color, lw=1.0, zorder=3, label="Population maximum")
            ax.axhline(0, color="0.20", ls="--", lw=1.0, zorder=2, label="Constraint boundary $g=0$")
            ax.set_ylim(lower-margin, upper+margin)
            ax.set_xlabel("Time")
            ax.set_ylabel("Path constraint $g(x(t))$")
            panel_caption(ax, f"({chr(97 + row_index * 3 + col_index)}) {name}: {label}")
            ax.grid(alpha=.20)
            # A compact, transparent boundary zoom sits slightly left of centre
            # in the lower part of the panel, leaving the right-side label clear.
            inset = ax.inset_axes([.23, .055, .30, .29])
            inset.patch.set_alpha(0)
            inset.plot(time, series.T, color=color, alpha=.025, lw=.32)
            inset.plot(time, series.max(axis=0), color=color, lw=.85)
            inset.axhline(0, color="0.20", ls="--", lw=.55)
            inset.set_xlim(*x_zoom); inset.set_ylim(*zoom)
            inset.set_xticks([]); inset.set_yticks([])
    save(fig, "population_path_constraint_envelopes")


def _paired_method_metrics(problem: str, supervised_method: str, kkt_method: str,
                           constraint_key: str, performance_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    supervised_constraint: list[float] = []
    kkt_constraint: list[float] = []
    supervised_performance: list[float] = []
    kkt_performance: list[float] = []
    for directory in SEED_DIRS[problem]:
        with (directory / "per_sample.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        left = {row["sample_index"]: row for row in rows if row["method"] == supervised_method}
        right = {row["sample_index"]: row for row in rows if row["method"] == kkt_method}
        if left.keys() != right.keys():
            raise RuntimeError(f"paired KKT comparison is incomplete in {directory}")
        for index in sorted(left, key=int):
            supervised_constraint.append(float(left[index][constraint_key]))
            kkt_constraint.append(float(right[index][constraint_key]))
            supervised_performance.append(float(left[index][performance_key]))
            kkt_performance.append(float(right[index][performance_key]))
    return (np.asarray(supervised_constraint), np.asarray(kkt_constraint),
            np.asarray(supervised_performance), np.asarray(kkt_performance))


def _paired_scatter(ax: plt.Axes, before: np.ndarray, after: np.ndarray, title: str,
                    xlabel: str, ylabel: str, color: str, threshold: float | None = None) -> None:
    ax.scatter(before, after, s=8, color=color, alpha=.13, edgecolors="none", label="Matched test samples")
    lo = min(before.min(), after.min()); hi = max(before.max(), after.max())
    padding = max(.06 * (hi - lo), 1e-6)
    lo, hi = lo - padding, hi + padding
    ax.plot([lo, hi], [lo, hi], color="0.45", ls=":", lw=1, label="No KKT change")
    if threshold is not None:
        ax.axvline(threshold, color="0.20", ls="--", lw=.85, label="Gate severity threshold")
        ax.axhline(threshold, color="0.20", ls="--", lw=.85)
        before_severe = int(np.sum(before > threshold)); after_severe = int(np.sum(after > threshold))
        ax.text(.03, .92, f"severe: {before_severe} $\\rightarrow$ {after_severe}", transform=ax.transAxes,
                color=COLORS["corrected"], fontsize=7.2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=.20)


def plot_kkt_refinement_effect() -> None:
    """Matched before/after evidence for what KKT refinement changes."""
    vdp = _paired_method_metrics("VDP", "Never-KKT: S", "Always-KKT: S+KKT", "nominal_hds_max_g", "nominal_cost")
    pen = _paired_method_metrics("Penicillin", "Never-KKT: S", "Always-KKT: S+true-KKT", "raw_hds_max_g", "raw_product")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.7))
    _paired_scatter(axes[0, 0], np.maximum(vdp[0], 0), np.maximum(vdp[1], 0),
                    "(a) VDP: path violation", "S: $[g_{\\max}]_+$", "S+KKT: $[g_{\\max}]_+$", COLORS["VDP"], .025 * .4)
    _paired_scatter(axes[0, 1], vdp[2], vdp[3],
                    "(b) VDP: terminal cost", "S: cost $J$", "S+KKT: cost $J$", COLORS["VDP"])
    _paired_scatter(axes[1, 0], np.maximum(pen[0], 0), np.maximum(pen[1], 0),
                    "(c) Penicillin: path violation", "S: $[g_{\\max}]_+$", "S+KKT: $[g_{\\max}]_+$", COLORS["Penicillin"], .025 * .5)
    _paired_scatter(axes[1, 1], pen[2], pen[3],
                    "(d) Penicillin: terminal product", "S: product $x_3(T)$", "S+KKT: product $x_3(T)$", COLORS["Penicillin"])
    axes[0, 0].legend(frameon=False, loc="lower right")
    save(fig, "kkt_refinement_effect")


def _paired_migration(ax: plt.Axes, before: np.ndarray, after: np.ndarray, title: str,
                       ylabel: str, before_color: str, after_color: str,
                       threshold: float | None = None,
                       xlabels: tuple[str, str] = ("Supervised\npolicy", "KKT-refined\npolicy")) -> None:
    """All-sample paired movement plot used only for the alternate figure set."""
    rng = np.random.default_rng(20260717)
    jitter = rng.uniform(-.045, .045, size=len(before))
    for x, left, right in zip(jitter, before, after):
        ax.plot((x, 1 + x), (left, right), color="0.45", alpha=.025, lw=.35, zorder=1)
    ax.scatter(jitter, before, s=7, color=before_color, alpha=.10, edgecolors="none", zorder=2)
    ax.scatter(1 + jitter, after, s=7, color=after_color, alpha=.10, edgecolors="none", zorder=2)
    # Short, dark horizontal summaries give the average movement without
    # replacing the full cohort by a bar chart.
    ax.hlines(before.mean(), -.20, .20, color=before_color, lw=2.1, zorder=3)
    ax.hlines(after.mean(), .80, 1.20, color=after_color, lw=2.1, zorder=3)
    if threshold is not None:
        ax.axhline(threshold, color="0.25", ls="--", lw=.8, zorder=0)
    lo = min(before.min(), after.min(), threshold if threshold is not None else np.inf)
    hi = max(before.max(), after.max(), threshold if threshold is not None else -np.inf)
    padding = max(.08 * (hi - lo), 1e-5)
    ax.set_ylim(lo - padding, hi + padding)
    ax.set_xlim(-.34, 1.34)
    ax.set_xticks((0, 1), xlabels)
    ax.set_ylabel(ylabel)
    panel_caption(ax, title)
    ax.grid(axis="y", alpha=.20)


def plot_candidate_kkt_migration() -> None:
    """Matched KKT-refinement movement for all three benchmark problems."""
    vdp = _paired_method_metrics("VDP", "Never-KKT: S", "Always-KKT: S+KKT", "nominal_hds_max_g", "nominal_cost")
    pen = _paired_method_metrics("Penicillin", "Never-KKT: S", "Always-KKT: S+true-KKT", "raw_hds_max_g", "raw_product")
    cstr_path = ROOT / "kkt_collocation" / "results" / "cstr_kkt_ablation1200_900" / "per_sample.csv"
    with cstr_path.open(encoding="utf-8-sig", newline="") as handle:
        cstr_rows = list(csv.DictReader(handle))
    cstr_left = {(row["training_seed"], row["sample_index"]): row for row in cstr_rows if row["method"] == "Supervised"}
    cstr_right = {(row["training_seed"], row["sample_index"]): row for row in cstr_rows if row["method"] == "KKT-refined"}
    if cstr_left.keys() != cstr_right.keys():
        raise RuntimeError("CSTR matched KKT comparison is incomplete")
    cstr_g_s = np.asarray([float(cstr_left[key]["nominal_hds_max_g"]) for key in sorted(cstr_left)])
    cstr_g_k = np.asarray([float(cstr_right[key]["nominal_hds_max_g"]) for key in sorted(cstr_left)])
    cstr_j_s = np.asarray([float(cstr_left[key]["nominal_objective"]) for key in sorted(cstr_left)])
    cstr_j_k = np.asarray([float(cstr_right[key]["nominal_objective"]) for key in sorted(cstr_left)])
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.05))
    _paired_migration(axes[0, 0], np.maximum(vdp[0], 0), np.maximum(vdp[1], 0),
                      "(a) VDP: path violation", "$[g_{\\max}]_+$", COLORS["VDP"], COLORS["corrected"], .025 * .4)
    _paired_migration(axes[0, 1], np.maximum(pen[0], 0), np.maximum(pen[1], 0),
                      "(b) Penicillin: path violation", "$[g_{\\max}]_+$", COLORS["Penicillin"], COLORS["corrected"], .025 * .5)
    _paired_migration(axes[0, 2], np.maximum(cstr_g_s, 0), np.maximum(cstr_g_k, 0),
                      "(c) CSTR: path violation", "$[g_{\\max}]_+$", COLORS["CSTR"], COLORS["corrected"], 0.)
    _paired_migration(axes[1, 0], vdp[2], vdp[3],
                      "(d) VDP: terminal cost", "Cost $J$", COLORS["VDP"], COLORS["corrected"])
    _paired_migration(axes[1, 1], pen[2], pen[3],
                      "(e) Penicillin: terminal product", "Product $x_3(T)$", COLORS["Penicillin"], COLORS["corrected"])
    _paired_migration(axes[1, 2], cstr_j_s, cstr_j_k,
                      "(f) CSTR: objective", "Objective $J$", COLORS["CSTR"], COLORS["corrected"])
    save(fig, "candidate_kkt_paired_migration")


def plot_candidate_hds_safety_migration() -> None:
    """Alternate Figs. 4--5 evidence: every trajectory's raw-to-safe movement."""
    records = {name: _adaptive_safe_rows(name) for name in ("VDP", "Penicillin")}
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.9))
    for panel, (name, rows) in zip(("(a)", "(b)"), records.items()):
        raw_key = "nominal_hds_max_g" if name == "VDP" else "raw_hds_max_g"
        raw = np.asarray([float(row[raw_key]) for row in rows])
        corrected = np.asarray([float(row["applied_hds_max_g"]) for row in rows])
        ax = axes[0 if name == "VDP" else 1]
        rng = np.random.default_rng(20260717)
        jitter = rng.uniform(-.045, .045, size=len(raw))
        for x, left, right in zip(jitter, raw, corrected):
            ax.plot((x, 1 + x), (left, right), color="0.45", alpha=.022, lw=.35, zorder=1)
        ax.scatter(jitter, raw, s=7, color=COLORS[name], alpha=.10, edgecolors="none", zorder=2)
        ax.scatter(1 + jitter, corrected, s=7, color=COLORS["corrected"], alpha=.10, edgecolors="none", zorder=2)
        ax.hlines(raw.mean(), -.20, .20, color=COLORS[name], lw=2.1, zorder=3)
        ax.hlines(corrected.mean(), .80, 1.20, color=COLORS["corrected"], lw=2.1, zorder=3)
        ax.axhline(0, color="0.25", ls="--", lw=.8, zorder=0)
        lo, hi = min(raw.min(), corrected.min()), max(raw.max(), corrected.max())
        padding = max(.08 * (hi - lo), 1e-4)
        ax.set_ylim(lo - padding, hi + padding)
        ax.set_xlim(-.34, 1.34)
        ax.set_xticks((0, 1), ("Nominal\npolicy", "Accepted\ncorrection"))
        ax.set_ylabel("Continuous-time peak $g_{\\max}$")
        ax.set_title(f"{panel} {name}")
        ax.grid(axis="y", alpha=.20)
    save(fig, "candidate_hds_safety_migration")


def plot_candidate_hds_objective_migration() -> None:
    """Direct before/after objective evidence for the cost of HDS correction."""
    records = {name: _adaptive_safe_rows(name) for name in ("VDP", "Penicillin")}
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.9))
    for ax, (name, rows) in zip(axes, records.items()):
        if name == "VDP":
            nominal = np.asarray([float(row["nominal_cost"]) for row in rows])
            corrected = np.asarray([float(row["applied_cost"]) for row in rows])
            title, ylabel = "(a) VDP: terminal cost", "Cost $J$"
        else:
            nominal = np.asarray([float(row["raw_product"]) for row in rows])
            corrected = np.asarray([float(row["applied_product"]) for row in rows])
            title, ylabel = "(b) Penicillin: terminal product", "Product $x_3(T)$"
        _paired_migration(ax, nominal, corrected, title, ylabel, COLORS[name], COLORS["corrected"],
                          xlabels=("Nominal policy", "Applied policy"))
    save(fig, "candidate_hds_objective_migration")


def plot_hds_safety_performance_migration() -> None:
    """All-sample HDS safety/objective migration for every benchmark."""
    # All available in-domain observations are retained: 1200 VDP, 1200
    # penicillin, and 1200 CSTR trajectories.  CSTR is not added to the
    # KKT-specific panel because its validation gate retained Supervised.
    records = {name: _adaptive_safe_rows(name) for name in ("VDP", "Penicillin", "CSTR")}
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.55))
    for col_index, (name, rows) in enumerate(records.items()):
        raw_key = "nominal_hds_max_g" if name in ("VDP", "CSTR") else "raw_hds_max_g"
        raw_peak = np.asarray([float(row[raw_key]) for row in rows])
        repaired_peak = np.asarray([float(row["applied_hds_max_g"]) for row in rows])
        _paired_migration(axes[0, col_index], raw_peak, repaired_peak,
                          f"({chr(97 + col_index)}) {name}: safety peak",
                          "Continuous-time peak $g_{\\max}$", COLORS[name], COLORS["corrected"],
                          threshold=0., xlabels=("Nominal\npolicy", "Applied\npolicy"))
        if name == "VDP":
            nominal = np.asarray([float(row["nominal_cost"]) for row in rows])
            repaired = np.asarray([float(row["applied_cost"]) for row in rows])
            title, ylabel = "(d) VDP: terminal cost", "Cost $J$"
        elif name == "Penicillin":
            nominal = np.asarray([float(row["raw_product"]) for row in rows])
            repaired = np.asarray([float(row["applied_product"]) for row in rows])
            title, ylabel = "(e) Penicillin: terminal product", "Product $x_3(T)$"
        else:
            nominal = np.asarray([float(row["nominal_objective"]) for row in rows])
            repaired = np.asarray([float(row["applied_objective"]) for row in rows])
            title, ylabel = "(f) CSTR: objective", "Objective $J$"
        _paired_migration(axes[1, col_index], nominal, repaired, title, ylabel,
                          COLORS[name], COLORS["corrected"],
                          xlabels=("Nominal\npolicy", "Applied\npolicy"))
    save(fig, "hds_safety_performance_migration")


def plot_hds_repair_intensity() -> None:
    """All-test relationship between violation severity and HDS--lambda effort."""
    records = {name: _adaptive_safe_rows(name) for name in ("VDP", "Penicillin")}
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.55))
    for row_index, (name, rows) in enumerate(records.items()):
        raw_key = "nominal_hds_max_g" if name == "VDP" else "raw_hds_max_g"
        raw = np.maximum(np.asarray([float(row[raw_key]) for row in rows]), 0.)
        segments = np.asarray([float(row["corrected_segments"]) for row in rows])
        lambda_change = np.asarray([float(row["mean_abs_lambda_minus_one"]) for row in rows])
        for col_index, (values, ylabel, title) in enumerate(((segments, "Corrected ZOH segments", "repair extent"),
                                                              (lambda_change, "Mean $|\\lambda-1|$", "repair magnitude"))):
            ax = axes[row_index, col_index]
            ax.scatter(raw, values, s=8, color=COLORS[name], alpha=.14, edgecolors="none", label="All test trajectories")
            if col_index == 1:
                cap = .006 if name == "VDP" else .014
                outliers = values > cap
                if np.any(outliers):
                    ax.scatter(raw[outliers], np.full(outliers.sum(), cap), s=20, marker="^", color=COLORS[name],
                               alpha=.8, edgecolors="none", label=f"{outliers.sum()} values above axis range")
                ax.set_ylim(-.0003, cap)
            ax.axvline(0, color="0.2", ls="--", lw=.8)
            ax.set_xlabel("Raw violation magnitude $[g_{\\max}]_+$")
            ax.set_ylabel(ylabel)
            ax.set_title(f"({chr(97 + row_index*2 + col_index)}) {name}: {title}")
            ax.grid(alpha=.20)
            if row_index == 0 and col_index == 0:
                ax.legend(frameon=False, loc="upper left")
    save(fig, "hds_repair_intensity")


def plot_sampling_vs_hds_population() -> None:
    """Why sampled checks can miss continuous-time violations, trajectory by trajectory."""
    sources = {
        "VDP": ROOT / "kkt_collocation" / "results" / "final_sampling_vs_hds_vdp_3seeds" / "per_trajectory.csv",
        "Penicillin": ROOT / "kkt_collocation" / "results" / "final_sampling_vs_hds_penicillin_3seeds" / "per_trajectory.csv",
    }
    methods = (("zoh_endpoints", "ZOH endpoints", "#7F8C8D"),
               ("uniform_10", "10 uniform samples", "#4E79A7"),
               ("uniform_100", "100 uniform samples", "#59A14F"))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.95))
    for ax, (name, path) in zip(axes, sources.items()):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        hds = np.asarray([float(row["hds_peak"]) for row in rows])
        low = min(hds.min(), *(float(row[f"{key}_peak"]) for row in rows for key, _, _ in methods))
        high = max(hds.max(), *(float(row[f"{key}_peak"]) for row in rows for key, _, _ in methods))
        pad = max(.06 * (high-low), 1e-4); lo, hi = low-pad, high+pad
        ax.fill_between([0, hi], lo, 0, color="#C75B39", alpha=.055, zorder=0)
        for key, label, color in methods:
            sampled = np.asarray([float(row[f"{key}_peak"]) for row in rows])
            ax.scatter(hds, sampled, s=7, color=color, alpha=.10, edgecolors="none", label=label)
        ax.plot([lo, hi], [lo, hi], color="0.45", ls=":", lw=1, label="Agreement with HDS")
        ax.axvline(0, color="0.2", ls="--", lw=.8); ax.axhline(0, color="0.2", ls="--", lw=.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Continuous-time HDS peak $g_{\\max}$")
        ax.set_ylabel("Sampled peak $g_{\\max}$")
        ax.set_title(name)
        ax.grid(alpha=.20)
        handles, labels = ax.get_legend_handles_labels()
        unique = {label: handle for handle, label in zip(handles, labels) if label != "Binned mean"}
        ax.legend(unique.values(), unique.keys(), frameon=False, loc="upper left")
    save(fig, "hds_sampling_population")


def plot_method_pipeline() -> None:
    """Compact paper schematic of the offline selection and deployment safeguards."""
    fig, ax = plt.subplots(figsize=(7.1, 2.4)); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    def box(x, y, w, h, text, color):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                               ec=color, fc=color, alpha=.16, lw=1.2)
        ax.add_patch(patch); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=7.3)
    def arrow(x0, y0, x1, y1, text=None):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.25"))
        if text: ax.text((x0+x1)/2, (y0+y1)/2+.035, text, ha="center", fontsize=6.4, color="0.25")
    # Offline selection lane.
    box(.035, .65, .16, .20, "Safe teacher labels\nand supervised policy", COLORS["VDP"])
    box(.275, .65, .16, .20, "Independent validation\npre-registered gate", COLORS["corrected"])
    box(.515, .65, .16, .20, "retain supervised\nor refine with KKT", COLORS["Penicillin"])
    arrow(.195, .75, .275, .75); arrow(.435, .75, .515, .75, "select once")
    # Deployment lane.
    box(.035, .14, .14, .22, "Measured\ninitial state $x_0$", "#7F8C8D")
    box(.245, .14, .15, .22, "Operating-domain\nguard", "#7F8C8D")
    box(.465, .14, .15, .22, "Nominal policy\n(full ZOH sequence)", COLORS["VDP"])
    box(.695, .14, .13, .22, "HDS safety\ncorrection", COLORS["corrected"])
    box(.875, .14, .09, .22, "Execute\nor fallback", "#7F8C8D")
    arrow(.175, .25, .245, .25); arrow(.395, .25, .465, .25, "in domain"); arrow(.615, .25, .695, .25); arrow(.825, .25, .875, .25)
    arrow(.320, .65, .320, .37, "validation result")
    ax.text(.320, .06, "out of domain or no $\\lambda$-ray $\\Rightarrow$ offline optimizer", ha="center", fontsize=6.8, color="0.25")
    save(fig, "method_pipeline")


def plot_typical_trajectories() -> None:
    """One severe but accepted in-domain case per benchmark, recreated from frozen models."""
    vdp_dir = ROOT / "kkt_collocation" / "results" / "final_multiseed_vdp900_penalty_seed20260751"
    pen_dir = ROOT / "kkt_collocation" / "results" / "final_multiseed_penicillin400_penalty_seed20260761"
    vdp_case = max(_rows(vdp_dir / "per_sample.csv", "Never-KKT: S"), key=lambda row: float(row["nominal_hds_max_g"]))
    pen_case = max(_rows(pen_dir / "per_sample.csv", "Always-KKT: S+true-KKT"), key=lambda row: float(row["raw_hds_max_g"]))
    vdp_initial = np.array([float(vdp_case["y1_0"]), float(vdp_case["y2_0"]), 0.0])
    pen_initial = np.array([1.0, float(pen_case["x2_0"]), 0.001, 250.0])
    vdp_corrector = HDSLambdaCorrector(vdp_ode, vdp_g, vdp_gdot, (-.3, 1.0), HDSLambdaConfig(grid_size=31, max_step_fraction=100.0))
    pen_corrector = HDSLambdaCorrector(pen_ode, pen_g, pen_gdot, (0.0, PEN_UMAX), HDSLambdaConfig(grid_size=31, max_step_fraction=100.0))
    vdp_nominal = _vdp_controls(_checkpoint(vdp_dir / "models.pth"), vdp_initial)
    pen_nominal = _penicillin_controls(_checkpoint(pen_dir / "models.pth"), pen_initial[1])
    vdp_safe = vdp_corrector.correct(vdp_initial, vdp_nominal, .5)
    pen_safe = pen_corrector.correct(pen_initial, pen_nominal, PEN_DT)
    if not vdp_safe.accepted or not pen_safe.accepted:
        raise RuntimeError("selected frozen typical case was not accepted by HDS-lambda")
    tvn, xvn = _rollout(vdp_corrector, vdp_initial, vdp_nominal, .5)
    tvc, xvc = _rollout(vdp_corrector, vdp_initial, vdp_safe.controls, .5)
    tpn, xpn = _rollout(pen_corrector, pen_initial, pen_nominal, PEN_DT)
    tpc, xpc = _rollout(pen_corrector, pen_initial, pen_safe.controls, PEN_DT)
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.5))
    axes[0, 0].plot(tvn, xvn[:, 0], color=COLORS["nominal"], lw=1.5, label="Nominal")
    axes[0, 0].plot(tvc, xvc[:, 0], color=COLORS["corrected"], lw=1.5, label="HDS correction")
    axes[0, 0].axhline(-.4, color="0.25", ls="--", lw=1, label="limit")
    axes[0, 0].set_ylabel("VDP state $y_1$")
    axes[0, 0].set_title("(a) State constraint")
    _step(axes[0, 1], vdp_nominal, .5, COLORS["nominal"], "Nominal")
    _step(axes[0, 1], vdp_safe.controls, .5, COLORS["corrected"], "HDS correction")
    axes[0, 1].set_ylabel("Control $u$"); axes[0, 1].set_title("(b) ZOH correction")
    axes[0, 1].axhline(-.3, color="0.55", ls=":", lw=.8); axes[0, 1].axhline(1., color="0.55", ls=":", lw=.8)
    axes[0, 2].plot(tvn, xvn[:, 2], color=COLORS["nominal"], lw=1.5, label="Nominal")
    axes[0, 2].plot(tvc, xvc[:, 2], color=COLORS["corrected"], lw=1.5, label="HDS correction")
    axes[0, 2].set_ylabel("Cumulative cost $J$"); axes[0, 2].set_title("(c) Performance")
    axes[1, 0].plot(tpn, xpn[:, 1], color=COLORS["nominal"], lw=1.5, label="Nominal")
    axes[1, 0].plot(tpc, xpc[:, 1], color=COLORS["corrected"], lw=1.5, label="HDS correction")
    axes[1, 0].axhline(.5, color="0.25", ls="--", lw=1, label="limit")
    axes[1, 0].set_ylabel("Substrate $x_2$"); axes[1, 0].set_title("(d) State constraint")
    _step(axes[1, 1], pen_nominal, PEN_DT, COLORS["nominal"], "Nominal")
    _step(axes[1, 1], pen_safe.controls, PEN_DT, COLORS["corrected"], "HDS correction")
    axes[1, 1].set_ylabel("Feed $u$"); axes[1, 1].set_title("(e) ZOH correction")
    axes[1, 1].axhline(0, color="0.55", ls=":", lw=.8); axes[1, 1].axhline(PEN_UMAX, color="0.55", ls=":", lw=.8)
    axes[1, 2].plot(tpn, xpn[:, 2], color=COLORS["nominal"], lw=1.5, label="Nominal")
    axes[1, 2].plot(tpc, xpc[:, 2], color=COLORS["corrected"], lw=1.5, label="HDS correction")
    axes[1, 2].set_ylabel("Product $x_3$"); axes[1, 2].set_title("(f) Performance")
    for row in axes:
        for ax in row:
            ax.set_xlabel("Time")
            ax.grid(alpha=.20)
    axes[0, 0].legend(frameon=False, loc="best")
    save(fig, "typical_hds_lambda_corrections")


def _control_envelope_data() -> dict[str, np.ndarray]:
    """Recreate all frozen in-domain Adaptive controls before/after HDS repair.

    The per-sample audit table stores scalar statistics, not the complete ZOH
    sequence.  This cache therefore reconstructs the policy outputs and their
    accepted HDS--lambda counterparts for every one of the 1,200 test points
    (three frozen seeds times 400 points) in each benchmark.
    """
    packed: dict[str, np.ndarray] = {}
    for name in ("VDP", "Penicillin"):
        prefix = "vdp" if name == "VDP" else "pen"
        caches = [np.load(CONSERVATIVE / f"{name.lower()}_seed{seed}" / "population_controls.npz")
                  for seed in CONSERVATIVE_SEEDS[name]]
        packed[f"{prefix}_nominal"] = np.vstack([cache["nominal_controls"] for cache in caches])
        packed[f"{prefix}_corrected"] = np.vstack([cache["applied_controls"] for cache in caches])
    return packed


def plot_population_control_envelopes() -> None:
    """Pure population-level ZOH controls, preserving every in-domain test case."""
    data = _control_envelope_data()
    cstr = _cstr_population_data()
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.95))
    entries = (
        ("VDP", data["vdp_nominal"], data["vdp_corrected"], .5, (-.3, 1.), "Control $u$"),
        ("Penicillin", data["pen_nominal"], data["pen_corrected"], PEN_DT, (0., PEN_UMAX), "Feed $u$"),
        ("CSTR", cstr["nominal_controls"], cstr["corrected_controls"], .15, (0., 100.),
         "Heat removal $q_c$"),
    )
    for col_index, (name, nominal, corrected, duration, bounds, ylabel) in enumerate(entries):
        for row_index, (series, title, color) in enumerate(((nominal, "Nominal policy", COLORS["nominal"]),
                                                              (corrected, "Applied policy", COLORS["corrected"]))):
            ax = axes[row_index, col_index]
            time = np.arange(series.shape[1] + 1) * duration
            for control in series:
                ax.step(time, np.r_[control, control[-1]], where="post", color=color, alpha=.011, lw=.28, zorder=1)
            for value in bounds:
                ax.axhline(value, color="0.38", ls=":", lw=.8, zorder=2)
            pad = .075 * (bounds[1] - bounds[0])
            ax.set_xlim(time[0], time[-1])
            ax.set_ylim(bounds[0] - pad, bounds[1] + pad)
            ax.set_xlabel("Time")
            ax.set_ylabel(ylabel)
            panel_caption(ax, f"({chr(97 + row_index * 3 + col_index)}) {name}: {title}")
            ax.grid(axis="x", alpha=.16)
    save(fig, "population_control_envelopes")


def plot_ood_stress() -> None:
    """OOD diagnosis with fallback marks anchored to their nominal trajectories."""
    sources = {
        "VDP": CONSERVATIVE / "ood_vdp_seed20260751",
        "Penicillin": CONSERVATIVE / "ood_penicillin_seed20260761",
        "CSTR": CONSERVATIVE / "ood_cstr_seed20260722",
    }
    # A single-column vertical stack keeps the OOD diagnostic adjacent to its
    # discussion in the IEEE two-column layout without discarding any points.
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 5.0))
    layers = (("near_10_percent", "Near OOD", "o"), ("far_20_percent", "Far OOD", "^"))
    for ax, (name, directory) in zip(axes, sources.items()):
        raw_all: list[float] = []
        repaired_all: list[float] = []
        for layer, label, marker in layers:
            with (directory / f"{layer}_per_sample.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            raw = np.asarray([float(row["raw_hds_max_g"]) for row in rows])
            repaired = np.asarray([float(row["applied_hds_max_g"]) for row in rows])
            accepted = np.asarray([row["accepted_after_hds_lambda"].strip().lower() == "true" for row in rows])
            # Deterministic visual separation only; every point is retained.
            jitter = np.linspace(-.035, .035, num=len(rows), endpoint=True)
            for x, left, right in zip(jitter[accepted], raw[accepted], repaired[accepted]):
                ax.plot((x, 1+x), (left, right), color="0.45", alpha=.055, lw=.35, zorder=1)
            ax.scatter(jitter, raw, s=13, color=COLORS[name], alpha=.18,
                       marker=marker, edgecolors="none", label=f"{label}: nominal", zorder=2)
            ax.scatter(1+jitter[accepted], repaired[accepted], s=13, color=COLORS["corrected"], alpha=.18,
                       marker=marker, edgecolors="none", zorder=2)
            # A fallback has no corrected ordinate. Mark its associated raw point,
            # rather than placing an unanchored cross in the corrected column.
            if np.any(~accepted):
                ax.scatter(jitter[~accepted], raw[~accepted], s=45, marker=marker,
                           facecolors="none", edgecolors="#A83E32", linewidths=.85, zorder=4)
                ax.scatter(jitter[~accepted], raw[~accepted], s=34, marker="x",
                           color="#A83E32", linewidths=1.05,
                           label="No safe $\\lambda$ candidate $\\rightarrow$ offline-optimizer dispatch", zorder=5)
            raw_all.extend(raw)
            repaired_all.extend(repaired[accepted])
        ax.hlines(np.mean(raw_all), -.20, .20, color=COLORS[name], lw=1.8, zorder=3)
        if repaired_all:
            ax.hlines(np.mean(repaired_all), .80, 1.20, color=COLORS["corrected"], lw=1.8, zorder=3)
        lo = min(min(raw_all), min(repaired_all, default=0.)); hi = max(max(raw_all), max(repaired_all, default=0.))
        pad = max(.08 * (hi-lo), 1e-4)
        ax.axhline(0, color="0.25", ls="--", lw=.8, zorder=0)
        ax.set_ylim(lo-pad, hi+pad)
        ax.set_xlim(-.30, 1.30)
        ax.set_xticks((0, 1), ("Nominal\npolicy", "Accepted\ncorrection"))
        ax.set_ylabel("Continuous-time peak $g_{\\max}$")
        panel_caption(ax, f"({chr(97 + list(sources).index(name))}) {name}: OOD stress diagnostic")
        ax.grid(axis="y", alpha=.20)
    save(fig, "ood_stress_diagnostic")


def main() -> None:
    plot_method_pipeline()
    plot_gate_decision()
    plot_population_figures()
    plot_path_constraint_envelopes()
    plot_kkt_refinement_effect()
    plot_candidate_kkt_migration()
    plot_candidate_hds_safety_migration()
    plot_candidate_hds_objective_migration()
    plot_hds_safety_performance_migration()
    plot_hds_repair_intensity()
    plot_sampling_vs_hds_population()
    plot_typical_trajectories()
    plot_population_control_envelopes()
    plot_ood_stress()
    print(OUT)


if __name__ == "__main__":
    main()
