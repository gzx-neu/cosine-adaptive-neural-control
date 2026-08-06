"""Submission figure for the guard-bypassed out-of-domain stress diagnostic.

Figure contract
---------------
Core conclusion:
    When the domain guard is intentionally bypassed for diagnosis, every failed
    HDS--lambda search is visibly anchored to its own nominal trajectory rather
    than being represented as an unpaired point in the corrected column.
Evidence hierarchy:
    Left: all nominal HDS peaks.  Right: accepted corrected peaks only.
    A red cross over a left-side nominal point denotes no safe candidate and an
    explicit offline-solver fallback, not a corrected numerical value.
Archetype: quantitative paired-cohort diagnostic, double-column IEEE figure.
Scope: One frozen selected checkpoint per benchmark; diagnostic only.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "论文写作" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 7.2,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 6.2,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

COLORS = {"VDP": "#6E8FA8", "Penicillin": "#6E8FA8", "CSTR": "#6E8FA8", "corrected": "#C75D44", "fallback": "#7A5148"}
SOURCES = {
    "VDP": ROOT / "kkt_collocation" / "results" / "domain_stress_vdp_seed20260751",
    "Penicillin": ROOT / "kkt_collocation" / "results" / "domain_stress_penicillin_seed20260761",
    "CSTR": ROOT / "kkt_collocation" / "results" / "domain_stress_cstr_900_seed20260722",
}
LAYERS = (("near_10_percent", "Near OOD", "o"), ("far_20_percent", "Far OOD", "^"))


def read_layer(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    raw = np.asarray([float(row["raw_hds_max_g"]) for row in rows])
    corrected = np.asarray([float(row["applied_hds_max_g"]) for row in rows])
    accepted = np.asarray([row["accepted_after_hds_lambda"].strip().lower() == "true" for row in rows])
    return raw, corrected, accepted


def plot_panel(ax: plt.Axes, name: str, directory: Path, panel: str) -> None:
    raw_all: list[float] = []
    corrected_all: list[float] = []
    for layer, label, marker in LAYERS:
        raw, corrected, accepted = read_layer(directory / f"{layer}_per_sample.csv")
        # Deterministic visual separation only; all observations remain plotted.
        jitter = np.linspace(-0.035, 0.035, num=len(raw), endpoint=True)
        for x, left, right in zip(jitter[accepted], raw[accepted], corrected[accepted]):
            ax.plot((x, 1 + x), (left, right), color="0.45", alpha=0.055, lw=0.35, zorder=1)
        ax.scatter(jitter, raw, s=13, color=COLORS[name], alpha=0.18, marker=marker,
                   edgecolors="none", label=f"{label}: nominal", zorder=2)
        ax.scatter(1 + jitter[accepted], corrected[accepted], s=13, color=COLORS["corrected"],
                   alpha=0.18, marker=marker, edgecolors="none", zorder=2)
        if np.any(~accepted):
            # There is deliberately no right-side coordinate for these samples.
            ax.scatter(jitter[~accepted], raw[~accepted], s=16, marker=marker,
                       facecolors="none", edgecolors=COLORS["fallback"], linewidths=0.45, alpha=0.65, zorder=4)
            ax.scatter(jitter[~accepted], raw[~accepted], s=11, marker="x", color=COLORS["fallback"],
                       linewidths=0.45, alpha=0.68,
                       label="No safe $\\lambda$ candidate $\\rightarrow$ solver fallback", zorder=5)
        raw_all.extend(raw)
        corrected_all.extend(corrected[accepted])

    ax.hlines(np.mean(raw_all), -0.20, 0.20, color=COLORS[name], lw=1.8, zorder=3)
    ax.hlines(np.mean(corrected_all), 0.80, 1.20, color=COLORS["corrected"], lw=1.8, zorder=3)
    low = min(min(raw_all), min(corrected_all, default=0.0))
    high = max(max(raw_all), max(corrected_all, default=0.0))
    padding = max(0.08 * (high - low), 1e-4)
    ax.axhline(0, color="0.25", ls="--", lw=0.8, zorder=0)
    ax.set(xlim=(-0.30, 1.30), ylim=(low - padding, high + padding))
    ax.set_xticks((0, 1), ("Nominal policy", "Accepted HDS correction"))
    ax.set_ylabel("Continuous-time peak $g_{\\max}$")
    ax.grid(axis="y", color="0.86", lw=0.55)
    ax.text(0.5, -0.27, f"({panel}) {name}: OOD stress diagnostic", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.0)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.65))
    for panel, ax, (name, directory) in zip(("a", "b", "c"), axes, SOURCES.items()):
        plot_panel(ax, name, directory, panel)
    fig.tight_layout()
    stem = OUT / "ood_stress_diagnostic"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
