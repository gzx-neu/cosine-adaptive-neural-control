"""Plot complete 30-seed CA-KKT control populations with local corrections.

All 30 x 400 nominal ZOH control sequences are retained for every benchmark.
The HDS-applied input is drawn only on segments whose control was actually
changed.  Dense population layers are rasterized in vector exports while axes,
labels, summaries, and legends remain editable.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kkt_collocation"))

from kkt_collocation.economou_cstr_hds_fast import (  # noqa: E402
    candidates as cstr_candidates,
    segment_event_audit,
)
from kkt_collocation.plot_linear_cosine_constraint_population_30seeds import (  # noqa: E402
    METHOD,
    _cstr_protocol,
    _predict_cstr,
    _predict_scalar,
    _read_rows,
    _seed_dir,
)
from kkt_collocation.run_penicillin_ablation import (  # noqa: E402
    DT as PEN_DT,
    UMAX as PEN_UMAX,
    g as pen_g,
    gdot as pen_gdot,
    ode as pen_ode,
)
from kkt_collocation.run_vdp_ablation import (  # noqa: E402
    constraint as vdp_g,
    constraint_derivative as vdp_gdot,
    vdp_ode,
)
from offline_safe_control.adaptive_event_hds import (  # noqa: E402
    AdaptiveEventHDSConfig,
    AdaptiveEventHDSCorrector,
)


SEEDS = tuple(range(20260771, 20260801))
RESULTS = ROOT / "kkt_collocation" / "results"
OUT = RESULTS / "paper30_linear_cosine_control_corrections_margin1e6_20260803_v1"
FIGURES = ROOT / "论文写作" / "figures"
CSTR_MARGIN_RESULTS = RESULTS / "multiseed30_cstr_k10_margin1e6_20260803_v1"
CSTR_THRESHOLD = -1e-6
CSTR_GRID = 31


def _cstr_correct_controls(state0: np.ndarray, controls: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the frozen closest-to-one correction without a report-only final pass."""
    state = np.asarray(state0, float).copy()
    output = np.asarray(controls, float).copy()
    changed = np.zeros(cfg.zoh_steps, dtype=bool)
    for index in range(cfg.zoh_steps):
        peaks, next_state, _ = segment_event_audit(state, output[index], cfg)
        if float(np.max(peaks)) <= CSTR_THRESHOLD:
            state = next_state
            continue
        low, span, normalized, order = cstr_candidates(output[index], cfg, CSTR_GRID)
        found = None
        for lam in order:
            candidate = low + np.clip(lam * normalized, 0.0, 1.0) * span
            candidate_peaks, candidate_state, _ = segment_event_audit(state, candidate, cfg)
            if float(np.max(candidate_peaks)) <= CSTR_THRESHOLD:
                found = candidate, candidate_state
                break
        if found is None:
            raise RuntimeError(f"No safe CSTR candidate at segment {index}")
        output[index], state = found
        changed[index] = True
    return output, changed


def _cache_scalar(benchmark: str, seed: int) -> dict[str, np.ndarray]:
    directory = _seed_dir(benchmark, seed)
    states = np.load(directory / "test_states.npy")
    rows = _read_rows(directory / f"test_per_sample_{METHOD}.csv")
    nominal = _predict_scalar(benchmark, directory, states)
    if benchmark == "vdp":
        initials = states
        duration = 0.5
        corrector = AdaptiveEventHDSCorrector(
            vdp_ode,
            vdp_g,
            vdp_gdot,
            (-0.3, 1.0),
            AdaptiveEventHDSConfig(grid_size=31, safety_margin=1e-6),
        )
    else:
        initials = np.c_[np.ones(400), states, np.full(400, 0.001), np.full(400, 250.0)]
        duration = PEN_DT
        corrector = AdaptiveEventHDSCorrector(
            pen_ode,
            pen_g,
            pen_gdot,
            (0.0, PEN_UMAX),
            AdaptiveEventHDSConfig(grid_size=31, safety_margin=1e-6),
        )

    applied = nominal.copy()
    changed = np.zeros_like(nominal, dtype=bool)
    for index, (initial, controls, row) in enumerate(zip(initials, nominal, rows)):
        expected = int(float(row["corrected_segments"]))
        if expected:
            result = corrector.correct(initial, controls, duration)
            if not result.accepted or result.controls is None:
                raise RuntimeError(f"{benchmark} seed {seed} sample {index}: unexpected fallback")
            applied[index] = np.asarray(result.controls, float)
            changed[index] = np.asarray([segment.corrected for segment in result.segments], bool)
        observed = int(np.sum(changed[index]))
        if observed != expected:
            raise ValueError(
                f"{benchmark} seed {seed} sample {index}: changed {observed} != stored {expected}"
            )
    return {"nominal": nominal, "applied": applied, "changed": changed}


def _cache_cstr(seed: int) -> dict[str, np.ndarray]:
    cfg, states = _cstr_protocol()
    directory = _seed_dir("cstr", seed)
    nominal = _predict_cstr(directory, cfg, states)
    rows = _read_rows(
        CSTR_MARGIN_RESULTS
        / "linear_cosine"
        / "cstr"
        / f"seed{seed}"
        / "hds_test400_margin1e6"
        / f"test_per_sample_{METHOD}.csv"
    )
    applied = nominal.copy()
    changed = np.zeros(nominal.shape[:2], dtype=bool)
    for index, (state, controls, row) in enumerate(zip(states, nominal, rows)):
        corrected, mask = _cstr_correct_controls(state, controls, cfg)
        expected = int(float(row["corrected_segments"]))
        observed = int(np.sum(mask))
        if observed != expected:
            raise ValueError(f"cstr seed {seed} sample {index}: changed {observed} != stored {expected}")
        applied[index] = corrected
        changed[index] = mask
    return {"nominal": nominal, "applied": applied, "changed": changed}


def cache_seed(benchmark: str, seed: int, force: bool = False) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{benchmark}_seed{seed}_controls.npz"
    if target.exists() and not force:
        return f"{benchmark} seed {seed}: cached"
    data = _cache_cstr(seed) if benchmark == "cstr" else _cache_scalar(benchmark, seed)
    np.savez_compressed(
        target,
        nominal=np.asarray(data["nominal"], np.float32),
        applied=np.asarray(data["applied"], np.float32),
        changed=np.asarray(data["changed"], bool),
        sample_index=np.arange(400, dtype=np.int32),
        training_seed=np.full(400, seed, dtype=np.int32),
    )
    return f"{benchmark} seed {seed}: wrote {target.name}"


def cache_benchmark(benchmark: str, workers: int, force: bool) -> None:
    with ProcessPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(cache_seed, benchmark, seed, force): seed for seed in SEEDS}
        for future in as_completed(jobs):
            print(future.result(), flush=True)


def _population(benchmark: str) -> dict[str, np.ndarray]:
    caches = [np.load(OUT / f"{benchmark}_seed{seed}_controls.npz") for seed in SEEDS]
    if any(len(cache["sample_index"]) != 400 for cache in caches):
        raise ValueError(f"{benchmark}: incomplete seed cache")
    return {
        "nominal": np.concatenate([cache["nominal"] for cache in caches], axis=0),
        "applied": np.concatenate([cache["applied"] for cache in caches], axis=0),
        "changed": np.concatenate([cache["changed"] for cache in caches], axis=0),
        "training_seed": np.concatenate([cache["training_seed"] for cache in caches]),
    }


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def _step_path(controls: np.ndarray, duration: float) -> list[np.ndarray]:
    edges = np.arange(controls.shape[1] + 1, dtype=float) * duration
    x = np.repeat(edges, 2)[1:-1]
    return [np.column_stack((x, np.repeat(row, 2))) for row in controls]


def _changed_segments(
    applied: np.ndarray, changed: np.ndarray, duration: float
) -> np.ndarray:
    trajectory, segment = np.nonzero(changed)
    if len(trajectory) == 0:
        return np.empty((0, 2, 2), float)
    start = segment.astype(float) * duration
    stop = start + duration
    value = applied[trajectory, segment]
    return np.stack(
        (np.column_stack((start, value)), np.column_stack((stop, value))), axis=1
    )


def _draw_control(
    ax,
    nominal: np.ndarray,
    applied: np.ndarray,
    changed: np.ndarray,
    *,
    duration: float,
    bounds: tuple[float, float],
    ylabel: str,
    title: str,
    panel: str,
    nominal_color: str = "#6E8FA8",
    corrected_color: str = "#C75D44",
    show_panel: bool = True,
) -> dict[str, float]:
    from matplotlib.collections import LineCollection

    nominal_paths = _step_path(nominal, duration)
    population = LineCollection(
        nominal_paths,
        colors=nominal_color,
        linewidths=0.18,
        alpha=0.006,
        rasterized=True,
        zorder=1,
    )
    ax.add_collection(population)
    corrected = _changed_segments(applied, changed, duration)
    alpha = min(0.32, 8.0 / np.sqrt(max(len(corrected), 1)))
    if len(corrected):
        overlay = LineCollection(
            corrected,
            colors=corrected_color,
            linewidths=0.52,
            alpha=alpha,
            rasterized=True,
            zorder=3,
        )
        ax.add_collection(overlay)
    edges = np.arange(nominal.shape[1] + 1, dtype=float) * duration
    median = np.median(nominal, axis=0)
    ax.step(edges, np.r_[median, median[-1]], where="post", color="0.16", lw=0.85, zorder=4)
    for value in bounds:
        ax.axhline(value, color="0.45", ls=":", lw=0.75, zorder=2)
    pad = 0.07 * (bounds[1] - bounds[0])
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(bounds[0] - pad, bounds[1] + pad)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.grid(axis="x", color="0.91", lw=0.4, zorder=0)
    if show_panel:
        ax.text(-0.15, 1.04, panel, transform=ax.transAxes, fontweight="bold", va="bottom")
    count = int(np.sum(changed))
    total = int(changed.size)
    ax.text(
        0.98,
        0.04,
        f"modified: {count:,}/{total:,} segments\n({100.0 * count / total:.2f}%)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
        color=corrected_color,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.1},
    )
    return {"modified_segments": count, "total_segments": total, "modified_percent": 100.0 * count / total}


def render() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _style()
    vdp = _population("vdp")
    pen = _population("penicillin")
    cstr = _population("cstr")
    cfg, _ = _cstr_protocol()

    fig, axes = plt.subplots(1, 4, figsize=(7.25, 2.55), constrained_layout=False)
    ax_vdp, ax_pen, ax_ti, ax_flow = axes
    # Match the four-column geometry used by each row of the path-constraint
    # figure so every benchmark/input channel has the same physical panel size.
    for ax in axes:
        ax.set_box_aspect(0.72)
        ax.set_anchor("N")

    summaries: dict[str, dict[str, float]] = {}
    summaries["vdp"] = _draw_control(
        ax_vdp,
        vdp["nominal"],
        vdp["applied"],
        vdp["changed"],
        duration=0.5,
        bounds=(-0.3, 1.0),
        ylabel=r"Control $u$",
        title="VDP",
        panel="(a)",
    )
    summaries["penicillin"] = _draw_control(
        ax_pen,
        pen["nominal"],
        pen["applied"],
        pen["changed"],
        duration=PEN_DT,
        bounds=(0.0, PEN_UMAX),
        ylabel=r"Feed rate $u$",
        title="Penicillin",
        panel="(b)",
    )
    ti_summary = _draw_control(
        ax_ti,
        cstr["nominal"][:, :, 0],
        cstr["applied"][:, :, 0],
        cstr["changed"],
        duration=cfg.zoh_duration_s,
        bounds=cfg.ti_bounds_K,
        ylabel=r"$T_i$ (K)",
        title=r"CSTR: inlet $T_i$",
        panel="(c)",
    )
    flow_summary = _draw_control(
        ax_flow,
        cstr["nominal"][:, :, 1],
        cstr["applied"][:, :, 1],
        cstr["changed"],
        duration=cfg.zoh_duration_s,
        bounds=cfg.flow_bounds,
        ylabel=r"Flow $F$",
        title=r"CSTR: flow $F$",
        panel="(d)",
    )
    summaries["cstr"] = {
        "modified_segments": ti_summary["modified_segments"],
        "total_segments": ti_summary["total_segments"],
        "modified_percent": ti_summary["modified_percent"],
        "channels": 2,
    }
    if flow_summary["modified_segments"] != ti_summary["modified_segments"]:
        raise ValueError("CSTR channel masks differ")

    legend = [
        Line2D([0], [0], color="#6E8FA8", lw=0.7, alpha=0.65, label="All nominal CA-KKT controls"),
        Line2D([0], [0], color="#C75D44", lw=1.2, label=r"HDS--$\lambda$ modified segments"),
        Line2D([0], [0], color="0.16", lw=0.9, label="Nominal population median"),
        Line2D([0], [0], color="0.45", ls=":", lw=0.8, label="Control bounds"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.905),
        ncol=4,
        columnspacing=1.15,
        handlelength=2.1,
    )
    fig.suptitle(
        "Complete control populations with segment-local HDS--$\\lambda$ correction\n"
        "30 training seeds $\\times$ 400 matched test points per benchmark",
        y=0.995,
        fontsize=8.4,
    )
    fig.subplots_adjust(left=0.080, right=0.995, bottom=0.195, top=0.705, wspace=0.52)

    FIGURES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "linear_cosine_control_corrections_30seeds_all12000"
    svg_target = OUT / f"{stem}.svg"
    pdf_target = OUT / f"{stem}.pdf"
    png_target = OUT / f"{stem}.png"
    tiff_target = OUT / f"{stem}.tiff"
    fig.savefig(svg_target, bbox_inches="tight")
    fig.savefig(pdf_target, bbox_inches="tight")
    fig.savefig(png_target, dpi=600, bbox_inches="tight")
    fig.savefig(tiff_target, dpi=600, bbox_inches="tight")
    for target in (svg_target, pdf_target, png_target, tiff_target):
        shutil.copy2(target, FIGURES / target.name)
    plt.close(fig)

    summary = {
        "method": "cosine-adaptive KKT continuation (CA-KKT), 200+10 epochs",
        "training_seeds": list(SEEDS),
        "test_points_per_seed": 400,
        "trajectories_per_benchmark": 12000,
        "display_rule": "all nominal sequences; applied controls only on changed ZOH segments",
        "cstr_acceptance_rule": "max(g1,g2) <= -1e-6",
        "benchmarks": summaries,
        "hds_statement": "continuous-time numerical audit evidence under the declared model and numerical settings",
    }
    (OUT / "control_correction_population_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (OUT / "control_correction_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("benchmark", "trajectories", "modified_segments", "total_segments", "modified_percent"),
        )
        writer.writeheader()
        for benchmark, values in summaries.items():
            writer.writerow({
                "benchmark": benchmark,
                "trajectories": 12000,
                "modified_segments": values["modified_segments"],
                "total_segments": values["total_segments"],
                "modified_percent": values["modified_percent"],
            })
    (OUT / "README.md").write_text(
        "# Complete 30-seed CA-KKT control populations\n\n"
        "All 12,000 nominal ZOH sequences per benchmark are shown. The applied control is overlaid only "
        "on ZOH segments that were changed by the frozen adaptive DOP853/event-located HDS--lambda "
        "procedure. No trajectory, seed, test point, or control segment was excluded. Dense line layers "
        "are rasterized at 600 dpi in vector exports; text and axes remain editable.\n",
        encoding="utf-8",
    )
    print(FIGURES / f"{stem}.png", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("vdp", "penicillin", "cstr", "all"), default="all")
    parser.add_argument("--workers", type=int, default=12)
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
