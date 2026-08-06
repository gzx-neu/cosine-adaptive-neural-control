"""Build the 30-seed gradient-conflict figure and matched-solver timing table."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "kkt_collocation" / "results"
OUTPUT = RESULTS / "paper30_gradient_conflict_and_timing_20260803_v1"
SEEDS = tuple(range(20260771, 20260801))
METHODS = ("supervised", "unprocessed", "linear_cosine", "standard_pcgrad")
METHOD_LABELS = {
    "supervised": "S-u",
    "unprocessed": "Unprocessed KKT",
    "linear_cosine": "Linear cosine",
    "standard_pcgrad": "Standard PCGrad",
}
BENCHMARKS = ("VDP", "Penicillin", "CSTR")
LOG_SOURCES = {
    "VDP": (
        RESULTS / "multiseed10_vdp_k10_cuda_20260803_v1",
        RESULTS / "multiseed_vdp_k10_cuda_seeds21_30_20260803_v1",
    ),
    "Penicillin": (
        RESULTS / "multiseed10_penicillin_k10_cpu_20260803_v1",
        RESULTS / "multiseed_penicillin_k10_cpu_seeds21_30_20260803_v1",
    ),
    "CSTR": (
        RESULTS / "multiseed10_cstr_k10_cuda_20260803_v1",
        RESULTS / "multiseed_cstr_k10_cuda_seeds21_30_20260803_v1",
    ),
}
BENCHMARK_DIR = {"VDP": "vdp", "Penicillin": "penicillin", "CSTR": "cstr"}
AGGREGATES = {
    "VDP": RESULTS / "multiseed30_vdp_k10_cuda_20260803_v1" / "aggregate_30seeds_summary.json",
    "Penicillin": RESULTS / "multiseed30_penicillin_k10_cpu_20260803_v1" / "aggregate_30seeds_summary.json",
    "CSTR": RESULTS / "multiseed30_cstr_k10_cuda_20260803_v1" / "aggregate_30seeds_summary.json",
}
COLUMN_COLORS = {"gradient_cosine": "#6E8FA8", "projection_fraction_eta": "#C75D44"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def locate_log(benchmark: str, seed: int) -> Path:
    relative = (
        Path("linear_cosine") / BENCHMARK_DIR[benchmark] / f"seed{seed}"
        / "S-u+K_training_log.json"
    )
    matches = [root / relative for root in LOG_SOURCES[benchmark] if (root / relative).exists()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one linear-cosine log for {benchmark} seed {seed}: {matches}")
    return matches[0]


def extract_gradient_records() -> pd.DataFrame:
    rows = []
    for benchmark in BENCHMARKS:
        for seed in SEEDS:
            payload = read_json(locate_log(benchmark, seed))
            continuation = [row for row in payload["history"] if row["stage"] == "continuation"]
            if len(continuation) != 10 or [row["epoch"] for row in continuation] != list(range(1, 11)):
                raise RuntimeError(f"Continuation history invariant failed: {benchmark} seed {seed}")
            for row in continuation:
                cosine = float(row["kkt_conflict_cosine"])
                eta = float(row["kkt_projection_fraction_used"])
                expected_eta = max(0.0, -cosine)
                if not np.isclose(eta, expected_eta, atol=2e-7, rtol=0.0):
                    raise RuntimeError(
                        f"Projection rule mismatch: {benchmark} seed {seed} epoch {row['epoch']}"
                    )
                rows.append({
                    "benchmark": benchmark,
                    "seed": seed,
                    "continuation_epoch": int(row["epoch"]),
                    "gradient_cosine": cosine,
                    "projection_fraction_eta": eta,
                    "conflict": cosine < 0.0,
                })
    frame = pd.DataFrame(rows)
    expected = len(BENCHMARKS) * len(SEEDS) * 10
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} records, found {len(frame)}")
    return frame


def summarize_gradient_records(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def summarize_group(group: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n": len(group),
            "cosine_mean": group["gradient_cosine"].mean(),
            "cosine_sd": group["gradient_cosine"].std(ddof=1),
            "cosine_median": group["gradient_cosine"].median(),
            "cosine_q25": group["gradient_cosine"].quantile(0.25),
            "cosine_q75": group["gradient_cosine"].quantile(0.75),
            "conflict_rate": group["conflict"].mean(),
            "eta_mean": group["projection_fraction_eta"].mean(),
            "eta_sd": group["projection_fraction_eta"].std(ddof=1),
            "eta_median": group["projection_fraction_eta"].median(),
            "eta_q25": group["projection_fraction_eta"].quantile(0.25),
            "eta_q75": group["projection_fraction_eta"].quantile(0.75),
        })

    by_epoch = (
        frame.groupby(["benchmark", "continuation_epoch"], sort=False)
        .apply(summarize_group, include_groups=False).reset_index()
    )
    overall = (
        frame.groupby("benchmark", sort=False)
        .apply(summarize_group, include_groups=False).reset_index()
    )
    return by_epoch, overall


def build_timing_table() -> pd.DataFrame:
    jiang = read_json(RESULTS / "jiang_fu_matched400_comparison" / "summary.json")
    jiang_points = pd.read_csv(
        RESULTS / "jiang_fu_matched400_comparison" / "per_point_seed.csv"
    )
    cstr_reference = read_json(
        RESULTS / "economou_cstr_reduced_kkt_n100_test400_lhs_margin0" / "summary.json"
    )
    cstr_solve_times = np.asarray([
        float(json.loads(line)["solve_seconds"])
        for line in (
            RESULTS / "economou_cstr_reduced_kkt_n100_test400_lhs_margin0" / "records.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ])
    deterministic = {
        "VDP": {
            "seconds": float(jiang["problems"]["VDP"]["jiang_solve_seconds"]["mean"]),
            "sd": float(jiang_points.loc[
                jiang_points["problem"] == "VDP", "jiang_solve_seconds"
            ].std(ddof=1)),
            "solver": "Jiang--Fu Algorithm 1, MATLAB cold start",
        },
        "Penicillin": {
            "seconds": float(jiang["problems"]["Penicillin"]["jiang_solve_seconds"]["mean"]),
            "sd": float(jiang_points.loc[
                jiang_points["problem"] == "Penicillin", "jiang_solve_seconds"
            ].std(ddof=1)),
            "solver": "Jiang--Fu Algorithm 1, MATLAB cold start",
        },
        "CSTR": {
            "seconds": float(cstr_reference["mean_solve_seconds_successful"]),
            "sd": float(cstr_solve_times.std(ddof=1)),
            "solver": "Cold-start reduced-space RK4 control-vector NLP (N=100, RK10)",
        },
    }
    rows = []
    for benchmark in BENCHMARKS:
        report = read_json(AGGREGATES[benchmark])
        for method in METHODS:
            item = report["methods"][method]
            inference_ms = float(item["mean_inference_ms"]["mean"])
            inference_sd_ms = float(item["mean_inference_ms"]["sample_sd"])
            hds_ms = float(item["mean_hds_ms"]["mean"])
            hds_sd_ms = float(item["mean_hds_ms"]["sample_sd"])
            total_ms = float(item["mean_total_ms"]["mean"])
            total_sd_ms = float(item["mean_total_ms"]["sample_sd"])
            reference_seconds = deterministic[benchmark]["seconds"]
            rows.append({
                "benchmark": benchmark,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "inference_ms_per_point": inference_ms,
                "inference_seed_sd_ms": inference_sd_ms,
                "hds_ms_per_point": hds_ms,
                "hds_seed_sd_ms": hds_sd_ms,
                "total_predeployment_ms_per_point": total_ms,
                "total_predeployment_seed_sd_ms": total_sd_ms,
                "deterministic_solver": deterministic[benchmark]["solver"],
                "deterministic_cold_start_seconds_per_point": reference_seconds,
                "deterministic_cold_start_point_sd_seconds": deterministic[benchmark]["sd"],
                "recorded_speedup": reference_seconds / (total_ms / 1000.0),
            })
    return pd.DataFrame(rows)


def style_violin(parts: dict, color: str) -> None:
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
        body.set_linewidth(0.6)
    if "cmedians" in parts:
        parts["cmedians"].set_color("#202020")
        parts["cmedians"].set_linewidth(1.0)


def plot_gradient_figure(frame: pd.DataFrame, overall: pd.DataFrame) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 3.55), sharex=True, constrained_layout=True)
    rng = np.random.default_rng(20260803)
    for column_index, benchmark in enumerate(BENCHMARKS):
        subset = frame[frame["benchmark"] == benchmark]
        stats = overall[overall["benchmark"] == benchmark].iloc[0]
        for row_index, (field, ylabel) in enumerate((
            ("gradient_cosine", r"Gradient cosine $\cos(g_B,g_K)$"),
            ("projection_fraction_eta", r"Applied projection fraction $\eta$"),
        )):
            ax = axes[row_index, column_index]
            ax.set_box_aspect(0.72)
            ax.set_anchor("N")
            color = COLUMN_COLORS[field]
            values = [
                subset.loc[subset["continuation_epoch"] == epoch, field].to_numpy(float)
                for epoch in range(1, 11)
            ]
            parts = ax.violinplot(
                values, positions=np.arange(1, 11), widths=0.72,
                showmeans=False, showmedians=True, showextrema=False,
            )
            style_violin(parts, color)
            for epoch, epoch_values in enumerate(values, start=1):
                jitter = rng.uniform(-0.16, 0.16, size=len(epoch_values))
                ax.scatter(
                    epoch + jitter, epoch_values, s=5.5, color=color,
                    alpha=0.48, linewidths=0, rasterized=True,
                )
            ax.grid(axis="y", color="#E4E4E4", linewidth=0.4, alpha=0.65)
            ax.set_xlim(0.45, 10.55)
            ax.set_xticks(range(1, 11))
            if column_index == 0:
                ax.set_ylabel(ylabel)
            if row_index == 1:
                ax.set_xlabel("KKT refinement epoch")
                ax.tick_params(axis="x", labelbottom=True)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if field == "gradient_cosine":
                ax.axhline(0.0, color="#5A5A5A", linewidth=0.8, linestyle="--", zorder=0)
                ax.set_ylim(-1.05, 1.05)
                annotation = (
                    f"median={stats['cosine_median']:.2f}\n"
                    f"conflict={100.0 * stats['conflict_rate']:.0f}%"
                )
            else:
                ax.set_ylim(-0.035, 1.035)
                annotation = rf"median $\eta$={stats['eta_median']:.2f}"
            annotation_y = 0.96
            if field == "projection_fraction_eta" and benchmark in ("Penicillin", "CSTR"):
                annotation_y = 0.84
            ax.text(
                0.98, annotation_y, annotation, transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5, color="#303030",
            )
            letter = "abcdef"[row_index * len(BENCHMARKS) + column_index]
            title = f"{benchmark}: " + ("gradient conflict" if row_index == 0 else "adaptive projection")
            ax.set_title(title, loc="left", fontweight="bold", pad=4)
            ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight="bold")

    fig.suptitle(
        "Conflict geometry during 10-epoch discrete-KKT refinement\n"
        "30 seeds per benchmark; points show individual seeds",
        fontsize=8.2, fontweight="normal",
    )
    stem = OUTPUT / "gradient_cosine_projection_distribution_30seeds"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def write_report(overall: pd.DataFrame, timing: pd.DataFrame) -> None:
    linear = timing[timing["method"] == "linear_cosine"].set_index("benchmark")
    lines = [
        "# 30-seed gradient-conflict mechanism and matched-solver timing", "",
        "## Gradient mechanism summary", "",
        "| Benchmark | Gradient cosine median | Conflict rate | Projection fraction median | Records |",
        "|---|---:|---:|---:|---:|",
    ]
    for benchmark in BENCHMARKS:
        item = overall[overall["benchmark"] == benchmark].iloc[0]
        lines.append(
            f"| {benchmark} | {item['cosine_median']:.4f} | "
            f"{100.0 * item['conflict_rate']:.2f}% | {item['eta_median']:.4f} | {int(item['n'])} |"
        )
    lines.extend([
        "", "Each benchmark contributes 30 seeds x 10 refinement epochs = 300 observed gradient pairs. "
        "No seed or epoch was excluded. The recorded projection fraction satisfies "
        "eta=max(0,-cos(g_B,g_K)) for every observation.", "",
        "## Linear-cosine deployment timing versus matched deterministic cold start", "",
        "| Benchmark | Inference (ms/point) | HDS audit + correction (ms/point) | Total predeployment (ms/point) | Deterministic cold start (s/point) | Recorded speedup |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for benchmark in BENCHMARKS:
        item = linear.loc[benchmark]
        lines.append(
            f"| {benchmark} | {item['inference_ms_per_point']:.4f} +/- {item['inference_seed_sd_ms']:.4f} | "
            f"{item['hds_ms_per_point']:.4f} +/- {item['hds_seed_sd_ms']:.4f} | "
            f"{item['total_predeployment_ms_per_point']:.4f} +/- {item['total_predeployment_seed_sd_ms']:.4f} | "
            f"{item['deterministic_cold_start_seconds_per_point']:.4f} +/- "
            f"{item['deterministic_cold_start_point_sd_seconds']:.4f} | "
            f"{item['recorded_speedup']:.1f}x |"
        )
    lines.extend([
        "", "## All-method deployment timing", "",
        "| Benchmark | Method | Inference (ms/point) | HDS audit + correction (ms/point) | Total predeployment (ms/point) | Recorded speedup |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for benchmark in BENCHMARKS:
        for _, item in timing[timing["benchmark"] == benchmark].iterrows():
            lines.append(
                f"| {benchmark} | {item['method_label']} | "
                f"{item['inference_ms_per_point']:.4f} +/- {item['inference_seed_sd_ms']:.4f} | "
                f"{item['hds_ms_per_point']:.4f} +/- {item['hds_seed_sd_ms']:.4f} | "
                f"{item['total_predeployment_ms_per_point']:.4f} +/- "
                f"{item['total_predeployment_seed_sd_ms']:.4f} | "
                f"{item['recorded_speedup']:.1f}x |"
            )
    lines.extend([
        "", "VDP and penicillin use the frozen matched-400 Jiang--Fu MATLAB cold-start records. "
        "CSTR uses the independent cold-start reduced-space RK4 control-vector NLP (N=100, RK10), "
        "where states are eliminated by forward propagation. Training time is excluded from speedup. "
        "Deployment-time +/- values are sample standard deviations across 30 seed-level mean timings; "
        "deterministic-solver +/- values are sample standard deviations across the 400 matched initial conditions. "
        "Speedup is the ratio of the two recorded means and is therefore not assigned a +/- value. "
        "These are recorded wall-clock comparisons under the stated implementations and hardware, "
        "not hardware-independent complexity claims.", "",
        "HDS results are continuous-time numerical audit evidence under the declared model and numerical settings; "
        "they are not an absolute physical-system safety guarantee.", "",
        "## Figure legend", "",
        "**Gradient conflict during adaptive discrete-KKT refinement.** Panels a--c show "
        "the cosine between the supervised-plus-anchor gradient g_B and weighted KKT gradient g_K "
        "at each of ten refinement epochs for VDP, penicillin and CSTR, respectively. Panels d--f "
        "show the applied projection fraction eta=max(0,-cos(g_B,g_K)). Each point is one "
        "independent training seed (n=30 seeds per epoch); violins show the full across-seed distribution "
        "and black bars denote medians. The dashed line marks zero gradient cosine. No observations were excluded.", "",
        "## QA notes", "",
        "- Quantitative-grid archetype; Python/matplotlib was used exclusively for drawing and export.",
        "- Source-data integrity: 900/900 expected epoch-seed observations were found and retained.",
        "- The random-number generator is used only for horizontal display jitter; it does not simulate, "
        "resample, transform, or exclude any measured value.",
        "- Source preflight: 13 PASS, 0 FAIL, 1 reviewed WARN. The warning concerns the display-jitter RNG "
        "and is not simulated data.",
        "- Exports: editable SVG/PDF, 600-dpi LZW TIFF, and 300-dpi PNG preview at 181.6-mm width.",
    ])
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = extract_gradient_records()
    by_epoch, overall = summarize_gradient_records(records)
    timing = build_timing_table()
    records.to_csv(OUTPUT / "gradient_projection_records.csv", index=False)
    by_epoch.to_csv(OUTPUT / "gradient_projection_by_epoch_summary.csv", index=False)
    overall.to_csv(OUTPUT / "gradient_projection_overall_summary.csv", index=False)
    timing.to_csv(OUTPUT / "deployment_timing_all_methods.csv", index=False)
    plot_gradient_figure(records, overall)
    write_report(overall, timing)
    print(overall.to_string(index=False))
    print(timing[timing["method"] == "linear_cosine"].to_string(index=False))


if __name__ == "__main__":
    main()
