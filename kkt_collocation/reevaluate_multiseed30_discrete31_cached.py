"""Re-evaluate all reproduced 30-seed policies with cached discrete-grid HDS.

Protocol:

* VDP CVP10, penicillin CVP10, and CSTR CVP100;
* seeds 20260771--20260800, four frozen methods, 400 matched points;
* adaptive DOP853 and derivative-zero event extrema;
* a 31-point lambda base grid, closest-to-one first-safe early stop;
* no lambda bisection;
* acceptance requires g_max <= -1e-8;
* terminal states and peaks from every accepted segment propagation are reused;
  the corrected sequence is not propagated a redundant second time.

No network is retrained by this command and no deterministic reference is
recomputed. Checkpoints are read from ``reproduced_results`` by default; the
archived aggregate output can be verified without checkpoints via
``scripts/verify_bundle.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "kkt_collocation" / "results"
DEFAULT_OUT = RESULTS / "formal_multiseed30_discrete31_cached_margin1e8_20260806_v1"
DEFAULT_CHECKPOINT_ROOT = ROOT / "reproduced_results"
SEEDS = tuple(range(20260771, 20260801))
METHODS = ("supervised", "unprocessed", "linear_cosine", "standard_pcgrad")
BRANCH = {"supervised": "S-u", "unprocessed": "S-u+K", "linear_cosine": "S-u+K", "standard_pcgrad": "S-u+K"}
GRID = 31
# Formal paper value is -1e-8.
THRESHOLD = float(os.environ.get("CAKKT_HDS_THRESHOLD", "-1e-8"))

from kkt_collocation.benchmark_hds_segment_cache_grid31_small import cstr_correct, scalar_correct
from kkt_collocation.compare_lambda_discrete31_vs51_small import scalar_corrector
from kkt_collocation.run_two_stage_vs_kkt_only_ablation import Problem
from kkt_collocation.run_unified_su_suj_sk_konly_ablation import UnifiedConfig
from kkt_collocation.reevaluate_cstr_margin1e6_30seeds import _load_checkpoint, _load_protocol, _predict as predict_cstr
from kkt_collocation.run_economou_cstr_supervised_hds import trajectory_objective


def train_dir(checkpoint_root: Path, benchmark: str, method: str, seed: int) -> Path:
    """Return the portable layout produced by scripts/reproduce.py train."""
    return checkpoint_root / benchmark / method / benchmark / f"seed{seed}"


def cstr_references() -> list[dict[str, str]]:
    path = RESULTS / "economou_cstr_n100_test400_hds_s200_sk10_fair" / "cold_reference_high_fidelity.csv"
    return read_rows(path, "index")


def read_rows(path: Path, index_field: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row[index_field]))
    if len(rows) != 400 or [int(row[index_field]) for row in rows] != list(range(400)):
        raise ValueError(f"Incomplete frozen source rows: {path}")
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def as_float(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def as_bool(row: dict, key: str) -> bool:
    return str(row.get(key, "false")).strip().lower() == "true"


def load_scalar_model(problem: Problem, path: Path) -> torch.nn.Module:
    checkpoint = _load_checkpoint(path)
    model = problem.make_model().eval()
    model.load_state_dict(checkpoint["model"])
    return model


def evaluate_scalar_job(benchmark: str, method: str, seed: int, output_root: str, checkpoint_root: str) -> dict:
    torch.set_num_threads(1)
    output = Path(output_root) / benchmark / method / f"seed{seed}"
    summary_path = output / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    problem = Problem(benchmark, torch.device("cpu"), seed, UnifiedConfig())
    train = train_dir(Path(checkpoint_root), benchmark, method, seed)
    model = load_scalar_model(problem, train / f"{BRANCH[method]}.pth")
    controls, inference_seconds = problem.predict(model, problem.test_states)
    corrector = scalar_corrector(benchmark, GRID)
    rows = []
    for index, (value, nominal) in enumerate(zip(problem.test_states, controls)):
        initial = problem.initial_state(value)
        nominal_peak = float(corrector.audit(initial, nominal, problem.duration))
        nominal_objective = problem.cost(value, nominal, corrector)
        started = time.perf_counter()
        result = scalar_correct(corrector, initial, nominal, problem.duration, final_reaudit=False, threshold=THRESHOLD)
        hds_seconds = time.perf_counter() - started
        accepted = bool(result["accepted"])
        applied = result["controls"]
        applied_objective = problem.cost(value, applied, corrector) if accepted else np.nan
        reference_record = problem.reference_records[index]
        reference = float(reference_record["objective"])
        denominator = max(abs(reference), 1e-12)
        rows.append({
            "benchmark": benchmark, "method_key": method, "method": BRANCH[method],
            "training_seed": seed, "sample_index": index,
            "accepted": accepted, "fallback": not accepted,
            "nominal_hds_max_g": nominal_peak,
            "applied_hds_max_g": float(result["final_peak"]),
            "nominal_objective": nominal_objective, "applied_objective": applied_objective,
            "nlp_reference_objective": reference,
            "nominal_relative_objective_gap": (nominal_objective - reference) / denominator,
            "applied_relative_objective_gap": (applied_objective - reference) / denominator if accepted else np.nan,
            "objective_change": applied_objective - nominal_objective if accepted else np.nan,
            "corrected_segments": int(result["corrected_segments"]),
            "candidate_evaluations": int(result["candidate_evaluations"]),
            "inference_seconds": inference_seconds, "hds_correction_seconds": hds_seconds,
            "total_seconds": inference_seconds + hds_seconds,
            "cold_reference_solve_seconds": float(reference_record["solve_seconds"]),
            "cold_reference_hds_gmax": float(reference_record["hds_gmax"]),
            "lambda_grid_size": GRID, "lambda_bisection": False,
            "acceptance_threshold": THRESHOLD, "segment_cache_reused": True,
            "final_reaudit_performed": False,
        })
    write_rows(output / f"test_per_sample_{BRANCH[method]}.csv", rows)
    accepted = [row for row in rows if row["accepted"]]
    summary = {
        "benchmark": benchmark, "method": method, "branch": BRANCH[method], "seed": seed,
        "samples": 400, "accepted": len(accepted), "fallback": 400 - len(accepted),
        "hds_gap_percent": float(100.0 * np.mean([row["applied_relative_objective_gap"] for row in accepted])),
        "nominal_gap_percent": float(100.0 * np.mean([row["nominal_relative_objective_gap"] for row in rows])),
        "nominal_violation_rate_percent": float(100.0 * np.mean([row["nominal_hds_max_g"] > 0 for row in rows])),
        "post_hds_violation_rate_percent": float(100.0 * np.mean([row["applied_hds_max_g"] > THRESHOLD for row in accepted])),
        "final_max_g": float(max(row["applied_hds_max_g"] for row in accepted)),
        "mean_corrected_segments": float(np.mean([row["corrected_segments"] for row in accepted])),
        "mean_hds_objective_change": float(np.mean([row["objective_change"] for row in accepted])),
        "mean_inference_seconds": float(np.mean([row["inference_seconds"] for row in rows])),
        "mean_hds_seconds": float(np.mean([row["hds_correction_seconds"] for row in rows])),
        "mean_total_seconds": float(np.mean([row["total_seconds"] for row in rows])),
        "source_checkpoint": str(train / f"{BRANCH[method]}.pth"),
        "source_metadata_rows": str(RESULTS / "jiang_fu_matched400_comparison" / "per_point_seed.csv"),
        "hds": {"grid_size": GRID, "bisection": False, "threshold": THRESHOLD, "segment_cache_reused": True, "final_reaudit_performed": False},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_cstr_job(method: str, seed: int, output_root: str, checkpoint_root: str) -> dict:
    torch.set_num_threads(1)
    benchmark = "cstr"
    output = Path(output_root) / benchmark / method / f"seed{seed}"
    summary_path = output / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    cfg, states = _load_protocol()
    train = train_dir(Path(checkpoint_root), benchmark, method, seed)
    controls, inference_seconds = predict_cstr(train, BRANCH[method], cfg, states)
    sources = cstr_references()
    rows = []
    for index, (initial, nominal, source) in enumerate(zip(states, controls, sources)):
        nominal_peak = cstr_audit(initial, nominal, cfg)
        nominal_objective = trajectory_objective(initial, nominal, cfg)
        started = time.perf_counter()
        result = cstr_correct(initial, nominal, cfg, final_reaudit=False, threshold=THRESHOLD)
        hds_seconds = time.perf_counter() - started
        accepted = bool(result["accepted"])
        applied_objective = trajectory_objective(initial, result["controls"], cfg) if accepted else np.nan
        reference = as_float(source, "high_fidelity_objective")
        denominator = max(abs(reference), 1e-12)
        qualified = as_bool(source, "continuous_audit_qualified")
        gap = 100.0 * (applied_objective - reference) / denominator if accepted else np.nan
        nominal_gap = 100.0 * (nominal_objective - reference) / denominator
        rows.append({
            "benchmark": benchmark, "method_key": method, "method": BRANCH[method],
            "training_seed": seed, "index": index, "initial_state": json.dumps(np.asarray(initial, float).tolist()),
            "accepted": accepted, "fallback": not accepted,
            "nominal_max_g": nominal_peak, "final_max_g": float(result["final_peak"]),
            "nominal_objective": nominal_objective, "hds_objective": applied_objective,
            "reference_high_fidelity_objective": reference,
            "reference_continuous_audit_qualified": qualified,
            "relative_nominal_gap_percent": nominal_gap,
            "relative_hds_gap_percent": gap,
            "relative_nominal_gap_qualified_reference_percent": nominal_gap if qualified else np.nan,
            "relative_hds_gap_qualified_reference_percent": gap if qualified and accepted else np.nan,
            "objective_change": applied_objective - nominal_objective if accepted else np.nan,
            "corrected_segments": int(result["corrected_segments"]),
            "candidate_evaluations": int(result["candidate_evaluations"]),
            "inference_seconds": inference_seconds, "hds_seconds": hds_seconds,
            "total_predeployment_seconds": inference_seconds + hds_seconds,
            "lambda_grid_size": GRID, "lambda_bisection": False,
            "acceptance_threshold": THRESHOLD, "segment_cache_reused": True,
            "final_reaudit_performed": False,
        })
    write_rows(output / f"test_per_sample_{BRANCH[method]}.csv", rows)
    accepted = [row for row in rows if row["accepted"]]
    qualified = [row for row in accepted if row["reference_continuous_audit_qualified"]]
    summary = {
        "benchmark": benchmark, "method": method, "branch": BRANCH[method], "seed": seed,
        "samples": 400, "qualified_reference_samples": len(qualified),
        "accepted": len(accepted), "fallback": 400 - len(accepted),
        "hds_gap_percent": float(np.mean([row["relative_hds_gap_percent"] for row in accepted])),
        "hds_gap_qualified286_percent": float(np.mean([row["relative_hds_gap_qualified_reference_percent"] for row in qualified])),
        "nominal_gap_percent": float(np.mean([row["relative_nominal_gap_percent"] for row in rows])),
        "nominal_violation_rate_percent": float(100.0 * np.mean([row["nominal_max_g"] > 0 for row in rows])),
        "post_hds_violation_rate_percent": float(100.0 * np.mean([row["final_max_g"] > THRESHOLD for row in accepted])),
        "final_max_g": float(max(row["final_max_g"] for row in accepted)),
        "mean_corrected_segments": float(np.mean([row["corrected_segments"] for row in accepted])),
        "mean_hds_objective_change": float(np.mean([row["objective_change"] for row in accepted])),
        "mean_inference_seconds": float(np.mean([row["inference_seconds"] for row in rows])),
        "mean_hds_seconds": float(np.mean([row["hds_seconds"] for row in rows])),
        "mean_total_seconds": float(np.mean([row["total_predeployment_seconds"] for row in rows])),
        "source_checkpoint": str(train / f"{BRANCH[method]}.pth"),
        "source_metadata_rows": str(RESULTS / "economou_cstr_n100_test400_hds_s200_sk10_fair" / "cold_reference_high_fidelity.csv"),
        "hds": {"grid_size": GRID, "bisection": False, "threshold": THRESHOLD, "segment_cache_reused": True, "final_reaudit_performed": False},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def cstr_audit(initial: np.ndarray, controls: np.ndarray, cfg) -> float:
    """Audit an unmodified CSTR sequence for the nominal-policy metric."""
    from kkt_collocation.economou_cstr_hds_fast import segment_event_audit

    state = np.asarray(initial, float).copy()
    peak = -np.inf
    for control in controls:
        local, state, _ = segment_event_audit(state, control, cfg)
        peak = max(peak, float(np.max(local)))
    return float(peak)


def evaluate_job(job: tuple[str, str, int, str, str]) -> dict:
    benchmark, method, seed, output_root, checkpoint_root = job
    if benchmark == "cstr":
        return evaluate_cstr_job(method, seed, output_root, checkpoint_root)
    return evaluate_scalar_job(benchmark, method, seed, output_root, checkpoint_root)


def preflight_checkpoints(checkpoint_root: Path, benchmarks: list[str]) -> None:
    missing = []
    for benchmark in benchmarks:
        for method in METHODS:
            for seed in SEEDS:
                path = train_dir(checkpoint_root, benchmark, method, seed) / f"{BRANCH[method]}.pth"
                if not path.is_file():
                    missing.append(path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:8])
        raise FileNotFoundError(
            f"Missing {len(missing)} reproduced checkpoints under {checkpoint_root}.\n"
            "Run `python scripts/reproduce.py train --benchmarks ...` first, or pass "
            "--checkpoint-root pointing to the portable benchmark/method/benchmark/seed layout.\n"
            f"First missing paths:\n{preview}"
        )


def sample_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, float)
    return {"mean": float(array.mean()), "sample_sd": float(array.std(ddof=1)), "median": float(np.median(array)), "min": float(array.min()), "max": float(array.max())}


def aggregate(summaries: list[dict], output: Path) -> dict:
    report = {
        "formal_protocol": bool(np.isclose(THRESHOLD, -1e-8)),
        "protocol": "30 frozen seeds, four methods, matched 400 points, discrete-only 31-point-base-grid cached HDS",
        "seeds": list(SEEDS), "grid_size": GRID, "bisection": False,
        "acceptance_threshold": THRESHOLD, "segment_cache_reused": True,
        "final_reaudit_performed": False, "benchmarks": {}, "paired": {},
        "timing_note": "Batch timers were collected under concurrent CPU load and are throughput diagnostics, not serial online latency.",
    }
    per_seed_rows = []
    for item in sorted(summaries, key=lambda x: (x["benchmark"], x["method"], x["seed"])):
        per_seed_rows.append({key: value for key, value in item.items() if not isinstance(value, dict)})
    write_rows(output / "per_seed_summary.csv", per_seed_rows)
    for benchmark in ("vdp", "penicillin", "cstr"):
        report["benchmarks"][benchmark] = {}
        for method in METHODS:
            group = [row for row in summaries if row["benchmark"] == benchmark and row["method"] == method]
            if len(group) != 30:
                raise ValueError(f"Expected 30 summaries for {benchmark}/{method}, got {len(group)}")
            metrics = {
                "hds_gap_percent": sample_stats([row["hds_gap_percent"] for row in group]),
                "nominal_gap_percent": sample_stats([row["nominal_gap_percent"] for row in group]),
                "nominal_violation_rate_percent": sample_stats([row["nominal_violation_rate_percent"] for row in group]),
                "post_hds_violation_rate_percent": sample_stats([row["post_hds_violation_rate_percent"] for row in group]),
                "mean_corrected_segments": sample_stats([row["mean_corrected_segments"] for row in group]),
                "mean_hds_ms": sample_stats([1e3 * row["mean_hds_seconds"] for row in group]),
                "mean_total_ms": sample_stats([1e3 * row["mean_total_seconds"] for row in group]),
                "accepted_total": int(sum(row["accepted"] for row in group)),
                "fallback_total": int(sum(row["fallback"] for row in group)),
                "final_max_g": float(max(row["final_max_g"] for row in group)),
            }
            if benchmark == "cstr":
                metrics["hds_gap_qualified286_percent"] = sample_stats([row["hds_gap_qualified286_percent"] for row in group])
            report["benchmarks"][benchmark][method] = metrics
        by_method = {method: {row["seed"]: row for row in summaries if row["benchmark"] == benchmark and row["method"] == method} for method in METHODS}
        report["paired"][benchmark] = {}
        for left, right in (("linear_cosine", "supervised"), ("linear_cosine", "unprocessed"), ("linear_cosine", "standard_pcgrad")):
            difference = np.asarray([by_method[left][seed]["hds_gap_percent"] - by_method[right][seed]["hds_gap_percent"] for seed in SEEDS])
            report["paired"][benchmark][f"{left}_minus_{right}"] = {"mean_percentage_points": float(difference.mean()), "left_wins": int(np.sum(difference < 0)), "right_wins": int(np.sum(difference > 0)), "ties": int(np.sum(difference == 0))}
    (output / "aggregate_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        f"# Cached discrete-31 HDS: {'formal' if report['formal_protocol'] else 'nonformal threshold sensitivity'} 30-seed matched-400 results", "",
        f"All values use 31 closest-to-one candidates without bisection, adaptive DOP853 event audits, segment propagation reuse, and g_max <= {THRESHOLD:.1e}.", "",
        "| Benchmark | Method | HDS gap (%; mean +/- seed SD) | Corrected segments | Nominal violation | Post-HDS violation | Accepted |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for benchmark, methods in report["benchmarks"].items():
        for method, value in methods.items():
            gap = value["hds_gap_percent"]; seg = value["mean_corrected_segments"]; violation = value["nominal_violation_rate_percent"]
            lines.append(f"| {benchmark} | {method} | {gap['mean']:.5f} +/- {gap['sample_sd']:.5f} | {seg['mean']:.4f} | {violation['mean']:.2f}% | {value['post_hds_violation_rate_percent']['mean']:.2f}% | {value['accepted_total']}/12000 |")
    lines += ["", "CSTR qualified-286 gap:", "", "| Method | Gap (%; mean +/- seed SD) |", "|---|---:|"]
    for method, value in report["benchmarks"]["cstr"].items():
        gap = value["hds_gap_qualified286_percent"]
        lines.append(f"| {method} | {gap['mean']:.5f} +/- {gap['sample_sd']:.5f} |")
    lines += ["", report["timing_note"]]
    (output / "aggregate_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--benchmarks", nargs="+", choices=("vdp", "penicillin", "cstr"), default=("vdp", "penicillin", "cstr"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Output exists; pass --resume: {args.output}")
    checkpoint_root = args.checkpoint_root.resolve()
    preflight_checkpoints(checkpoint_root, list(args.benchmarks))
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [
        (benchmark, method, seed, str(args.output), str(checkpoint_root))
        for benchmark in args.benchmarks for method in METHODS for seed in SEEDS
    ]
    summaries = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            summary = future.result()
            summaries.append(summary)
            completed += 1
            print(f"completed {completed}/{len(jobs)}: {job[0]} {job[1]} seed{job[2]}", flush=True)
    if set(args.benchmarks) == {"vdp", "penicillin", "cstr"}:
        aggregate(summaries, args.output)
        print(args.output / "aggregate_table.md")
    else:
        (args.output / "partial_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
