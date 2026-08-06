"""Re-evaluate the frozen 30-seed CSTR ablation with g_max <= -1e-6.

No model is retrained and no cold-start reference is recomputed.  Rows whose
previously applied event-located maximum already satisfies -1e-6 are reused:
the closest-to-one candidate decision is provably unchanged. All remaining
rows are freshly audited with the unchanged 31-point candidate set and order.
Within each trajectory, accepted segment terminal states and peaks are cached;
there is no redundant final replay of the corrected sequence.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kkt_collocation"))

from kkt_collocation.economou_cstr_hds_fast import candidates, segment_event_audit  # noqa: E402
from kkt_collocation.run_economou_cstr_supervised_hds import PolicyValue, trajectory_objective  # noqa: E402
from kkt_collocation.screen_economou_cstr_30x30 import EconomouScreenConfig  # noqa: E402


RESULTS = ROOT / "kkt_collocation" / "results"
OUT = Path(os.environ.get(
    "CAKKT_CSTR_MARGIN_OUT",
    RESULTS / "multiseed30_cstr_k10_margin1e6_20260803_v1",
)).resolve()
SOURCE_OVERRIDE = os.environ.get("CAKKT_CSTR_SOURCE_ROOT")
SEEDS = tuple(range(20260771, 20260801))
THRESHOLD = -1e-6
GRID = 31
METHODS = {
    "supervised": "S-u",
    "unprocessed": "S-u+K",
    "linear_cosine": "S-u+K",
    "standard_pcgrad": "S-u+K",
}


def _source_root(seed: int) -> Path:
    if SOURCE_OVERRIDE:
        return Path(SOURCE_OVERRIDE).resolve()
    return RESULTS / (
        "multiseed10_cstr_k10_cuda_20260803_v1"
        if seed <= 20260790
        else "multiseed_cstr_k10_cuda_seeds21_30_20260803_v1"
    )


def _train_dir(method: str, seed: int) -> Path:
    return _source_root(seed) / method / "cstr" / f"seed{seed}"


def _load_protocol() -> tuple[EconomouScreenConfig, np.ndarray]:
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
    test = np.asarray([row["initial_state"] for row in rows], float)
    if cfg.zoh_steps != 100 or cfg.substeps_per_zoh != 10 or cfg.node_margin != 0 or len(test) != 400:
        raise ValueError("Unexpected frozen CSTR protocol")
    return cfg, test


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _predict(train: Path, branch: str, cfg: EconomouScreenConfig, test: np.ndarray) -> tuple[np.ndarray, float]:
    checkpoint = _load_checkpoint(train / f"{branch}.pth")
    model = PolicyValue(cfg).eval()
    model.load_state_dict(checkpoint["model"])
    mean = np.asarray(checkpoint["state_mean"], float)
    std = np.asarray(checkpoint["state_std"], float)
    inputs = torch.tensor((test[:, [0, 2]] - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(10):
            model(inputs)
        started = time.perf_counter()
        _, controls = model(inputs)
        inference = (time.perf_counter() - started) / len(test)
    return controls.numpy(), inference


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["index"]))
    if len(rows) != 400 or [int(row["index"]) for row in rows] != list(range(400)):
        raise ValueError(f"Incomplete old evaluation: {path}")
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _nominal_peak(state0: np.ndarray, controls: np.ndarray, cfg: EconomouScreenConfig) -> float:
    state = np.asarray(state0, float).copy()
    peak = -np.inf
    for control in controls:
        local, state, _ = segment_event_audit(state, control, cfg)
        peak = max(peak, float(np.max(local)))
    return float(peak)


def correct_with_threshold(
    state0: np.ndarray, controls: np.ndarray, cfg: EconomouScreenConfig, threshold: float
) -> dict:
    state = np.asarray(state0, float).copy()
    output = np.asarray(controls, float).copy()
    raw_peak = -np.inf
    applied_peak = -np.inf
    changed = 0
    lambdas: list[float] = []
    for index in range(cfg.zoh_steps):
        peaks, next_state, _ = segment_event_audit(state, output[index], cfg)
        raw_peak = max(raw_peak, float(peaks.max()))
        if peaks.max() <= threshold:
            state = next_state
            applied_peak = max(applied_peak, float(peaks.max()))
            lambdas.append(1.0)
            continue
        low, span, normalized, order = candidates(output[index], cfg, GRID)
        found = None
        # The nominal propagation above has already established lambda=1 as
        # unsafe, so repeating that identical candidate cannot change the result.
        for lam in order:
            if abs(float(lam) - 1.0) <= 1e-14:
                continue
            candidate = low + np.clip(lam * normalized, 0.0, 1.0) * span
            candidate_peaks, candidate_state, _ = segment_event_audit(state, candidate, cfg)
            if candidate_peaks.max() <= threshold:
                found = float(lam), candidate, candidate_state, float(candidate_peaks.max())
                break
        if found is None:
            return {
                "accepted": False,
                "controls": output,
                "raw_peak": raw_peak,
                "final_peak": np.nan,
                "corrected_segments": changed,
                "lambdas": lambdas,
            }
        lam, output[index], state, corrected_peak = found
        applied_peak = max(applied_peak, corrected_peak)
        lambdas.append(lam)
        changed += 1
    return {
        "accepted": applied_peak <= threshold,
        "controls": output,
        "raw_peak": raw_peak,
        "final_peak": applied_peak,
        "corrected_segments": changed,
        "lambdas": lambdas,
        "segment_cache_reused": True,
        "final_reaudit_performed": False,
    }


def _float(row: dict, key: str) -> float:
    value = row.get(key, "nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def evaluate_job(method: str, seed: int, force: bool = False) -> dict:
    branch = METHODS[method]
    train = _train_dir(method, seed)
    output = OUT / method / "cstr" / f"seed{seed}" / "hds_test400_margin1e6"
    summary_path = output / "summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    cfg, test = _load_protocol()
    old_dir = train / "hds_test400"
    old_rows = _read_rows(old_dir / f"test_per_sample_{branch}.csv")
    controls, inference = _predict(train, branch, cfg, test)
    rows: list[dict] = []
    recomputed = 0
    reused = 0
    for index, (state, nominal, old) in enumerate(zip(test, controls, old_rows)):
        old_final = _float(old, "final_max_g")
        row = dict(old)
        if old_final <= THRESHOLD:
            # If the old closest-to-one candidate has peak <= the new margin,
            # every earlier rejected candidate was >1e-8 and hence >-1e-6.
            # Therefore the new threshold selects the identical controls.
            row["strict_margin_recomputed"] = False
            row["strict_margin_reuse_proof"] = "old applied peak <= -1e-6 implies identical closest-to-one decisions"
            reused += 1
        else:
            started = time.perf_counter()
            nominal_objective = trajectory_objective(state, nominal, cfg)
            nominal_peak = _nominal_peak(state, nominal, cfg)
            result = correct_with_threshold(state, nominal, cfg, THRESHOLD)
            applied_objective = (
                trajectory_objective(state, result["controls"], cfg) if result["accepted"] else np.nan
            )
            elapsed = time.perf_counter() - started
            if not np.isclose(nominal_peak, _float(old, "nominal_max_g"), atol=1e-6, rtol=0.0):
                raise ValueError(f"{method} seed {seed} sample {index}: nominal peak mismatch")
            if not np.isclose(nominal_objective, _float(old, "nominal_objective"), atol=2e-6, rtol=0.0):
                raise ValueError(f"{method} seed {seed} sample {index}: nominal objective mismatch")
            reference = _float(old, "reference_high_fidelity_objective")
            denominator = max(abs(reference), 1e-12)
            row.update({
                "accepted": bool(result["accepted"]),
                "fallback": not bool(result["accepted"]),
                "nominal_max_g": nominal_peak,
                "final_max_g": result["final_peak"],
                "nominal_objective": nominal_objective,
                "hds_objective": applied_objective,
                "corrected_segments": int(result["corrected_segments"]),
                "hds_seconds": elapsed,
                "inference_seconds": inference,
                "total_predeployment_seconds": inference + elapsed,
                "relative_nominal_gap_percent": 100.0 * (nominal_objective - reference) / denominator,
                "relative_hds_gap_percent": (
                    100.0 * (applied_objective - reference) / denominator if result["accepted"] else np.nan
                ),
                "relative_nominal_gap_qualified_reference_percent": (
                    100.0 * (nominal_objective - reference) / denominator
                    if str(old["reference_continuous_audit_qualified"]).lower() == "true" else np.nan
                ),
                "relative_hds_gap_qualified_reference_percent": (
                    100.0 * (applied_objective - reference) / denominator
                    if result["accepted"] and str(old["reference_continuous_audit_qualified"]).lower() == "true"
                    else np.nan
                ),
                "strict_margin_recomputed": True,
                "strict_margin_reuse_proof": "",
            })
            recomputed += 1
        row["strict_acceptance_threshold"] = THRESHOLD
        row["method_key"] = method
        row["training_seed"] = seed
        rows.append(row)

    _write_rows(output / f"test_per_sample_{branch}.csv", rows)
    accepted = [row for row in rows if str(row["accepted"]).lower() == "true"]
    qualified = [
        row for row in accepted
        if str(row["reference_continuous_audit_qualified"]).lower() == "true"
    ]
    summary = {
        "method": method,
        "branch": branch,
        "seed": seed,
        "acceptance_threshold": THRESHOLD,
        "source_training_directory": str(train),
        "source_old_evaluation": str(old_dir),
        "samples": len(rows),
        "reused_identical_rows": reused,
        "freshly_recomputed_rows": recomputed,
        "accepted": len(accepted),
        "fallback": len(rows) - len(accepted),
        "nominal_violation_rate_percent": float(100 * np.mean([_float(row, "nominal_max_g") > 0 for row in rows])),
        "final_max_g": float(np.nanmax([_float(row, "final_max_g") for row in accepted])) if accepted else None,
        "hds_gap_all400_percent": float(np.nanmean([_float(row, "relative_hds_gap_percent") for row in accepted])) if accepted else None,
        "hds_gap_qualified286_percent": float(np.nanmean([_float(row, "relative_hds_gap_qualified_reference_percent") for row in qualified])) if qualified else None,
        "mean_corrected_segments": float(np.mean([_float(row, "corrected_segments") for row in accepted])) if accepted else None,
        "mean_hds_seconds": float(np.mean([_float(row, "hds_seconds") for row in rows])),
        "mean_total_predeployment_seconds": float(np.mean([_float(row, "total_predeployment_seconds") for row in rows])),
        "hds_statement": "continuous-time numerical audit evidence under the declared model and numerical settings",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def regression_check() -> dict:
    cfg, test = _load_protocol()
    checks = []
    for method in METHODS:
        seed = SEEDS[0]
        branch = METHODS[method]
        train = _train_dir(method, seed)
        rows = _read_rows(train / "hds_test400" / f"test_per_sample_{branch}.csv")
        controls, _ = _predict(train, branch, cfg, test)
        ranked = np.argsort([abs(_float(row, "final_max_g")) for row in rows])
        indices = sorted(set([0, 1, 2, int(ranked[0]), int(ranked[1]), int(ranked[-1])]))
        for index in indices:
            result = correct_with_threshold(test[index], controls[index], cfg, 1e-8)
            old = rows[index]
            final_match = np.isclose(result["final_peak"], _float(old, "final_max_g"), atol=1e-8, rtol=0.0)
            segments_match = int(result["corrected_segments"]) == int(float(old["corrected_segments"]))
            accepted_match = bool(result["accepted"]) == (str(old["accepted"]).lower() == "true")
            checks.append({
                "method": method,
                "seed": seed,
                "index": index,
                "final_peak_match": bool(final_match),
                "corrected_segments_match": bool(segments_match),
                "accepted_match": bool(accepted_match),
                "old_final_peak": _float(old, "final_max_g"),
                "reconstructed_final_peak": result["final_peak"],
            })
    passed = all(
        row["final_peak_match"] and row["corrected_segments_match"] and row["accepted_match"]
        for row in checks
    )
    report = {"passed": passed, "threshold": 1e-8, "checks": checks}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "regression_old_threshold.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if not passed:
        raise RuntimeError("Old-threshold regression failed")
    return report


def _sample_stats(values: list[float]) -> dict:
    data = np.asarray(values, float)
    return {
        "mean": float(data.mean()),
        "sample_sd": float(data.std(ddof=1)),
        "median": float(np.median(data)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def aggregate() -> dict:
    per_method: dict[str, list[dict]] = {}
    for method in METHODS:
        items = []
        for seed in SEEDS:
            path = OUT / method / "cstr" / f"seed{seed}" / "hds_test400_margin1e6" / "summary.json"
            items.append(json.loads(path.read_text(encoding="utf-8")))
        per_method[method] = items
    metrics = (
        "hds_gap_all400_percent",
        "hds_gap_qualified286_percent",
        "mean_corrected_segments",
        "mean_hds_seconds",
        "mean_total_predeployment_seconds",
        "final_max_g",
    )
    methods = {
        method: {
            metric: _sample_stats([float(item[metric]) for item in items])
            for metric in metrics
        } | {
            "accepted_total": int(sum(item["accepted"] for item in items)),
            "fallback_total": int(sum(item["fallback"] for item in items)),
            "freshly_recomputed_rows": int(sum(item["freshly_recomputed_rows"] for item in items)),
            "reused_identical_rows": int(sum(item["reused_identical_rows"] for item in items)),
        }
        for method, items in per_method.items()
    }
    paired = {}
    baseline = np.asarray([item["hds_gap_all400_percent"] for item in per_method["supervised"]], float)
    for method in ("unprocessed", "linear_cosine", "standard_pcgrad"):
        current = np.asarray([item["hds_gap_all400_percent"] for item in per_method[method]], float)
        difference = current - baseline
        paired[f"{method}_minus_supervised"] = {
            "mean_difference_percentage_points": float(difference.mean()),
            "wins_vs_supervised": int(np.sum(difference < 0)),
            "losses_vs_supervised": int(np.sum(difference > 0)),
            "paired_t_p": float(scipy_stats.ttest_rel(current, baseline).pvalue),
            "wilcoxon_p": float(scipy_stats.wilcoxon(current, baseline).pvalue),
        }
    report = {
        "protocol": "CSTR N=100/RK10, frozen 30 seeds and models, HDS acceptance g_max <= -1e-6",
        "acceptance_threshold": THRESHOLD,
        "seeds": list(SEEDS),
        "methods": methods,
        "paired_all400": paired,
        "reuse_note": (
            "Rows with old final_max_g <= -1e-6 are mathematically identical under the new threshold; "
            "all other rows were freshly audited."
        ),
    }
    (OUT / "aggregate_30seeds_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    table = [
        "| Method | all-400 HDS gap (%) | qualified-286 gap (%) | Corrected segments | Final max g | Accepted / fallback |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        item = methods[method]
        table.append(
            f"| {method} | {item['hds_gap_all400_percent']['mean']:.5f} ± {item['hds_gap_all400_percent']['sample_sd']:.5f} | "
            f"{item['hds_gap_qualified286_percent']['mean']:.5f} ± {item['hds_gap_qualified286_percent']['sample_sd']:.5f} | "
            f"{item['mean_corrected_segments']['mean']:.4f} ± {item['mean_corrected_segments']['sample_sd']:.4f} | "
            f"{item['final_max_g']['max']:.3e} | {item['accepted_total']} / {item['fallback_total']} |"
        )
    (OUT / "aggregate_30seeds_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--regression-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if not args.aggregate_only:
        report = regression_check()
        print(f"old-threshold regression: {len(report['checks'])} checks passed", flush=True)
    if args.regression_only:
        return
    if not args.aggregate_only:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            jobs = {
                executor.submit(evaluate_job, method, seed, args.force): (method, seed)
                for method in METHODS for seed in SEEDS
            }
            for future in as_completed(jobs):
                result = future.result()
                print(
                    f"{result['method']} seed {result['seed']}: "
                    f"fresh {result['freshly_recomputed_rows']}, reused {result['reused_identical_rows']}, "
                    f"accepted {result['accepted']}", flush=True
                )
    aggregate()
    print(OUT / "aggregate_30seeds_table.md", flush=True)


if __name__ == "__main__":
    main()
