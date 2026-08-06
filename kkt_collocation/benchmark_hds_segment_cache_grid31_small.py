"""Benchmark grid-31 HDS segment-cache reuse against redundant final replay.

The frozen seed-20260771 models and the same 12 stress/coverage points used in
the lambda-grid sensitivity check are reused. Both variants use the identical
31-point-base-grid, closest-to-one, first-safe search without bisection. The only
difference is whether the complete corrected sequence is propagated a second
time after every applied segment has already been audited once.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from kkt_collocation import compare_lambda_discrete31_vs51_small as base
from kkt_collocation.run_two_stage_vs_kkt_only_ablation import Problem
from kkt_collocation.run_unified_su_suj_sk_konly_ablation import UnifiedConfig
from kkt_collocation.reevaluate_cstr_margin1e6_30seeds import _load_protocol, _predict as predict_cstr, _train_dir as cstr_train_dir
from kkt_collocation.economou_cstr_hds_fast import candidates as cstr_candidates, segment_event_audit


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "kkt_collocation" / "results"
OUT = RESULTS / "nonformal_hds_segment_cache_grid31_small_20260806_v1"
GRID = 31
THRESHOLD = -1e-8
METHODS = base.METHODS
BRANCH = base.BRANCH
VARIANTS = ("cached_segment_reuse", "redundant_final_reaudit")


def scalar_correct(corrector, initial, nominal, duration: float, final_reaudit: bool, threshold: float = THRESHOLD) -> dict:
    state = np.asarray(initial, float).copy()
    output = np.asarray(nominal, float).copy()
    applied_peak = -np.inf
    corrected = evaluations = 0
    for index, control in enumerate(output.copy()):
        peak, terminal = corrector.segment_peak(state, float(control), duration)
        if peak <= threshold:
            state = terminal
            applied_peak = max(applied_peak, peak)
            continue
        chosen = None
        for lam in corrector._candidate_lambdas(float(control), exclude_nominal=True):
            evaluations += 1
            candidate = float(lam * control)
            candidate_peak, candidate_terminal = corrector.segment_peak(state, candidate, duration)
            if candidate_peak <= threshold:
                chosen = candidate, candidate_peak, candidate_terminal
                break
        if chosen is None:
            return {"accepted": False, "controls": output, "corrected_segments": corrected, "candidate_evaluations": evaluations, "final_peak": np.nan}
        output[index], candidate_peak, state = chosen
        applied_peak = max(applied_peak, candidate_peak)
        corrected += 1
    final_peak = corrector.audit(initial, output, duration) if final_reaudit else float(applied_peak)
    return {"accepted": final_peak <= threshold, "controls": output, "corrected_segments": corrected, "candidate_evaluations": evaluations, "final_peak": final_peak}


def cstr_audit(initial, controls, cfg) -> float:
    state = np.asarray(initial, float).copy()
    peak = -np.inf
    for control in controls:
        local, state, _ = segment_event_audit(state, control, cfg)
        peak = max(peak, float(np.max(local)))
    return float(peak)


def cstr_correct(initial, nominal, cfg, final_reaudit: bool, threshold: float = THRESHOLD) -> dict:
    state = np.asarray(initial, float).copy()
    output = np.asarray(nominal, float).copy()
    applied_peak = -np.inf
    corrected = evaluations = 0
    for index in range(cfg.zoh_steps):
        peaks, terminal, _ = segment_event_audit(state, output[index], cfg)
        peak = float(np.max(peaks))
        if peak <= threshold:
            state = terminal
            applied_peak = max(applied_peak, peak)
            continue
        low, span, normalized, order = cstr_candidates(output[index], cfg, GRID)
        chosen = None
        for lam in order:
            if abs(float(lam) - 1.0) <= 1e-14:
                continue
            evaluations += 1
            candidate = low + np.clip(float(lam) * normalized, 0.0, 1.0) * span
            candidate_peaks, candidate_terminal, _ = segment_event_audit(state, candidate, cfg)
            candidate_peak = float(np.max(candidate_peaks))
            if candidate_peak <= threshold:
                chosen = candidate, candidate_peak, candidate_terminal
                break
        if chosen is None:
            return {"accepted": False, "controls": output, "corrected_segments": corrected, "candidate_evaluations": evaluations, "final_peak": np.nan}
        output[index], candidate_peak, state = chosen
        applied_peak = max(applied_peak, candidate_peak)
        corrected += 1
    final_peak = cstr_audit(initial, output, cfg) if final_reaudit else float(applied_peak)
    return {"accepted": final_peak <= threshold, "controls": output, "corrected_segments": corrected, "candidate_evaluations": evaluations, "final_peak": final_peak}


def record_pair(benchmark: str, method: str, sample_index: int, runner, reverse: bool) -> list[dict]:
    rows = []
    order = tuple(reversed(VARIANTS)) if reverse else VARIANTS
    for variant in order:
        started = time.perf_counter()
        result = runner(variant == "redundant_final_reaudit")
        elapsed = time.perf_counter() - started
        rows.append({
            "benchmark": benchmark, "method": method, "sample_index": sample_index,
            "variant": variant, "grid_size": GRID, "bisection": False,
            "accepted": bool(result["accepted"]), "final_max_g": float(result["final_peak"]),
            "corrected_segments": int(result["corrected_segments"]),
            "candidate_evaluations": int(result["candidate_evaluations"]),
            "hds_seconds": elapsed, "controls": result["controls"],
        })
    return rows


def evaluate_scalar(benchmark: str) -> tuple[list[dict], list[int]]:
    problem = Problem(benchmark, torch.device("cpu"), base.SEED, UnifiedConfig())
    models = base.load_scalar_models(benchmark, problem)
    indices = base.choose_indices([base.scalar_csv(benchmark, method) for method in METHODS], "nominal_hds_max_g", "sample_index")
    predictions = {method: problem.predict(model, problem.test_states)[0] for method, model in models.items()}
    corrector = base.scalar_corrector(benchmark, GRID)
    rows = []
    for method_index, method in enumerate(METHODS):
        for sample_index in indices:
            value = problem.test_states[sample_index]
            initial = problem.initial_state(value)
            nominal = predictions[method][sample_index]
            rows.extend(record_pair(
                benchmark, method, sample_index,
                lambda replay, a=initial, n=nominal: scalar_correct(corrector, a, n, problem.duration, replay),
                bool((method_index + sample_index) % 2),
            ))
    return rows, indices


def evaluate_cstr() -> tuple[list[dict], list[int]]:
    cfg, states = _load_protocol()
    indices = base.choose_indices([base.cstr_csv(method) for method in METHODS], "nominal_max_g", "index")
    predictions = {method: predict_cstr(cstr_train_dir(method, base.SEED), BRANCH[method], cfg, states)[0] for method in METHODS}
    rows = []
    for method_index, method in enumerate(METHODS):
        for sample_index in indices:
            initial = states[sample_index]
            nominal = predictions[method][sample_index]
            rows.extend(record_pair(
                "cstr", method, sample_index,
                lambda replay, a=initial, n=nominal: cstr_correct(a, n, cfg, replay),
                bool((method_index + sample_index) % 2),
            ))
    return rows, indices


def summarize(rows: list[dict], indices: dict[str, list[int]]) -> dict:
    report = {
        "formal_protocol": False, "seed": base.SEED, "grid_size": GRID, "bisection": False,
        "purpose": "paired timing regression for within-trajectory segment propagation reuse",
        "benchmarks": {"vdp": "CVP10", "penicillin": "CVP10", "cstr": "CVP100"},
        "indices": indices,
        "mechanism": "reuse the terminal state and peak from each accepted nominal or corrected candidate segment; omit redundant full-sequence replay",
        "results": {},
    }
    for benchmark in ("vdp", "penicillin", "cstr"):
        selected = [row for row in rows if row["benchmark"] == benchmark]
        by_key = {(row["method"], row["sample_index"], row["variant"]): row for row in selected}
        pairs = [(by_key[(method, idx, VARIANTS[0])], by_key[(method, idx, VARIANTS[1])]) for method in METHODS for idx in indices[benchmark]]
        cached_time = np.asarray([a["hds_seconds"] for a, _ in pairs])
        replay_time = np.asarray([b["hds_seconds"] for _, b in pairs])
        controls_match = [np.array_equal(a["controls"], b["controls"]) for a, b in pairs]
        report["results"][benchmark] = {
            "trajectories": len(pairs),
            "accepted_cached": int(sum(a["accepted"] for a, _ in pairs)),
            "accepted_reaudit": int(sum(b["accepted"] for _, b in pairs)),
            "exact_control_match": int(sum(controls_match)),
            "corrected_segment_match": int(sum(a["corrected_segments"] == b["corrected_segments"] for a, b in pairs)),
            "max_abs_final_peak_difference": float(max(abs(a["final_max_g"] - b["final_max_g"]) for a, b in pairs)),
            "mean_cached_ms": float(1e3 * np.mean(cached_time)),
            "mean_reaudit_ms": float(1e3 * np.mean(replay_time)),
            "speedup": float(np.mean(replay_time) / np.mean(cached_time)),
            "time_reduction_percent": float(100.0 * (1.0 - np.mean(cached_time) / np.mean(replay_time))),
        }
    return report


def write_csv(path: Path, rows: list[dict]) -> None:
    serial = [{k: v for k, v in row.items() if k != "controls"} for row in rows]
    fields = sorted({key for row in serial for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(serial)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    rows, indices = [], {}
    for benchmark in ("vdp", "penicillin"):
        part, selected = evaluate_scalar(benchmark); rows.extend(part); indices[benchmark] = selected
        print(f"{benchmark}: {len(part)} timed runs", flush=True)
    part, selected = evaluate_cstr(); rows.extend(part); indices["cstr"] = selected
    print(f"cstr: {len(part)} timed runs", flush=True)
    report = summarize(rows, indices)
    write_csv(OUT / "per_case.csv", rows)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Grid-31 HDS segment-cache timing", "", "| Benchmark | cached (ms) | redundant re-audit (ms) | speedup | reduction | exact controls |", "|---|---:|---:|---:|---:|---:|"]
    for benchmark, value in report["results"].items():
        lines.append(f"| {benchmark} | {value['mean_cached_ms']:.3f} | {value['mean_reaudit_ms']:.3f} | {value['speedup']:.3f}x | {value['time_reduction_percent']:.1f}% | {value['exact_control_match']}/{value['trajectories']} |")
    (OUT / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
