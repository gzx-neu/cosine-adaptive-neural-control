"""Nonformal small check of discrete-only 31 versus 51 lambda candidates.

Frozen seed-20260771 models are reused for the original VDP CVP10,
penicillin CVP10, and CSTR CVP100 experiments.  No network is retrained and
no deterministic reference is recomputed.  Both arms return the first safe
candidate in closest-to-one order and perform no bisection.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "kkt_collocation" / "results"
OUT = RESULTS / "nonformal_lambda_discrete31_vs51_small_20260804_v1"
SEED = 20260771
METHODS = ("supervised", "unprocessed", "linear_cosine", "standard_pcgrad")
BRANCH = {"supervised": "S-u", "unprocessed": "S-u+K", "linear_cosine": "S-u+K", "standard_pcgrad": "S-u+K"}
VARIANTS = {"discrete31": 31, "discrete51": 51}
THRESHOLD = -1e-6
FIXED_COVERAGE = (0, 79, 159, 239, 319, 399)

from offline_safe_control.adaptive_event_hds import AdaptiveEventHDSConfig, AdaptiveEventHDSCorrector
from kkt_collocation.run_two_stage_vs_kkt_only_ablation import Problem
from kkt_collocation.run_unified_su_suj_sk_konly_ablation import UnifiedConfig
from kkt_collocation.run_vdp_ablation import constraint as vdp_g, constraint_derivative as vdp_gdot, vdp_ode
from kkt_collocation.run_penicillin_ablation import DT as PEN_DT, UMAX as PEN_UMAX, g as pen_g, gdot as pen_gdot, ode as pen_ode
from kkt_collocation.reevaluate_cstr_margin1e6_30seeds import _load_checkpoint, _load_protocol, _predict as predict_cstr, _train_dir as cstr_train_dir
from kkt_collocation.economou_cstr_hds_fast import candidates as cstr_candidates, segment_event_audit
from kkt_collocation.run_economou_cstr_supervised_hds import trajectory_objective


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_root(benchmark: str) -> Path:
    name = "multiseed10_vdp_k10_cuda_20260803_v1" if benchmark == "vdp" else "multiseed10_penicillin_k10_cpu_20260803_v1"
    return RESULTS / name


def formal_root(benchmark: str) -> Path:
    name = "multiseed30_vdp_k10_refined_lambda_cached_margin1e6_20260804_v1" if benchmark == "vdp" else "multiseed30_penicillin_k10_refined_lambda_cached_margin1e6_20260804_v1"
    return RESULTS / name


def scalar_csv(benchmark: str, method: str) -> Path:
    return formal_root(benchmark) / method / benchmark / f"seed{SEED}" / f"test_per_sample_{BRANCH[method]}.csv"


def cstr_csv(method: str) -> Path:
    return RESULTS / "multiseed30_cstr_k10_refined_lambda_cached_margin1e6_20260804_v1" / method / "cstr" / f"seed{SEED}" / "hds_test400_margin1e6" / f"test_per_sample_{BRANCH[method]}.csv"


def choose_indices(paths: list[Path], peak_field: str, index_field: str) -> list[int]:
    worst = np.full(400, -np.inf)
    for path in paths:
        rows = read_rows(path)
        if len(rows) != 400:
            raise ValueError(f"Expected 400 rows in {path}")
        for row in rows:
            idx = int(row[index_field])
            worst[idx] = max(worst[idx], float(row[peak_field]))
    top = [int(value) for value in np.argsort(worst)[-6:][::-1]]
    return sorted(int(value) for value in set(top).union(FIXED_COVERAGE))


def scalar_corrector(benchmark: str, grid: int) -> AdaptiveEventHDSCorrector:
    config = AdaptiveEventHDSConfig(
        grid_size=grid, allow_nonformal_grid=(grid != 31), safety_margin=1e-6,
        rtol=1e-10, atol=1e-12,
    )
    if benchmark == "vdp":
        return AdaptiveEventHDSCorrector(vdp_ode, vdp_g, vdp_gdot, (-0.3, 1.0), config)
    return AdaptiveEventHDSCorrector(pen_ode, pen_g, pen_gdot, (0.0, PEN_UMAX), config)


def correct_scalar_discrete(corrector, initial: np.ndarray, nominal: np.ndarray, duration: float) -> dict:
    state = np.asarray(initial, float).copy()
    output = np.asarray(nominal, float).copy()
    lambdas: list[float] = []
    corrected = evaluations = 0
    for index, control in enumerate(output.copy()):
        peak, terminal = corrector.segment_peak(state, float(control), duration)
        if peak <= THRESHOLD:
            state = terminal
            lambdas.append(1.0)
            continue
        chosen = None
        for lam in corrector._candidate_lambdas(float(control), exclude_nominal=True):
            evaluations += 1
            candidate = float(lam * control)
            candidate_peak, candidate_terminal = corrector.segment_peak(state, candidate, duration)
            if candidate_peak <= THRESHOLD:
                chosen = float(lam), candidate, candidate_terminal
                break
        if chosen is None:
            return {"accepted": False, "controls": output, "corrected_segments": corrected, "lambdas": lambdas, "candidate_evaluations": evaluations}
        lam, output[index], state = chosen
        lambdas.append(lam)
        corrected += 1
    final_peak = corrector.audit(initial, output, duration)
    return {"accepted": final_peak <= THRESHOLD, "controls": output, "corrected_segments": corrected, "lambdas": lambdas, "candidate_evaluations": evaluations, "final_peak": final_peak}


def load_scalar_models(benchmark: str, problem: Problem) -> dict[str, torch.nn.Module]:
    models = {}
    for method in METHODS:
        checkpoint = _load_checkpoint(source_root(benchmark) / method / benchmark / f"seed{SEED}" / f"{BRANCH[method]}.pth")
        model = problem.make_model().eval()
        model.load_state_dict(checkpoint["model"])
        models[method] = model
    return models


def evaluate_scalar(benchmark: str) -> tuple[list[dict], list[int]]:
    problem = Problem(benchmark, torch.device("cpu"), SEED, UnifiedConfig())
    models = load_scalar_models(benchmark, problem)
    indices = choose_indices([scalar_csv(benchmark, method) for method in METHODS], "nominal_hds_max_g", "sample_index")
    predictions = {method: problem.predict(model, problem.test_states)[0] for method, model in models.items()}
    rows = []
    for method_index, method in enumerate(METHODS):
        for sample_index in indices:
            value = problem.test_states[sample_index]
            initial = problem.initial_state(value)
            nominal = predictions[method][sample_index]
            reference = problem.reference_records[sample_index]["objective"]
            names = tuple(VARIANTS) if (method_index + sample_index) % 2 == 0 else tuple(reversed(VARIANTS))
            for name in names:
                grid = VARIANTS[name]
                corrector = scalar_corrector(benchmark, grid)
                started = time.perf_counter()
                outcome = correct_scalar_discrete(corrector, initial, nominal, problem.duration)
                elapsed = time.perf_counter() - started
                objective = problem.cost(value, outcome["controls"], corrector) if outcome["accepted"] else np.nan
                lambdas = np.asarray(outcome["lambdas"], float)
                rows.append({
                    "benchmark": benchmark, "method": method, "sample_index": sample_index,
                    "variant": name, "grid_size": grid, "bisection": False,
                    "accepted": outcome["accepted"], "corrected_segments": outcome["corrected_segments"],
                    "final_max_g": outcome.get("final_peak", np.nan), "hds_objective": objective,
                    "gap_percent": 100.0 * (objective - reference) / max(abs(reference), 1e-12),
                    "correction_seconds": elapsed, "candidate_evaluations": outcome["candidate_evaluations"],
                    "mean_abs_lambda_minus_one": float(np.mean(np.abs(lambdas - 1.0))) if len(lambdas) else np.nan,
                })
    return rows, indices


def audit_cstr(state0: np.ndarray, controls: np.ndarray, cfg) -> float:
    state = np.asarray(state0, float).copy()
    peak = -np.inf
    for control in controls:
        local, state, _ = segment_event_audit(state, control, cfg)
        peak = max(peak, float(np.max(local)))
    return float(peak)


def correct_cstr_discrete(state0: np.ndarray, nominal: np.ndarray, cfg, grid: int) -> dict:
    state = np.asarray(state0, float).copy()
    output = np.asarray(nominal, float).copy()
    lambdas: list[float] = []
    corrected = evaluations = 0
    for index in range(cfg.zoh_steps):
        peaks, terminal, _ = segment_event_audit(state, output[index], cfg)
        if float(np.max(peaks)) <= THRESHOLD:
            state = terminal
            lambdas.append(1.0)
            continue
        low, span, normalized, order = cstr_candidates(output[index], cfg, grid)
        chosen = None
        for lam in order:
            if abs(float(lam) - 1.0) <= 1e-14:
                continue
            evaluations += 1
            candidate = low + np.clip(float(lam) * normalized, 0.0, 1.0) * span
            candidate_peaks, candidate_terminal, _ = segment_event_audit(state, candidate, cfg)
            if float(np.max(candidate_peaks)) <= THRESHOLD:
                chosen = float(lam), candidate, candidate_terminal
                break
        if chosen is None:
            return {"accepted": False, "controls": output, "corrected_segments": corrected, "lambdas": lambdas, "candidate_evaluations": evaluations}
        lam, output[index], state = chosen
        lambdas.append(lam)
        corrected += 1
    final_peak = audit_cstr(state0, output, cfg)
    return {"accepted": final_peak <= THRESHOLD, "controls": output, "corrected_segments": corrected, "lambdas": lambdas, "candidate_evaluations": evaluations, "final_peak": final_peak}


def evaluate_cstr() -> tuple[list[dict], list[int]]:
    cfg, states = _load_protocol()
    indices = choose_indices([cstr_csv(method) for method in METHODS], "nominal_max_g", "index")
    controls, references = {}, {}
    for method in METHODS:
        controls[method], _ = predict_cstr(cstr_train_dir(method, SEED), BRANCH[method], cfg, states)
        references[method] = {int(row["index"]): float(row["reference_high_fidelity_objective"]) for row in read_rows(cstr_csv(method))}
    rows = []
    for method_index, method in enumerate(METHODS):
        for sample_index in indices:
            names = tuple(VARIANTS) if (method_index + sample_index) % 2 == 0 else tuple(reversed(VARIANTS))
            for name in names:
                grid = VARIANTS[name]
                started = time.perf_counter()
                outcome = correct_cstr_discrete(states[sample_index], controls[method][sample_index], cfg, grid)
                elapsed = time.perf_counter() - started
                objective = trajectory_objective(states[sample_index], outcome["controls"], cfg) if outcome["accepted"] else np.nan
                reference = references[method][sample_index]
                lambdas = np.asarray(outcome["lambdas"], float)
                rows.append({
                    "benchmark": "cstr", "method": method, "sample_index": sample_index,
                    "variant": name, "grid_size": grid, "bisection": False,
                    "accepted": outcome["accepted"], "corrected_segments": outcome["corrected_segments"],
                    "final_max_g": outcome.get("final_peak", np.nan), "hds_objective": objective,
                    "gap_percent": 100.0 * (objective - reference) / max(abs(reference), 1e-12),
                    "correction_seconds": elapsed, "candidate_evaluations": outcome["candidate_evaluations"],
                    "mean_abs_lambda_minus_one": float(np.mean(np.abs(lambdas - 1.0))) if len(lambdas) else np.nan,
                })
    return rows, indices


def summarize(rows: list[dict], indices: dict[str, list[int]]) -> dict:
    report = {
        "formal_protocol": False, "seed": SEED,
        "purpose": "small sensitivity check of discrete-only 31 versus 51 lambda candidates",
        "benchmarks": {"vdp": "CVP10", "penicillin": "CVP10", "cstr": "CVP100"},
        "selection": "six largest union nominal peaks across methods plus six fixed coverage indices",
        "indices": indices,
        "shared_settings": {"integrator": "adaptive DOP853 with stationary-event extrema", "rtol": 1e-10, "atol": 1e-12, "acceptance_threshold": THRESHOLD, "bisection": False},
        "methods": {},
    }
    for benchmark in ("vdp", "penicillin", "cstr"):
        report["methods"][benchmark] = {}
        for method in METHODS:
            subset = [row for row in rows if row["benchmark"] == benchmark and row["method"] == method]
            by_key = {(int(row["sample_index"]), row["variant"]): row for row in subset}
            pairs = [(by_key[(idx, "discrete31")], by_key[(idx, "discrete51")]) for idx in indices[benchmark]]
            gap31 = np.asarray([a["gap_percent"] for a, _ in pairs], float)
            gap51 = np.asarray([b["gap_percent"] for _, b in pairs], float)
            time31 = np.asarray([a["correction_seconds"] for a, _ in pairs], float)
            time51 = np.asarray([b["correction_seconds"] for _, b in pairs], float)
            report["methods"][benchmark][method] = {
                "samples": len(pairs),
                "accepted_31": int(sum(bool(a["accepted"]) for a, _ in pairs)),
                "accepted_51": int(sum(bool(b["accepted"]) for _, b in pairs)),
                "mean_gap_31_percent": float(np.nanmean(gap31)),
                "mean_gap_51_percent": float(np.nanmean(gap51)),
                "mean_gap_51_minus_31_percentage_points": float(np.nanmean(gap51 - gap31)),
                "max_abs_gap_difference_percentage_points": float(np.nanmax(np.abs(gap51 - gap31))),
                "mean_corrected_segments_31": float(np.mean([a["corrected_segments"] for a, _ in pairs])),
                "mean_corrected_segments_51": float(np.mean([b["corrected_segments"] for _, b in pairs])),
                "corrected_segments_equal": int(sum(a["corrected_segments"] == b["corrected_segments"] for a, b in pairs)),
                "mean_correction_ms_31": float(1e3 * np.mean(time31)),
                "mean_correction_ms_51": float(1e3 * np.mean(time51)),
                "timing_ratio_51_over_31": float(np.mean(time51) / np.mean(time31)),
            }
    return report


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_table(path: Path, report: dict) -> None:
    lines = [
        "# Discrete-only lambda grid: 31 versus 51", "",
        "One frozen seed and a 12-point stress/coverage subset per benchmark; neither arm uses bisection.", "",
        "| Benchmark | Method | gap 31 (%) | gap 51 (%) | 51-31 (pp) | corrected segments 31 | corrected segments 51 | time 51/31 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark, methods in report["methods"].items():
        for method, value in methods.items():
            lines.append(
                f"| {benchmark} | {method} | {value['mean_gap_31_percent']:.8f} | {value['mean_gap_51_percent']:.8f} | "
                f"{value['mean_gap_51_minus_31_percentage_points']:.3e} | {value['mean_corrected_segments_31']:.3f} | "
                f"{value['mean_corrected_segments_51']:.3f} | {value['timing_ratio_51_over_31']:.3f} |"
            )
    lines += ["", "This is a nonformal sensitivity check and does not replace the 30-seed all-400 evaluation."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    rows, selected = [], {}
    for benchmark in ("vdp", "penicillin"):
        part, idx = evaluate_scalar(benchmark); rows.extend(part); selected[benchmark] = idx
        print(f"{benchmark}: {len(part)} evaluations", flush=True)
    part, idx = evaluate_cstr(); rows.extend(part); selected["cstr"] = idx
    print(f"cstr: {len(part)} evaluations", flush=True)
    report = summarize(rows, selected)
    write_csv(OUT / "per_case.csv", rows)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_table(OUT / "summary_table.md", report)
    print(OUT / "summary_table.md")


if __name__ == "__main__":
    main()
