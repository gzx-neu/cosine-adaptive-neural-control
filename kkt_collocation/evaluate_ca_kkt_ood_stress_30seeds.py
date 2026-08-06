"""Guard-bypassed OOD stress diagnostic for the frozen 30-seed CA-KKT policies.

This script does not train a model and does not solve new deterministic OOD
references.  Near and far points are deliberately outside the declared
training domain.  In the declared deployment workflow every such point is
routed by the domain guard to the deterministic solver.  Neural-policy and
HDS results here intentionally bypass that guard for diagnostic purposes only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kkt_collocation"))

from offline_safe_control.adaptive_event_hds import (  # noqa: E402
    AdaptiveEventHDSConfig,
    AdaptiveEventHDSCorrector,
)
from kkt_collocation.economou_cstr_hds_fast import segment_event_audit  # noqa: E402
from kkt_collocation.reevaluate_cstr_margin1e6_30seeds import (  # noqa: E402
    correct_with_threshold,
)
from kkt_collocation.run_economou_cstr_supervised_hds import (  # noqa: E402
    PolicyValue,
    trajectory_objective,
)
from kkt_collocation.run_penicillin_ablation import (  # noqa: E402
    DT as PEN_DT,
    UMAX as PEN_UMAX,
    Policy as PenicillinPolicy,
    g as pen_g,
    gdot as pen_gdot,
    ode as pen_ode,
    terminal_product,
)
from kkt_collocation.run_vdp_ablation import (  # noqa: E402
    constraint as vdp_g,
    constraint_derivative as vdp_gdot,
    terminal_cost,
    vdp_ode,
)
from kkt_collocation.screen_economou_cstr_30x30 import EconomouScreenConfig  # noqa: E402
from kkt_collocation.train_vdp_kkt_policy import (  # noqa: E402
    KKTPolicyValueNetwork,
    TrainConfig as VDPTrainConfig,
)


RESULTS = ROOT / "kkt_collocation" / "results"
DEFAULT_OUTPUT = RESULTS / "ca_kkt_ood_stress_30seeds_20260803_v1"
DEFAULT_SEEDS = tuple(range(20260771, 20260801))
THRESHOLD = -1.0e-6
GRID_SIZE = 31
LAYER_SPECS = {
    "near_ood_0_10pct": (0.0, 0.10),
    "far_ood_10_20pct": (0.10, 0.20),
}
POINT_SEEDS = {"vdp": 2026080311, "penicillin": 2026080321, "cstr": 2026080331}


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _dump_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _source_root(problem: str, seed: int) -> Path:
    if problem == "vdp":
        name = (
            "multiseed10_vdp_k10_cuda_20260803_v1"
            if seed <= 20260790
            else "multiseed_vdp_k10_cuda_seeds21_30_20260803_v1"
        )
    elif problem == "penicillin":
        name = (
            "multiseed10_penicillin_k10_cpu_20260803_v1"
            if seed <= 20260790
            else "multiseed_penicillin_k10_cpu_seeds21_30_20260803_v1"
        )
    elif problem == "cstr":
        name = (
            "multiseed10_cstr_k10_cuda_20260803_v1"
            if seed <= 20260790
            else "multiseed_cstr_k10_cuda_seeds21_30_20260803_v1"
        )
    else:
        raise ValueError(problem)
    return RESULTS / name


def _checkpoint_path(problem: str, seed: int) -> Path:
    return _source_root(problem, seed) / "linear_cosine" / problem / f"seed{seed}" / "S-u+K.pth"


def _load_cstr_config() -> EconomouScreenConfig:
    path = RESULTS / "economou_cstr_reduced_kkt_n100_test400_lhs_margin0" / "summary.json"
    raw = json.loads(path.read_text(encoding="utf-8"))["config"]
    for key in ("ti_bounds_K", "flow_bounds", "ca_initial_range", "temperature_initial_range_K"):
        raw[key] = tuple(raw[key])
    cfg = EconomouScreenConfig(**raw)
    if cfg.zoh_steps != 100 or cfg.substeps_per_zoh != 10 or cfg.node_margin != 0.0:
        raise ValueError("Unexpected frozen CSTR protocol")
    return cfg


def _ring_2d(
    lower: np.ndarray,
    upper: np.ndarray,
    samples: int,
    inner_expansion: float,
    outer_expansion: float,
    seed: int,
    initially_safe: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Uniform expanded-box ring, excluding the inner expanded box."""
    rng = np.random.default_rng(seed)
    width = upper - lower
    outer_lower = lower - outer_expansion * width
    outer_upper = upper + outer_expansion * width
    inner_lower = lower - inner_expansion * width
    inner_upper = upper + inner_expansion * width
    retained: list[np.ndarray] = []
    count = 0
    while count < samples:
        draw = rng.uniform(outer_lower, outer_upper, size=(max(16 * samples, 256), 2))
        outside_inner = np.any((draw < inner_lower) | (draw > inner_upper), axis=1)
        keep = draw[outside_inner & initially_safe(draw)]
        retained.append(keep)
        count += len(keep)
        if len(retained) > 1000:
            raise RuntimeError("Could not generate enough initially safe OOD points")
    return np.vstack(retained)[:samples]


def _penicillin_shell(samples: int, inner: float, outer: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower, upper = 0.1, 0.3
    width = upper - lower
    left_count = samples // 2
    values = np.r_[
        rng.uniform(lower - outer * width, lower - inner * width, size=left_count),
        rng.uniform(upper + inner * width, upper + outer * width, size=samples - left_count),
    ]
    rng.shuffle(values)
    if not np.all(values <= 0.5):
        raise ValueError("Generated an initially infeasible penicillin point")
    return values


def _make_cohorts(samples: int) -> tuple[dict[str, dict[str, np.ndarray]], EconomouScreenConfig]:
    cstr_cfg = _load_cstr_config()
    cohorts: dict[str, dict[str, np.ndarray]] = {name: {} for name in ("vdp", "penicillin", "cstr")}
    for layer_index, (layer, (inner, outer)) in enumerate(LAYER_SPECS.items()):
        vdp_xy = _ring_2d(
            np.array([-0.1, 0.9]), np.array([0.1, 1.1]), samples, inner, outer,
            POINT_SEEDS["vdp"] + layer_index,
            lambda points: (-0.4 - points[:, 0]) <= 0.0,
        )
        cohorts["vdp"][layer] = np.c_[vdp_xy, np.zeros(samples)]
        cohorts["penicillin"][layer] = _penicillin_shell(
            samples, inner, outer, POINT_SEEDS["penicillin"] + layer_index
        )
        cstr_xy = _ring_2d(
            np.array([cstr_cfg.ca_initial_range[0], cstr_cfg.temperature_initial_range_K[0]]),
            np.array([cstr_cfg.ca_initial_range[1], cstr_cfg.temperature_initial_range_K[1]]),
            samples, inner, outer, POINT_SEEDS["cstr"] + layer_index,
            lambda points: (points[:, 0] <= cstr_cfg.ca_max) & (points[:, 1] <= cstr_cfg.temperature_max_K),
        )
        cohorts["cstr"][layer] = np.c_[cstr_xy[:, 0], 1.0 - cstr_xy[:, 0], cstr_xy[:, 1]]
    return cohorts, cstr_cfg


def _save_cohorts(output: Path, cohorts: dict[str, dict[str, np.ndarray]]) -> None:
    folder = output / "cohorts"
    folder.mkdir(parents=True, exist_ok=True)
    headers = {"vdp": ["y1_0", "y2_0", "y3_0"], "penicillin": ["x2_0"],
               "cstr": ["CA_0", "CB_0", "T_0_K"]}
    for problem, layers in cohorts.items():
        for layer, values in layers.items():
            path = folder / f"{problem}_{layer}.csv"
            array = np.asarray(values)
            if array.ndim == 1:
                array = array[:, None]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sample_index", *headers[problem]])
                for index, row in enumerate(array):
                    writer.writerow([index, *row.tolist()])


def _predict_vdp(checkpoint: dict, states: np.ndarray) -> tuple[np.ndarray, float]:
    model = KKTPolicyValueNetwork(VDPTrainConfig()).eval()
    model.load_state_dict(checkpoint["model"])
    mean = np.asarray(checkpoint["normalization"]["mean"], float)
    std = np.asarray(checkpoint["normalization"]["std"], float)
    inputs = torch.tensor((states[:, :2] - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(3):
            model(inputs)
        started = time.perf_counter()
        _, controls = model(inputs)
        elapsed = time.perf_counter() - started
    return controls.numpy(), elapsed / len(states)


def _predict_penicillin(checkpoint: dict, values: np.ndarray) -> tuple[np.ndarray, float]:
    model = PenicillinPolicy().eval()
    model.load_state_dict(checkpoint["model"])
    mean = float(np.asarray(checkpoint["normalization"]["mean"]).reshape(-1)[0])
    std = float(np.asarray(checkpoint["normalization"]["std"]).reshape(-1)[0])
    inputs = torch.tensor(((values - mean) / std)[:, None], dtype=torch.float32)
    with torch.no_grad():
        for _ in range(3):
            model(inputs)
        started = time.perf_counter()
        _, controls = model(inputs)
        elapsed = time.perf_counter() - started
    return controls.numpy(), elapsed / len(values)


def _predict_cstr(checkpoint: dict, states: np.ndarray, cfg: EconomouScreenConfig) -> tuple[np.ndarray, float]:
    model = PolicyValue(cfg).eval()
    model.load_state_dict(checkpoint["model"])
    mean = np.asarray(checkpoint["state_mean"], float)
    std = np.asarray(checkpoint["state_std"], float)
    inputs = torch.tensor((states[:, [0, 2]] - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(3):
            model(inputs)
        started = time.perf_counter()
        _, controls = model(inputs)
        elapsed = time.perf_counter() - started
    return controls.numpy(), elapsed / len(states)


def _nominal_cstr_peak(state0: np.ndarray, controls: np.ndarray, cfg: EconomouScreenConfig) -> float:
    state = np.asarray(state0, float).copy()
    maximum = -np.inf
    for control in controls:
        peaks, state, _ = segment_event_audit(state, control, cfg)
        maximum = max(maximum, float(np.max(peaks)))
    return float(maximum)


def _evaluate_scalar_problem(
    problem: str,
    states: np.ndarray,
    controls: np.ndarray,
    inference_seconds: float,
) -> list[dict]:
    if problem == "vdp":
        corrector = AdaptiveEventHDSCorrector(
            vdp_ode, vdp_g, vdp_gdot, (-0.3, 1.0),
            AdaptiveEventHDSConfig(grid_size=GRID_SIZE, safety_margin=abs(THRESHOLD)),
        )
        duration = 0.5
    else:
        corrector = AdaptiveEventHDSCorrector(
            pen_ode, pen_g, pen_gdot, (0.0, PEN_UMAX),
            AdaptiveEventHDSConfig(grid_size=GRID_SIZE, safety_margin=abs(THRESHOLD)),
        )
        duration = PEN_DT
    rows: list[dict] = []
    for index, (stored_state, nominal) in enumerate(zip(states, controls)):
        initial = (
            np.asarray(stored_state, float)
            if problem == "vdp"
            else np.array([1.0, float(stored_state), 0.001, 250.0])
        )
        started = time.perf_counter()
        nominal_peak = float(corrector.audit(initial, nominal, duration))
        outcome = corrector.correct(initial, nominal, duration)
        accepted = bool(outcome.accepted)
        final_peak = (
            float(corrector.audit(initial, outcome.controls, duration)) if accepted else np.nan
        )
        # The independent full-sequence pass is the final authority.
        accepted = bool(accepted and final_peak <= THRESHOLD)
        hds_seconds = time.perf_counter() - started
        if problem == "vdp":
            nominal_objective = terminal_cost(initial, nominal, corrector, duration)
            applied_objective = (
                terminal_cost(initial, outcome.controls, corrector, duration) if accepted else np.nan
            )
        else:
            x2 = float(stored_state)
            nominal_objective = -terminal_product(x2, nominal, corrector)
            applied_objective = (
                -terminal_product(x2, outcome.controls, corrector) if accepted else np.nan
            )
        corrected_segments = int(sum(segment.corrected for segment in outcome.segments))
        delta = applied_objective - nominal_objective if accepted else np.nan
        rows.append({
            "sample_index": index,
            "nominal_max_g": nominal_peak,
            "accepted": accepted,
            "fallback": not accepted,
            "final_max_g": final_peak,
            "corrected_segments": corrected_segments,
            "corrected_trajectory": corrected_segments > 0,
            "nominal_objective_J": nominal_objective,
            "hds_objective_J": applied_objective,
            "hds_objective_change_J": delta,
            "hds_objective_change_percent_of_abs_nominal": (
                100.0 * delta / max(abs(nominal_objective), 1e-12) if accepted else np.nan
            ),
            "inference_seconds": inference_seconds,
            "hds_audit_correction_seconds": hds_seconds,
            "total_guard_bypassed_seconds": inference_seconds + hds_seconds,
            "domain_guard_deterministic_dispatch": True,
        })
    return rows


def _evaluate_cstr(
    states: np.ndarray,
    controls: np.ndarray,
    inference_seconds: float,
    cfg: EconomouScreenConfig,
) -> list[dict]:
    rows: list[dict] = []
    for index, (initial, nominal) in enumerate(zip(states, controls)):
        started = time.perf_counter()
        nominal_peak = _nominal_cstr_peak(initial, nominal, cfg)
        outcome = correct_with_threshold(initial, nominal, cfg, THRESHOLD)
        accepted = bool(outcome["accepted"])
        hds_seconds = time.perf_counter() - started
        nominal_objective = trajectory_objective(initial, nominal, cfg)
        applied_objective = (
            trajectory_objective(initial, outcome["controls"], cfg) if accepted else np.nan
        )
        delta = applied_objective - nominal_objective if accepted else np.nan
        corrected_segments = int(outcome["corrected_segments"])
        rows.append({
            "sample_index": index,
            "nominal_max_g": nominal_peak,
            "accepted": accepted,
            "fallback": not accepted,
            "final_max_g": float(outcome["final_peak"]),
            "corrected_segments": corrected_segments,
            "corrected_trajectory": corrected_segments > 0,
            "nominal_objective_J": nominal_objective,
            "hds_objective_J": applied_objective,
            "hds_objective_change_J": delta,
            "hds_objective_change_percent_of_abs_nominal": (
                100.0 * delta / max(abs(nominal_objective), 1e-12) if accepted else np.nan
            ),
            "inference_seconds": inference_seconds,
            "hds_audit_correction_seconds": hds_seconds,
            "total_guard_bypassed_seconds": inference_seconds + hds_seconds,
            "domain_guard_deterministic_dispatch": True,
        })
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        raw = list(csv.DictReader(handle))
    rows: list[dict] = []
    for row in raw:
        rows.append({
            **row,
            "sample_index": int(row["sample_index"]),
            "nominal_max_g": float(row["nominal_max_g"]),
            "accepted": row["accepted"].lower() == "true",
            "fallback": row["fallback"].lower() == "true",
            "final_max_g": float(row["final_max_g"]),
            "corrected_segments": int(row["corrected_segments"]),
            "corrected_trajectory": row["corrected_trajectory"].lower() == "true",
            "nominal_objective_J": float(row["nominal_objective_J"]),
            "hds_objective_J": float(row["hds_objective_J"]),
            "hds_objective_change_J": float(row["hds_objective_change_J"]),
            "hds_objective_change_percent_of_abs_nominal": float(
                row["hds_objective_change_percent_of_abs_nominal"]
            ),
            "inference_seconds": float(row["inference_seconds"]),
            "hds_audit_correction_seconds": float(row["hds_audit_correction_seconds"]),
            "total_guard_bypassed_seconds": float(row["total_guard_bypassed_seconds"]),
            "domain_guard_deterministic_dispatch": row["domain_guard_deterministic_dispatch"].lower() == "true",
        })
    return rows


def _layer_summary(rows: list[dict], control_segments: int) -> dict:
    nominal = np.asarray([row["nominal_max_g"] for row in rows], float)
    accepted = np.asarray([row["accepted"] for row in rows], bool)
    final = np.asarray([row["final_max_g"] for row in rows], float)
    corrected = np.asarray([row["corrected_segments"] for row in rows], float)
    objective_change = np.asarray([row["hds_objective_change_J"] for row in rows], float)
    objective_change_percent = np.asarray(
        [row["hds_objective_change_percent_of_abs_nominal"] for row in rows], float
    )
    return {
        "samples": len(rows),
        "nominal_violation_rate_percent": 100.0 * float(np.mean(nominal > 0.0)),
        "nominal_strict_rejection_rate_percent": 100.0 * float(np.mean(nominal > THRESHOLD)),
        "nominal_max_g": float(np.max(nominal)),
        "hds_acceptance_rate_percent": 100.0 * float(np.mean(accepted)),
        "fallback_rate_percent": 100.0 * float(np.mean(~accepted)),
        "final_max_g_on_accepted": float(np.nanmax(final)) if np.any(accepted) else np.nan,
        "corrected_trajectory_rate_percent": 100.0 * float(np.mean(corrected > 0)),
        "corrected_control_segment_rate_percent": 100.0 * float(np.sum(corrected) / (len(rows) * control_segments)),
        "mean_corrected_segments": float(np.mean(corrected)),
        "mean_hds_objective_change_J_on_accepted": float(np.nanmean(objective_change)),
        "mean_hds_objective_change_percent_on_accepted": float(np.nanmean(objective_change_percent)),
        "mean_inference_seconds": float(np.mean([row["inference_seconds"] for row in rows])),
        "mean_hds_audit_correction_seconds": float(
            np.mean([row["hds_audit_correction_seconds"] for row in rows])
        ),
        "mean_total_guard_bypassed_seconds": float(
            np.mean([row["total_guard_bypassed_seconds"] for row in rows])
        ),
        "default_deployment_deterministic_dispatch_rate_percent": 100.0,
    }


def _evaluate_seed_job(payload: tuple) -> dict:
    problem, seed, layers, cstr_config, output_string, resume = payload
    torch.set_num_threads(1)
    output = Path(output_string) / problem / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(problem, seed)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    summaries: dict[str, dict] = {}
    for layer, states in layers.items():
        csv_path = output / f"{layer}_per_sample.csv"
        if resume and csv_path.exists():
            rows = _read_rows(csv_path)
        else:
            values = np.asarray(states, float)
            if problem == "vdp":
                controls, inference = _predict_vdp(checkpoint, values)
                rows = _evaluate_scalar_problem(problem, values, controls, inference)
            elif problem == "penicillin":
                controls, inference = _predict_penicillin(checkpoint, values)
                rows = _evaluate_scalar_problem(problem, values, controls, inference)
            else:
                cfg = EconomouScreenConfig(**cstr_config)
                controls, inference = _predict_cstr(checkpoint, values, cfg)
                rows = _evaluate_cstr(values, controls, inference, cfg)
            _write_rows(csv_path, rows)
        summaries[layer] = _layer_summary(rows, 100 if problem == "cstr" else 10)
    result = {
        "problem": problem,
        "training_seed": seed,
        "method": "CA-KKT (linear cosine conflict-aware KKT continuation)",
        "checkpoint": str(checkpoint_path),
        "formal_protocol": False,
        "diagnostic_only": True,
        "diagnostic_label": "guard-bypassed OOD stress diagnostic",
        "layers": summaries,
    }
    _dump_json(output / "summary.json", result)
    return result


def _mean_sd(values: list[float]) -> dict:
    array = np.asarray(values, float)
    finite = array[np.isfinite(array)]
    return {
        "mean": float(np.mean(finite)) if len(finite) else np.nan,
        "sample_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "n": int(len(finite)),
    }


def _aggregate(summaries: list[dict], seeds: tuple[int, ...], samples: int) -> dict:
    metrics = (
        "nominal_violation_rate_percent",
        "nominal_strict_rejection_rate_percent",
        "nominal_max_g",
        "hds_acceptance_rate_percent",
        "fallback_rate_percent",
        "final_max_g_on_accepted",
        "corrected_trajectory_rate_percent",
        "corrected_control_segment_rate_percent",
        "mean_corrected_segments",
        "mean_hds_objective_change_J_on_accepted",
        "mean_hds_objective_change_percent_on_accepted",
        "mean_inference_seconds",
        "mean_hds_audit_correction_seconds",
        "mean_total_guard_bypassed_seconds",
        "default_deployment_deterministic_dispatch_rate_percent",
    )
    aggregate = {
        "formal_protocol": False,
        "diagnostic_only": True,
        "diagnostic_label": "guard-bypassed OOD stress diagnostic",
        "method": "CA-KKT (linear cosine conflict-aware KKT continuation)",
        "training_seeds": list(seeds),
        "seed_count": len(seeds),
        "samples_per_layer": samples,
        "ood_layers": {
            "near_ood_0_10pct": "outside the training box, within the 10% expanded box",
            "far_ood_10_20pct": "outside the 10% expanded box, within the 20% expanded box",
        },
        "initial_feasibility_filter": "retain only points satisfying the physical path constraints at t=0",
        "hds": {
            "integrator": "DOP853 adaptive step",
            "peak_location": "constraint-derivative stationary events plus segment endpoints",
            "rtol": 1e-10,
            "atol": 1e-12,
            "lambda_candidates": GRID_SIZE,
            "candidate_order": "ascending absolute distance from lambda=1 with first-safe early stop",
            "acceptance_threshold": THRESHOLD,
            "claim_limit": "continuous-time numerical audit evidence under the declared model and numerical settings",
        },
        "objective_comparison": (
            "HDS objective change relative to the same nominal extrapolated neural control; "
            "no OOD deterministic reference was solved and no OOD optimality gap is reported"
        ),
        "deployment_rule": (
            "all OOD points are dispatched by the domain guard to the deterministic solver; "
            "the neural-policy results intentionally bypass that guard for diagnosis only"
        ),
        "problems": {},
    }
    for problem in ("vdp", "penicillin", "cstr"):
        selected = [item for item in summaries if item["problem"] == problem]
        if len(selected) != len(seeds):
            raise ValueError(f"{problem}: expected {len(seeds)} summaries, found {len(selected)}")
        aggregate["problems"][problem] = {}
        for layer in LAYER_SPECS:
            aggregate["problems"][problem][layer] = {
                metric: _mean_sd([item["layers"][layer][metric] for item in selected])
                for metric in metrics
            }
    return aggregate


def _fmt(metric: dict, digits: int = 4) -> str:
    return f"{metric['mean']:.{digits}f} ± {metric['sample_sd']:.{digits}f}"


def _fmt_sci(metric: dict) -> str:
    return f"{metric['mean']:.4e} ± {metric['sample_sd']:.4e}"


def _write_table(path: Path, aggregate: dict) -> None:
    labels = {"vdp": "VDP", "penicillin": "Penicillin", "cstr": "Economou CSTR"}
    layer_labels = {"near_ood_0_10pct": "Near OOD (0–10%)", "far_ood_10_20pct": "Far OOD (10–20%)"}
    lines = [
        f"# CA-KKT {aggregate['seed_count']}-seed guard-bypassed OOD stress diagnostic",
        "",
        f"Values are mean ± sample SD across the {aggregate['seed_count']} fixed training seeds. Each seed is evaluated on the same {aggregate['samples_per_layer']} initial states per OOD layer.",
        "",
        "| Benchmark | OOD layer | Nominal violation (%) | HDS accepted (%) | Corrected segments | HDS-induced objective change ΔJ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for problem in ("vdp", "penicillin", "cstr"):
        for layer in LAYER_SPECS:
            row = aggregate["problems"][problem][layer]
            lines.append(
                f"| {labels[problem]} | {layer_labels[layer]} | "
                f"{_fmt(row['nominal_violation_rate_percent'])} | "
                f"{_fmt(row['hds_acceptance_rate_percent'])} | "
                f"{_fmt(row['mean_corrected_segments'])} | "
                f"{_fmt(row['mean_hds_objective_change_J_on_accepted'], 6)} |"
            )
    lines += [
        "",
        "Notes: ΔJ = J_HDS - J_nominal is the absolute objective change for the same OOD initial condition; it is not normalized and is not a gap to a deterministic reference. Positive ΔJ denotes objective degradation under the unified minimization convention (Penicillin uses J = -final x3). Objective units differ across benchmarks, so ΔJ magnitudes must not be compared between benchmarks. Nominal violation uses g > 0; HDS acceptance requires final g_max ≤ -1e-6. The omitted fallback rate is 100% minus the HDS acceptance rate. All OOD points would be dispatched to the deterministic solver by the domain guard in default deployment. The bypassed neural/HDS results are diagnostic numerical evidence, not an OOD safety or optimality guarantee.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_seeds(text: str) -> tuple[int, ...]:
    if text == "default":
        return DEFAULT_SEEDS
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("No seeds supplied")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-layer", type=int, default=100)
    parser.add_argument("--seeds", default="default", help="default or comma-separated integers")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.samples_per_layer < 1:
        raise ValueError("samples-per-layer must be positive")
    seeds = _parse_seeds(args.seeds)
    missing = [str(_checkpoint_path(problem, seed)) for problem in ("vdp", "penicillin", "cstr")
               for seed in seeds if not _checkpoint_path(problem, seed).exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(f"Output is not empty; pass --resume to reuse completed layers: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    cohorts, cstr_cfg = _make_cohorts(args.samples_per_layer)
    _save_cohorts(args.output, cohorts)
    protocol = {
        "formal_protocol": False,
        "diagnostic_only": True,
        "diagnostic_label": "guard-bypassed OOD stress diagnostic",
        "seeds": list(seeds),
        "samples_per_layer": args.samples_per_layer,
        "threshold": THRESHOLD,
        "lambda_grid_size": GRID_SIZE,
        "cstr_config": asdict(cstr_cfg),
    }
    _dump_json(args.output / "protocol.json", protocol)
    payloads = []
    cstr_raw = asdict(cstr_cfg)
    # Submit long CSTR jobs first so their tail does not serialize the run.
    for problem in ("cstr", "penicillin", "vdp"):
        for seed in seeds:
            payloads.append((problem, seed, cohorts[problem], cstr_raw, str(args.output), args.resume))
    summaries: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {executor.submit(_evaluate_seed_job, payload): (payload[0], payload[1]) for payload in payloads}
        for completed, future in enumerate(as_completed(future_to_job), start=1):
            problem, seed = future_to_job[future]
            summaries.append(future.result())
            print(f"[{completed}/{len(payloads)}] completed {problem} seed {seed}", flush=True)
    summaries.sort(key=lambda item: (item["problem"], item["training_seed"]))
    aggregate = _aggregate(summaries, seeds, args.samples_per_layer)
    _dump_json(args.output / "aggregate_summary.json", aggregate)
    _write_table(args.output / "aggregate_table.md", aggregate)
    print(args.output / "aggregate_table.md")


if __name__ == "__main__":
    main()
