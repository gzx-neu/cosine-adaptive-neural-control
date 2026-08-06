"""Guard-bypassed CSTR out-of-domain stress diagnostic.

The declared deployment policy never evaluates the neural policy here: an
out-of-domain initial state goes directly to the offline optimizer.  This
script intentionally bypasses that guard solely to quantify nominal
extrapolation and the instance-wise behaviour of HDS--lambda correction.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control import BoxOperatingDomain, assess_initial_state  # noqa: E402
from kkt_collocation.run_cstr_full_simulation import (  # noqa: E402
    CSTRConfig, Policy, make_corrector, objective,
)


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def ring(cfg: CSTRConfig, n: int, expansion: float, seed: int) -> np.ndarray:
    """LHS-like uniform expanded-box ring, retaining only initially safe points."""
    rng = np.random.default_rng(seed)
    lower = np.array((cfg.ca_range[0], cfg.temperature_range[0]))
    upper = np.array((cfg.ca_range[1], cfg.temperature_range[1]))
    ext_lower = lower - expansion * (upper - lower)
    ext_upper = upper + expansion * (upper - lower)
    keep: list[np.ndarray] = []
    total = 0
    while total < n:
        draw = rng.uniform(ext_lower, ext_upper, size=(max(8 * n, 128), 2))
        outside = np.any((draw < lower) | (draw > upper), axis=1)
        safe = draw[:, 1] <= cfg.temperature_max
        retained = draw[outside & safe]
        keep.append(retained)
        total += len(retained)
    return np.vstack(keep)[:n]


def predictor(checkpoint: dict, cfg: CSTRConfig):
    model = Policy(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mean, std = np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"])

    def predict(states: np.ndarray) -> tuple[np.ndarray, float]:
        x = torch.tensor((states - mean) / std, dtype=torch.float32)
        start = time.perf_counter()
        with torch.no_grad():
            _, controls = model(x)
        return controls.numpy(), (time.perf_counter() - start) / len(states)
    return predict


def run_layer(states: np.ndarray, predict, cfg: CSTRConfig) -> list[dict]:
    domain = BoxOperatingDomain((cfg.ca_range[0], cfg.temperature_range[0]),
                                (cfg.ca_range[1], cfg.temperature_range[1]),
                                state_indices=(0, 1), name="CSTR operating domain")
    corrector = make_corrector(cfg)
    controls, inference = predict(states)
    rows: list[dict] = []
    for index, (state, nominal) in enumerate(zip(states, controls)):
        assessment = assess_initial_state(state, domain, [lambda x: cfg.temperature_max - x[1]])
        raw_peak = corrector.audit(state, nominal, cfg.zoh_duration)
        raw_objective = objective(state, nominal, cfg)
        result = corrector.correct(state, nominal, cfg.zoh_duration)
        accepted = bool(result.accepted)
        applied = result.controls if accepted else None
        applied_peak = corrector.audit(state, applied, cfg.zoh_duration) if accepted else np.nan
        applied_objective = objective(state, applied, cfg) if accepted else np.nan
        lambdas = np.array([s.lambda_value for s in result.segments if s.lambda_value is not None], dtype=float)
        rows.append({
            "sample_index": index, "CA0": state[0], "T0": state[1],
            "domain_guard_use_policy": assessment.use_policy,
            "domain_guard_action": assessment.action,
            "raw_hds_max_g": raw_peak,
            "accepted_after_hds_lambda": accepted,
            "applied_hds_max_g": applied_peak,
            "corrected_segments": int(sum(s.corrected for s in result.segments)),
            "mean_abs_lambda_minus_one": float(np.mean(np.abs(lambdas - 1.0))) if len(lambdas) else np.nan,
            "objective_change_from_nominal": applied_objective - raw_objective if accepted else np.nan,
            "inference_seconds": inference,
        })
    return rows


def summarise(rows: list[dict]) -> dict:
    raw = np.asarray([r["raw_hds_max_g"] for r in rows], dtype=float)
    accepted = np.asarray([r["accepted_after_hds_lambda"] for r in rows], dtype=bool)
    applied = np.asarray([r["applied_hds_max_g"] for r in rows], dtype=float)
    return {
        "samples": len(rows),
        "raw_violation_rate": float(np.mean(raw > 0.0)),
        "maximum_raw_hds_g": float(raw.max()),
        "bypass_guard_accepted_rate": float(accepted.mean()),
        "bypass_guard_fallback_rate": float(1.0 - accepted.mean()),
        "maximum_applied_hds_g_on_accepted": float(np.nanmax(applied)) if accepted.any() else None,
        "mean_corrected_segments": float(np.mean([r["corrected_segments"] for r in rows])),
        "mean_abs_lambda_minus_one": float(np.nanmean([r["mean_abs_lambda_minus_one"] for r in rows])),
        "mean_objective_change_from_nominal": float(np.nanmean([r["objective_change_from_nominal"] for r in rows])),
        "default_deployment_offline_fallback_rate": float(np.mean([not r["domain_guard_use_policy"] for r in rows])),
    }


def read_rows(path: Path) -> list[dict]:
    """Load a completed layer so interrupted diagnostics can resume exactly."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[dict] = []
    for row in raw_rows:
        rows.append({
            **row,
            "raw_hds_max_g": float(row["raw_hds_max_g"]),
            "accepted_after_hds_lambda": row["accepted_after_hds_lambda"].strip().lower() == "true",
            "applied_hds_max_g": float(row["applied_hds_max_g"]),
            "corrected_segments": int(row["corrected_segments"]),
            "mean_abs_lambda_minus_one": float(row["mean_abs_lambda_minus_one"]),
            "objective_change_from_nominal": float(row["objective_change_from_nominal"]),
            "domain_guard_use_policy": row["domain_guard_use_policy"].strip().lower() == "true",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "kkt_collocation" / "results" / "cstr_full_simulation_900")
    parser.add_argument("--output", type=Path, default=ROOT / "kkt_collocation" / "results" / "domain_stress_cstr_900_seed20260722")
    parser.add_argument("--samples-per-layer", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--safety-margin", type=float, default=1e-6)
    parser.add_argument("--resume", action="store_true", help="Reuse complete existing layer CSV files.")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.experiment / "cstr_supervised.pth")
    cfg = replace(CSTRConfig(**checkpoint["config"]), hds_tolerance=args.safety_margin)
    predict = predictor(checkpoint, cfg)
    layers = {"near_10_percent": ring(cfg, args.samples_per_layer, 0.10, args.seed),
              "far_20_percent": ring(cfg, args.samples_per_layer, 0.20, args.seed + 1)}
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"problem": "CSTR", "checkpoint": str(args.experiment / "cstr_supervised.pth"),
              "policy_key": "Supervised", "protocol": "Out-of-domain robustness stress test; no distribution-level safety or optimality guarantee is claimed.",
              "deployment_rule": "All out-of-domain initial states are routed directly to the offline optimizer; HDS-lambda results below intentionally bypass that guard for diagnosis only.",
              "safety_rule": f"event-located HDS peak <= -{args.safety_margin:g}",
              "layers": {}}
    for name, states in layers.items():
        csv_path = args.output / f"{name}_per_sample.csv"
        rows = read_rows(csv_path) if args.resume and csv_path.exists() else run_layer(states, predict, cfg)
        if not (args.resume and csv_path.exists()):
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        report["layers"][name] = summarise(rows)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
