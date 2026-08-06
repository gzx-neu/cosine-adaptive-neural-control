"""Diagnostic: audit a VALC plan made from an imperfect initial measurement.

The controller sees a clipped, uniformly perturbed initial condition while the
independent plant-side audit starts from the unperturbed frozen test condition.
This is explicitly a model-consistent measurement-error diagnostic, not a
robust safety guarantee.  It uses the final selected checkpoints and does not
retrain any model.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from kkt_collocation.run_vdp_ablation import AblationConfig, KKTPolicyValueNetwork, TrainConfig, constraint as vdp_g, constraint_derivative as vdp_gdot, lhs_states as vdp_lhs, predict as vdp_predict, vdp_ode
from kkt_collocation.run_penicillin_ablation import Config as PenConfig, Policy as PenPolicy, DT, UMAX, g as pen_g, gdot as pen_gdot, lhs as pen_lhs, ode as pen_ode, predict as pen_predict
from kkt_collocation.run_cstr_full_simulation import CSTRConfig, Policy as CSTRPolicy, lhs_states as cstr_lhs, make_corrector as make_cstr_corrector, predict as cstr_predict


def load_torch(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def perturb(values: np.ndarray, lower: np.ndarray, upper: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """Independent uniform measurement errors, scaled by declared domain widths."""
    noise = rng.uniform(-level, level, size=values.shape) * (upper - lower)
    return np.clip(values + noise, lower, upper)


def summarise(rows: list[dict]) -> dict:
    result = {}
    for name in sorted({r["benchmark"] for r in rows}):
        result[name] = {}
        for level in sorted({r["error_percent"] for r in rows if r["benchmark"] == name}):
            group = [r for r in rows if r["benchmark"] == name and r["error_percent"] == level]
            by_seed = []
            for seed in sorted({r["seed"] for r in group}):
                sub = [r for r in group if r["seed"] == seed]
                by_seed.append((np.mean([r["designed_accepted"] for r in sub]), np.mean([r["true_accepted"] for r in sub]), max(r["true_peak"] for r in sub)))
            a = np.asarray(by_seed, float)
            result[name][f"{level:g}%"] = {
                "samples": len(group), "design_acceptance_percent_mean": float(100 * a[:, 0].mean()),
                "true_initial_acceptance_percent_mean": float(100 * a[:, 1].mean()),
                "true_initial_acceptance_percent_sample_std": float(100 * a[:, 1].std(ddof=1)),
                "largest_true_peak": float(a[:, 2].max()),
            }
    return result


def evaluate(name: str, seed: int, truth: np.ndarray, measured: np.ndarray, controls: np.ndarray,
             corrector: HDSLambdaCorrector, duration: float, level: float, rows: list[dict]) -> None:
    for index, (x, xm, u) in enumerate(zip(truth, measured, controls)):
        outcome = corrector.correct(xm, u, duration)
        applied = outcome.controls if outcome.accepted else u
        true_peak = corrector.audit(x, applied, duration)
        rows.append({"benchmark": name, "seed": seed, "sample_index": index, "error_percent": level,
                     "designed_accepted": bool(outcome.accepted), "true_peak": float(true_peak),
                     "true_accepted": bool(outcome.accepted and true_peak <= corrector.config.acceptance_threshold)})


def main() -> None:
    out = ROOT / "kkt_collocation" / "results" / "eai_extension"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    levels = (1.0, 5.0)
    n = 100
    print("starting VDP diagnostic", flush=True)

    # VDP selected S checkpoint, using the first 100 frozen test points per seed.
    for seed in (20260751, 20260752, 20260753):
        directory = ROOT / "kkt_collocation" / "results" / f"final_multiseed_vdp900_penalty_seed{seed}"
        stored = load_torch(directory / "models.pth")
        cfg = AblationConfig(**stored["config"])
        model = KKTPolicyValueNetwork(TrainConfig()); model.load_state_dict(stored["S"]); model.eval()
        mean, std = map(np.asarray, stored["normalization"]["S"])
        truth = np.load(directory / "test_states.npy")[:n]
        corrector = HDSLambdaCorrector(vdp_ode, vdp_g, vdp_gdot, (-0.3, 1.0), HDSLambdaConfig(grid_size=31, max_step_fraction=100.0))
        for level in levels:
            measured2 = perturb(truth[:, :2], np.array([-.1, .9]), np.array([.1, 1.1]), level / 100, np.random.default_rng(731000 + seed + int(level)))
            measured = truth.copy(); measured[:, :2] = measured2
            controls, _ = vdp_predict(model, mean, std, measured, torch.device("cpu"))
            evaluate("VDP", seed, truth, measured, controls, corrector, cfg.duration, level, rows)
        print(f"finished VDP seed {seed}", flush=True)

    # Penicillin selected true-KKT checkpoint, again using frozen test points.
    for seed in (20260761, 20260762, 20260763):
        directory = ROOT / "kkt_collocation" / "results" / f"final_multiseed_penicillin400_penalty_seed{seed}"
        stored = load_torch(directory / "models.pth")
        model = PenPolicy(); model.load_state_dict(stored["true_KKT"]); model.eval()
        mean, std = map(float, stored["normalization"]["true_KKT"])
        truth_x2 = np.load(directory / "test_x2.npy")[:n]
        corrector = HDSLambdaCorrector(pen_ode, pen_g, pen_gdot, (0., UMAX), HDSLambdaConfig(grid_size=31, max_step_fraction=100.0))
        for level in levels:
            measured_x2 = perturb(truth_x2[:, None], np.array([.1]), np.array([.3]), level / 100, np.random.default_rng(732000 + seed + int(level))).reshape(-1)
            controls, _ = pen_predict(model, mean, std, measured_x2, torch.device("cpu"))
            truth = np.column_stack((np.ones(n), truth_x2, np.full(n, .001), np.full(n, 250.)))
            measured = np.column_stack((np.ones(n), measured_x2, np.full(n, .001), np.full(n, 250.)))
            evaluate("Penicillin", seed, truth, measured, controls, corrector, DT, level, rows)
        print(f"finished Penicillin seed {seed}", flush=True)

    # CSTR has seed-specific selected checkpoints and generated frozen test cohorts.
    labels = ROOT / "kkt_collocation" / "results" / "cstr_full_simulation_900" / "cstr_labels.pkl"
    import pickle
    with labels.open("rb") as handle: payload = pickle.load(handle)
    cfg = CSTRConfig(**payload["config"])
    for seed, test_seed in zip((20260718, 20260725, 20260726), (20260724, 20260825, 20260826)):
        directory = ROOT / "kkt_collocation" / "results" / "cstr_multiseed_test1200_900_clean"
        stored = load_torch(directory / f"cstr_seed{seed}.pth")
        model = CSTRPolicy(cfg); model.load_state_dict(stored["model"]); model.eval()
        truth = cstr_lhs(cfg, 400, test_seed)[:n]
        corrector = make_cstr_corrector(cfg)
        for level in levels:
            measured = perturb(truth, np.array(cfg.ca_range + cfg.temperature_range, float)[[0, 2]], np.array(cfg.ca_range + cfg.temperature_range, float)[[1, 3]], level / 100, np.random.default_rng(733000 + seed + int(level)))
            controls, _ = cstr_predict(model, np.asarray(stored["mean"]), np.asarray(stored["std"]), measured)
            evaluate("CSTR", seed, truth, measured, controls, corrector, cfg.zoh_duration, level, rows)
        print(f"finished CSTR seed {seed}", flush=True)

    csv_path = out / "initial_measurement_error_diagnostic.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = {"protocol": "For each frozen test initial condition p, VALC plans from a clipped uniformly perturbed measurement p_tilde; the applied sequence is independently HDS-audited from p under the same model.", "per_level": summarise(rows)}
    (out / "initial_measurement_error_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
