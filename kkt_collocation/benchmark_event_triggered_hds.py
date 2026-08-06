"""Compare legacy and event-triggered lambda searches on identical final tests.

Both variants perform one HDS peak inspection per ZOH segment.  This is needed
to identify a violation and to re-audit downstream states after an earlier
correction.  The optimized variant only removes the redundant lambda=1
simulation after that nominal segment has already been established unsafe.
It therefore preserves the numerical safety logic and selected controls.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kkt_collocation.run_penicillin_ablation import (  # noqa: E402
    DT as PEN_DURATION,
    Policy as PenicillinPolicy,
    UMAX as PEN_UMAX,
    g as pen_constraint,
    gdot as pen_constraint_derivative,
    ode as pen_ode,
)
from kkt_collocation.run_vdp_ablation import (  # noqa: E402
    constraint as vdp_constraint,
    constraint_derivative as vdp_constraint_derivative,
    vdp_ode,
)
from kkt_collocation.train_vdp_kkt_policy import KKTPolicyValueNetwork, TrainConfig
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector


def _vdp_policy(result_dir: Path, states: np.ndarray) -> tuple[np.ndarray, str, float]:
    with (result_dir / "summary.json").open(encoding="utf-8") as handle:
        selected = json.load(handle)["adaptive_gate"]["selected_branch"]
    checkpoint = torch.load(result_dir / "models.pth", map_location="cpu", weights_only=False)
    key = "S" if selected == "S" else "KKT"
    model = KKTPolicyValueNetwork(TrainConfig()).eval()
    model.load_state_dict(checkpoint[key])
    mean, std = (np.asarray(value, dtype=np.float32) for value in checkpoint["normalization"][key])
    with torch.no_grad():
        _, controls = model(torch.tensor((states[:, :2] - mean) / std, dtype=torch.float32))
    return controls.numpy(), selected, 0.5


def _penicillin_policy(result_dir: Path, x2: np.ndarray) -> tuple[np.ndarray, str, float]:
    with (result_dir / "summary.json").open(encoding="utf-8") as handle:
        selected = json.load(handle)["adaptive_gate"]["selected_branch"]
    checkpoint = torch.load(result_dir / "models.pth", map_location="cpu", weights_only=False)
    key = "S" if selected == "S" else "true_KKT"
    model = PenicillinPolicy().eval()
    model.load_state_dict(checkpoint[key])
    mean, std = checkpoint["normalization"][key]
    with torch.no_grad():
        _, controls = model(torch.tensor(((x2 - float(mean)) / float(std))[:, None], dtype=torch.float32))
    return controls.numpy(), selected, PEN_DURATION


def _evaluate_variant(name: str, corrector: HDSLambdaCorrector, states: np.ndarray, controls: np.ndarray,
                      duration: float, *, detect_then_correct: bool = False) -> list[dict]:
    rows = []
    for index, (state, nominal) in enumerate(zip(states, controls)):
        start = time.perf_counter()
        certificate = (corrector.correct_detect_then_correct(state, nominal, duration)
                       if detect_then_correct else corrector.correct(state, nominal, duration))
        elapsed = time.perf_counter() - start
        segments = certificate.segments
        rows.append({
            "variant": name,
            "sample_index": index,
            "accepted": certificate.accepted,
            "fallback": certificate.requires_reoptimization,
            "raw_max_hds_g": max(segment.nominal_peak_g for segment in segments),
            "applied_max_hds_g": max((segment.corrected_peak_g for segment in segments if segment.corrected_peak_g is not None), default=np.nan),
            "applied_controls": None if certificate.controls is None else certificate.controls.tolist(),
            "corrected_segments": sum(segment.corrected for segment in segments),
            "candidate_evaluations": sum(segment.candidate_evaluations for segment in segments),
            "correction_seconds": elapsed,
        })
    return rows


def _summary(rows: list[dict]) -> dict:
    grouped = {}
    for name in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == name]
        values = lambda key: np.asarray([row[key] for row in group], dtype=float)
        grouped[name] = {
            "samples": len(group),
            "accepted_rate": float(np.mean(values("accepted"))),
            "fallback_rate": float(np.mean(values("fallback"))),
            "raw_violation_rate": float(np.mean(values("raw_max_hds_g") > 1e-8)),
            "max_applied_hds_g": float(np.nanmax(values("applied_max_hds_g"))),
            "mean_corrected_segments": float(np.mean(values("corrected_segments"))),
            "mean_candidate_evaluations": float(np.mean(values("candidate_evaluations"))),
            "mean_correction_seconds": float(np.mean(values("correction_seconds"))),
            "median_correction_seconds": float(np.median(values("correction_seconds"))),
        }
    legacy = grouped["legacy_repeat_lambda_1"]
    optimized = grouped["event_triggered_skip_lambda_1"]
    report = {
        "variants": grouped,
        "candidate_evaluation_reduction": float(1.0 - optimized["mean_candidate_evaluations"] / legacy["mean_candidate_evaluations"]) if legacy["mean_candidate_evaluations"] else 0.0,
        "mean_time_reduction": float(1.0 - optimized["mean_correction_seconds"] / legacy["mean_correction_seconds"]),
        "speedup": float(legacy["mean_correction_seconds"] / optimized["mean_correction_seconds"]),
    }
    if "offline_detect_then_correct" in grouped:
        batch = grouped["offline_detect_then_correct"]
        report["detect_then_correct_vs_one_sweep"] = {
            "time_ratio": float(batch["mean_correction_seconds"] / optimized["mean_correction_seconds"]),
            "time_change": float(batch["mean_correction_seconds"] / optimized["mean_correction_seconds"] - 1.0),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("vdp", "penicillin"), required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vdp-result-dir", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "final_vdp_900_adaptive_true_kkt_e200_test400_seed20260731")
    parser.add_argument("--penicillin-result-dir", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "final_penicillin_true_kkt_e200_w001_test400_seed20260741")
    args = parser.parse_args()
    if args.problem == "vdp":
        all_states = np.load(args.vdp_result_dir / "test_states.npy")
        indices = np.linspace(0, len(all_states) - 1, args.samples, dtype=int)
        states = all_states[indices]
        controls, selected, duration = _vdp_policy(args.vdp_result_dir, states)
        builder = lambda skip: HDSLambdaCorrector(vdp_ode, vdp_constraint, vdp_constraint_derivative, (-.3, 1.), HDSLambdaConfig(grid_size=31, max_step_fraction=100., skip_known_unsafe_nominal_candidate=skip))
    else:
        x2_all = np.load(args.penicillin_result_dir / "test_x2.npy")
        indices = np.linspace(0, len(x2_all) - 1, args.samples, dtype=int)
        x2 = x2_all[indices]
        states = np.column_stack((np.ones(len(x2)), x2, np.full(len(x2), .001), np.full(len(x2), 250.)))
        controls, selected, duration = _penicillin_policy(args.penicillin_result_dir, x2)
        builder = lambda skip: HDSLambdaCorrector(pen_ode, pen_constraint, pen_constraint_derivative, (0., PEN_UMAX), HDSLambdaConfig(grid_size=31, max_step_fraction=100., skip_known_unsafe_nominal_candidate=skip))
    if not 1 <= args.samples <= len(states):
        raise ValueError("--samples must be in [1, number of final test points]")
    legacy_rows = _evaluate_variant("legacy_repeat_lambda_1", builder(False), states, controls, duration)
    optimized_rows = _evaluate_variant("event_triggered_skip_lambda_1", builder(True), states, controls, duration)
    batch_rows = _evaluate_variant("offline_detect_then_correct", builder(True), states, controls, duration,
                                  detect_then_correct=True)
    matches = []
    for legacy, optimized in zip(legacy_rows, optimized_rows):
        same_acceptance = legacy["accepted"] == optimized["accepted"]
        if legacy["accepted"] and optimized["accepted"]:
            same_controls = np.allclose(legacy["applied_controls"], optimized["applied_controls"], rtol=0., atol=1e-12)
        else:
            same_controls = legacy["applied_controls"] == optimized["applied_controls"]
        matches.append(bool(same_acceptance and same_controls))
    acceptance_matches = [legacy["accepted"] == batch["accepted"]
                          for legacy, batch in zip(optimized_rows, batch_rows)]
    report = _summary(legacy_rows + optimized_rows + batch_rows)
    report.update({
        "problem": args.problem,
        "adaptive_selected_policy": selected,
        "test_indices": indices.tolist(),
        "exact_acceptance_and_control_match_rate": float(np.mean(matches)),
        "detect_then_correct_acceptance_match_rate": float(np.mean(acceptance_matches)),
        "safety_note": "Both variants retain segmentwise HDS inspection. The optimized variant only omits a duplicated lambda=1 candidate after its nominal peak was already found unsafe.",
    })
    args.output.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for row in legacy_rows + optimized_rows:
        csv_rows.append({key: value for key, value in row.items() if key != "applied_controls"})
    with (args.output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
