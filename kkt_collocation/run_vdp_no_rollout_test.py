"""Controlled VDP ablation removing only rollout consistency.

The final VDP data, training seeds, frozen validation/test states, validation
gate, HDS corrector, and cold-start reference objectives are retained. The
single change is ``rollout_weight=0`` in the existing training routine.
"""
from __future__ import annotations

import copy
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.adaptive_kkt_gate import AdaptiveKKTThresholds, audit_raw_hds_peaks
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from kkt_collocation.run_vdp_ablation import AblationConfig, constraint, constraint_derivative, load_data, predict, terminal_cost, train_policy, vdp_ode

SEEDS = (20260751, 20260752, 20260753)


def mean_sd(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "sample_std": float(x.std(ddof=1))}


def main() -> None:
    out = ROOT / "kkt_collocation" / "results" / "vdp_no_rollout_w0"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = ROOT / "kkt_collocation" / "data" / "vdp_kkt_30x30_warm_s10_margin1e3.pkl"
    initial, controls_ref, objectives, duals, _ = load_data(data_path, device)
    with (ROOT / "kkt_collocation" / "results" / "jiang_fu_matched400_comparison" / "per_point_seed.csv").open(newline="", encoding="utf-8") as handle:
        refs = [row for row in csv.DictReader(handle) if row["problem"] == "VDP" and int(row["seed"]) == SEEDS[0]]
    if len(refs) != 400:
        raise RuntimeError("Expected the 400-point frozen VDP Jiang--Fu reference cohort")
    reference_objective = np.asarray([float(row["jiang_objective"]) for row in refs])

    rows: list[dict] = []
    gate_reports: dict[str, dict] = {}
    for seed in SEEDS:
        source = ROOT / "kkt_collocation" / "results" / f"final_multiseed_vdp900_penalty_seed{seed}"
        validation_states = np.load(source / "validation_states.npy")
        test_states = np.load(source / "test_states.npy")
        cfg = AblationConfig(epochs=200, seed=seed, lambda_grid_size=31, rollout_weight=0.0,
                             validation_samples=len(validation_states), test_samples=len(test_states), kkt_weight=1e-3)
        corrector = HDSLambdaCorrector(vdp_ode, constraint, constraint_derivative, (-.3, 1.),
                                       HDSLambdaConfig(grid_size=31, max_step_fraction=100.0))
        print(f"training VDP seed {seed} without rollout consistency", flush=True)
        supervised, mean, std = train_policy(initial, controls_ref, objectives, duals, False, cfg, device)
        val_controls, _ = predict(supervised, mean, std, validation_states, device)
        gate = audit_raw_hds_peaks(
            np.asarray([corrector.audit(x, u, cfg.duration) for x, u in zip(validation_states, val_controls)]),
            AdaptiveKKTThresholds(numerical_violation_tolerance=1e-8, allowed_violation_rate=.05,
                                  rate_normalized_violation=.025, allowed_normalized_peak_violation=.03,
                                  engineering_constraint_scale=.4),
        )
        selected = "Supervised"; model = supervised; selected_mean, selected_std = mean, std
        if gate.kkt_refinement_required:
            model, selected_mean, selected_std = train_policy(initial, controls_ref, objectives, duals, True, cfg, device,
                                                               model=copy.deepcopy(supervised), epochs=20, learning_rate=1e-5)
            selected = "KKT-refined"
        gate_reports[str(seed)] = {"selected_branch": selected, "gate": asdict(gate)}
        torch.save({"model": model.state_dict(), "mean": selected_mean, "std": selected_std, "config": asdict(cfg),
                    "selected_branch": selected}, out / f"vdp_seed{seed}.pth")
        nominal_controls, inference_seconds = predict(model, selected_mean, selected_std, test_states, device)
        for index, (state, nominal) in enumerate(zip(test_states, nominal_controls)):
            raw_peak = corrector.audit(state, nominal, cfg.duration)
            raw_cost = terminal_cost(state, nominal, corrector, cfg.duration)
            start = time.perf_counter(); outcome = corrector.correct(state, nominal, cfg.duration); filter_seconds = time.perf_counter() - start
            if not outcome.accepted:
                raise RuntimeError(f"unexpected HDS dispatch at seed={seed}, point={index}")
            applied_peak = corrector.audit(state, outcome.controls, cfg.duration)
            applied_cost = terminal_cost(state, outcome.controls, corrector, cfg.duration)
            ref = reference_objective[index]
            rows.append({"training_seed": seed, "sample_index": index, "selected_branch": selected,
                         "y1_0": state[0], "y2_0": state[1], "nominal_hds_max_g": raw_peak,
                         "applied_hds_max_g": applied_peak, "nominal_cost": raw_cost, "applied_cost": applied_cost,
                         "reference_objective": ref,
                         "relative_objective_difference_percent": abs(applied_cost-ref)/max(abs(ref), 1e-12)*100,
                         "corrected_segments": int(sum(s.corrected for s in outcome.segments)),
                         "filter_seconds": filter_seconds, "inference_seconds": inference_seconds})
        print(f"evaluated VDP seed {seed}", flush=True)

    with (out / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metrics = {}
    for seed in SEEDS:
        group = [r for r in rows if r["training_seed"] == seed]
        metrics[str(seed)] = {"nominal_violation_rate_percent": float(100*np.mean([r["nominal_hds_max_g"] > 1e-8 for r in group])),
                              "accepted_peak_max": float(max(r["applied_hds_max_g"] for r in group)),
                              "mean_corrected_segments": float(np.mean([r["corrected_segments"] for r in group])),
                              "relative_objective_difference_percent": mean_sd([r["relative_objective_difference_percent"] for r in group])}
    report = {"controlled_change": "VDP rollout consistency weight changed from 0.5 to 0; all other final-protocol settings held fixed.",
              "rollout_weight": 0.0, "gate_reports": gate_reports, "per_seed": metrics,
              "aggregate": {"nominal_violation_rate_percent": mean_sd([metrics[str(s)]["nominal_violation_rate_percent"] for s in SEEDS]),
                            "mean_corrected_segments": mean_sd([metrics[str(s)]["mean_corrected_segments"] for s in SEEDS]),
                            "relative_objective_difference_percent": mean_sd([metrics[str(s)]["relative_objective_difference_percent"]["mean"] for s in SEEDS])},
              "reference_note": "All three VDP runs use the unchanged 400-point Jiang--Fu cold-start reference cohort in the frozen test-state order."}
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
