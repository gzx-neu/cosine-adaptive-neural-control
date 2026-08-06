"""CSTR controlled test of the VDP-style rollout-consistency loss.

Only the training loss changes from the frozen CSTR protocol: a fixed
``0.5 * MSE(predicted value, differentiable rollout objective)`` term is added
to both supervised training and, if selected by the unchanged validation gate,
KKT refinement. Labels, seeds, gate, HDS corrector, and test cohorts are held
fixed. The seed-20260718 test cohort is also matched to the existing cold-start
NLP CSV, so its relative objective difference is directly comparable.
"""
from __future__ import annotations

import copy
import csv
import json
import os
import pickle
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
from kkt_collocation.run_cstr_full_simulation import CSTRConfig, Policy, lhs_states, make_corrector, objective, predict, train_branch

TRAIN_SEEDS = (20260718, 20260725, 20260726)
TEST_SEEDS = (20260724, 20260825, 20260826)
ROLLOUT_WEIGHT = 0.5


def mean_sd(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "sample_std": float(x.std(ddof=1))}


def main() -> None:
    labels_path = ROOT / "kkt_collocation" / "results" / "cstr_full_simulation_900" / "cstr_labels.pkl"
    out = ROOT / "kkt_collocation" / "results" / "cstr_rollout_consistency_w05"
    out.mkdir(parents=True, exist_ok=True)
    with labels_path.open("rb") as handle:
        data = pickle.load(handle)
    cfg = CSTRConfig(**data["config"])

    # Existing reference data for the seed-20260718 matched 400-point test.
    reference_path = ROOT / "kkt_collocation" / "results" / "cstr_matched400_coldstart_900" / "cstr_coldstart_nlp_comparison.csv"
    with reference_path.open(newline="", encoding="utf-8") as handle:
        reference = list(csv.DictReader(handle))

    rows: list[dict] = []
    gate_reports: dict[str, dict] = {}
    for seed, test_seed in zip(TRAIN_SEEDS, TEST_SEEDS):
        print(f"training seed {seed} with rollout weight {ROLLOUT_WEIGHT}", flush=True)
        supervised, mean, std, _ = train_branch(data, cfg, kkt=False, epochs=300, seed=seed, rollout_weight=ROLLOUT_WEIGHT)
        validation = lhs_states(cfg, 60, seed + 1000)
        validation_controls, _ = predict(supervised, mean, std, validation)
        corrector = make_corrector(cfg)
        gate = audit_raw_hds_peaks(
            np.asarray([corrector.audit(x, u, cfg.zoh_duration) for x, u in zip(validation, validation_controls)]),
            AdaptiveKKTThresholds(allowed_violation_rate=.05, rate_normalized_violation=.025,
                                  allowed_normalized_peak_violation=.03,
                                  engineering_constraint_scale=5., numerical_violation_tolerance=1e-8),
        )
        selected = "Supervised"
        model = supervised
        if gate.kkt_refinement_required:
            model, mean, std, _ = train_branch(data, cfg, kkt=True, epochs=30, seed=seed + 500,
                                                base=copy.deepcopy(supervised), rollout_weight=ROLLOUT_WEIGHT)
            selected = "KKT-refined"
        gate_reports[str(seed)] = {"selected_branch": selected, "gate": asdict(gate)}
        torch.save({"model": model.state_dict(), "mean": mean, "std": std, "config": asdict(cfg),
                    "rollout_weight": ROLLOUT_WEIGHT, "selected_branch": selected}, out / f"cstr_seed{seed}.pth")

        states = lhs_states(cfg, 400, test_seed)
        controls, inference = predict(model, mean, std, states)
        if seed == 20260718:
            expected = np.array([[float(r["CA0"]), float(r["T0"])] for r in reference])
            if not np.allclose(states, expected, rtol=0, atol=1e-12):
                raise RuntimeError("The matched cold-start reference states do not match the frozen seed-20260718 cohort")
        for index, (state, nominal) in enumerate(zip(states, controls)):
            raw_peak = corrector.audit(state, nominal, cfg.zoh_duration)
            raw_objective = objective(state, nominal, cfg)
            start = time.perf_counter(); outcome = corrector.correct(state, nominal, cfg.zoh_duration); filter_seconds = time.perf_counter() - start
            if not outcome.accepted:
                raise RuntimeError(f"unexpected HDS dispatch at seed={seed}, index={index}")
            applied_peak = corrector.audit(state, outcome.controls, cfg.zoh_duration)
            applied_objective = objective(state, outcome.controls, cfg)
            ref_obj = float(reference[index]["reference_objective"]) if seed == 20260718 else np.nan
            rel = abs(applied_objective-ref_obj) / max(abs(ref_obj), 1e-12) * 100 if np.isfinite(ref_obj) else np.nan
            rows.append({"training_seed": seed, "test_seed": test_seed, "sample_index": index,
                         "CA0": state[0], "T0": state[1], "selected_branch": selected,
                         "nominal_hds_max_g": raw_peak, "applied_hds_max_g": applied_peak,
                         "nominal_objective": raw_objective, "applied_objective": applied_objective,
                         "reference_objective": ref_obj, "relative_objective_difference_percent": rel,
                         "corrected_segments": int(sum(s.corrected for s in outcome.segments)),
                         "filter_seconds": filter_seconds, "inference_seconds": inference})
        print(f"evaluated seed {seed}", flush=True)

    fields = list(rows[0])
    with (out / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    by_seed = {}
    for seed in TRAIN_SEEDS:
        group = [r for r in rows if r["training_seed"] == seed]
        by_seed[str(seed)] = {
            "nominal_violation_rate_percent": float(100*np.mean([r["nominal_hds_max_g"] > 1e-8 for r in group])),
            "accepted_peak_max_K": float(max(r["applied_hds_max_g"] for r in group)),
            "mean_corrected_segments": float(np.mean([r["corrected_segments"] for r in group])),
            "mean_filter_seconds": float(np.mean([r["filter_seconds"] for r in group])),
        }
    strict = [r["relative_objective_difference_percent"] for r in rows if np.isfinite(r["relative_objective_difference_percent"])]
    report = {"controlled_change": "Added rollout-consistency loss with fixed weight 0.5 to CSTR supervised and KKT training only.",
              "rollout_weight": ROLLOUT_WEIGHT, "gate_reports": gate_reports, "per_seed": by_seed,
              "matched_seed20260718_coldstart_nlp": {"points": len(strict), "relative_objective_difference_percent": mean_sd(strict)},
              "note": "The reference objective and test states are the unchanged cstr_matched400_coldstart_900 protocol."}
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
