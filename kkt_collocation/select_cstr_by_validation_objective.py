"""Select CSTR S versus KKT checkpoints on independent reference validation sets.

The selection criterion is applied before the frozen deployment comparisons.
For each seed, both candidates are HDS-corrected and scored against 60
independent cold-start NLP references.  The selected branch minimizes the
mean relative objective difference among candidates accepted at every
validation point.
"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kkt_collocation.run_cstr_full_simulation import CSTRConfig, CSTRTranscription, Policy, lhs_states, make_corrector, objective, predict

SEEDS = (20260718, 20260725, 20260726)


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def score(model_path: Path, cfg: CSTRConfig, states: np.ndarray, reference: np.ndarray) -> dict:
    item = load(model_path); model = Policy(cfg); model.load_state_dict(item["model"]); model.eval()
    controls, _ = predict(model, np.asarray(item["mean"]), np.asarray(item["std"]), states)
    corrector = make_corrector(replace(cfg, hds_tolerance=1e-6))
    gaps = []; corrected = []
    for state, control, ref in zip(states, controls, reference):
        outcome = corrector.correct(state, control, cfg.zoh_duration)
        if not outcome.accepted:
            return {"accepted": False, "objective_gap_percent": None, "mean_corrected_segments": None}
        applied = outcome.controls
        gaps.append(abs(objective(state, applied, cfg) - ref) / max(abs(ref), 1e-12) * 100)
        corrected.append(sum(segment.corrected for segment in outcome.segments))
    return {"accepted": True, "objective_gap_percent": float(np.mean(gaps)),
            "objective_gap_sample_std_percent": float(np.std(gaps, ddof=1)),
            "mean_corrected_segments": float(np.mean(corrected))}


def main() -> None:
    labels_path = ROOT / "kkt_collocation" / "results" / "cstr_full_simulation_900" / "cstr_labels.pkl"
    with labels_path.open("rb") as handle: labels = pickle.load(handle)
    cfg = CSTRConfig(**labels["config"])
    source = ROOT / "kkt_collocation" / "results" / "cstr_kkt_ablation1200_900"
    report = {"criterion": "Select the accepted candidate with lower mean relative objective difference on an independent 60-point cold-start-NLP validation cohort.", "seeds": {}}
    for seed in SEEDS:
        states = lhs_states(cfg, 60, seed + 1000)
        nlp = CSTRTranscription(cfg)
        references = np.asarray([nlp.solve(state, controls=None)["objective"] for state in states], dtype=float)
        candidates = {
            "Supervised": score(source / f"cstr_seed{seed}_supervised.pth", cfg, states, references),
            "KKT-refined": score(source / f"cstr_seed{seed}_kkt_refined.pth", cfg, states, references),
        }
        eligible = [(value["objective_gap_percent"], name) for name, value in candidates.items() if value["accepted"]]
        selected = min(eligible)[1] if eligible else "offline-optimizer dispatch"
        report["seeds"][str(seed)] = {"validation_seed": seed + 1000, "reference_points": 60,
                                      "candidates": candidates, "selected_branch": selected}
        print(seed, report["seeds"][str(seed)], flush=True)
    output = ROOT / "kkt_collocation" / "results" / "eai_extension" / "cstr_validation_objective_selection.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
