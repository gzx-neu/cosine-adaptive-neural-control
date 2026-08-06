"""Diagnose CSTR objective quality of pre-existing checkpoints on frozen references.

This is a diagnostic screen only.  It must not be used to select a final model
from the cold-start test cohort; any replacement protocol must select its
hyperparameters on a separate validation set.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kkt_collocation.run_cstr_full_simulation import CSTRConfig, Policy, make_corrector, objective, predict


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, float)
    return {"mean": float(a.mean()), "sample_std": float(a.std(ddof=1))}


def main() -> None:
    source = ROOT / "kkt_collocation" / "results" / "cstr_matched400_coldstart_900" / "cstr_coldstart_nlp_comparison.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    states = np.asarray([[float(row["CA0"]), float(row["T0"])] for row in rows])
    reference = np.asarray([float(row["reference_objective"]) for row in rows])
    labels = ROOT / "kkt_collocation" / "results" / "cstr_full_simulation_900" / "cstr_labels.pkl"
    import pickle
    with labels.open("rb") as handle: data = pickle.load(handle)
    cfg = CSTRConfig(**data["config"]); corrector = make_corrector(cfg)
    candidates = {
        "base_supervised": ROOT / "kkt_collocation" / "results" / "cstr_full_simulation_900" / "cstr_supervised.pth",
        "base_kkt": ROOT / "kkt_collocation" / "results" / "cstr_full_simulation_900" / "cstr_kkt_refined.pth",
        "final_seed_18": ROOT / "kkt_collocation" / "results" / "cstr_multiseed_test1200_900_clean" / "cstr_seed20260718.pth",
        "final_seed_25": ROOT / "kkt_collocation" / "results" / "cstr_multiseed_test1200_900_clean" / "cstr_seed20260725.pth",
        "final_seed_26": ROOT / "kkt_collocation" / "results" / "cstr_multiseed_test1200_900_clean" / "cstr_seed20260726.pth",
    }
    report = {"note": "Post-hoc diagnostic only; final hyperparameters must be selected without this cohort.", "candidates": {}}
    for name, path in candidates.items():
        item = load(path); model = Policy(cfg); model.load_state_dict(item["model"]); model.eval()
        controls, _ = predict(model, np.asarray(item["mean"]), np.asarray(item["std"]), states)
        gaps = []; accepted = 0; corrected_counts = []
        for state, control, ref in zip(states, controls, reference):
            result = corrector.correct(state, control, cfg.zoh_duration)
            if not result.accepted:
                continue
            applied = result.controls
            gaps.append(abs(objective(state, applied, cfg) - ref) / max(abs(ref), 1e-12) * 100)
            accepted += 1; corrected_counts.append(sum(s.corrected for s in result.segments))
        report["candidates"][name] = {"accepted": accepted, "objective_difference_percent": stats(gaps),
                                        "mean_corrected_segments": float(np.mean(corrected_counts)) if corrected_counts else None}
        print(name, report["candidates"][name], flush=True)
    output = ROOT / "kkt_collocation" / "results" / "eai_extension" / "cstr_checkpoint_screen.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
