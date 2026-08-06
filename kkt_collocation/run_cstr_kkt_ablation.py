"""Matched three-seed CSTR supervised-versus-KKT ablation.

This supplementary experiment is deliberately separate from the validation-gated
deployment evaluation.  For every fixed training seed and its disjoint 400-point
test cohort, it evaluates the supervised policy and a KKT-refined continuation
on exactly the same initial states.  The resulting 2 x 1200 nominal HDS audits
support the CSTR column of the manuscript's training-time KKT comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kkt_collocation.run_cstr_full_simulation import (  # noqa: E402
    CSTRConfig, lhs_states, make_corrector, objective, predict, train_branch,
)

TRAIN_SEEDS = (20260718, 20260725, 20260726)
TEST_SEEDS = (20260724, 20260825, 20260826)
FIELDS = ("method", "training_seed", "sample_index", "CA0", "T0", "nominal_hds_max_g", "nominal_objective")


def load_model(path: Path, cfg: CSTRConfig):
    try:
        saved = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        saved = torch.load(path, map_location="cpu")
    from kkt_collocation.run_cstr_full_simulation import Policy
    model = Policy(cfg)
    model.load_state_dict(saved["model"])
    return model, np.asarray(saved["mean"]), np.asarray(saved["std"])


def save_model(path: Path, model, mean: np.ndarray, std: np.ndarray, cfg: CSTRConfig) -> None:
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "config": asdict(cfg)}, path)


def train_or_load(seed: int, data: dict, cfg: CSTRConfig, output: Path):
    supervised_path = output / f"cstr_seed{seed}_supervised.pth"
    kkt_path = output / f"cstr_seed{seed}_kkt_refined.pth"
    if supervised_path.exists():
        supervised, mean_s, std_s = load_model(supervised_path, cfg)
    else:
        supervised, mean_s, std_s, _ = train_branch(data, cfg, kkt=False, epochs=300, seed=seed)
        save_model(supervised_path, supervised, mean_s, std_s, cfg)
    if kkt_path.exists():
        refined, mean_k, std_k = load_model(kkt_path, cfg)
    else:
        refined, mean_k, std_k, _ = train_branch(data, cfg, kkt=True, epochs=30, seed=seed + 500, base=supervised)
        save_model(kkt_path, refined, mean_k, std_k, cfg)
    return (supervised, mean_s, std_s), (refined, mean_k, std_k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "cstr_full_simulation_900" / "cstr_labels.pkl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "cstr_kkt_ablation1200")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with args.labels.open("rb") as handle:
        data = pickle.load(handle)
    cfg = CSTRConfig(**data["config"])
    rows: list[dict] = []
    for seed, test_seed in zip(TRAIN_SEEDS, TEST_SEEDS):
        supervised, refined = train_or_load(seed, data, cfg, args.output)
        states = lhs_states(cfg, 400, test_seed)
        corrector = make_corrector(cfg)
        for method, (model, mean, std) in (("Supervised", supervised), ("KKT-refined", refined)):
            controls, _ = predict(model, mean, std, states)
            for index, (state, control) in enumerate(zip(states, controls)):
                rows.append({"method": method, "training_seed": seed, "sample_index": index,
                             "CA0": state[0], "T0": state[1],
                             "nominal_hds_max_g": corrector.audit(state, control, cfg.zoh_duration),
                             "nominal_objective": objective(state, control, cfg)})
                if (index + 1) % 25 == 0:
                    print(f"seed {seed}, {method}: {index + 1}/400", flush=True)
        print(f"completed CSTR KKT ablation seed {seed}", flush=True)
    with (args.output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    report = {"protocol": "Matched nominal-policy comparison: supervised and KKT-refined CSTR policies evaluated on the same 3 x 400 initial conditions before HDS correction.",
              "training_seeds": list(TRAIN_SEEDS), "test_seeds": list(TEST_SEEDS), "pairs": 1200}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
