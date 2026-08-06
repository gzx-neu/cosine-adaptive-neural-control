"""Evaluate the exploratory N=40 cosine-projection CSTR checkpoints.

The checkpoints are compared only with the frozen, matched N=40 cold-start
reference solutions used by the earlier N=40 study.  For every identical
initial state the reported signed gap is
    100 * (J_network+HDS - J_cold_NLP) / abs(J_cold_NLP).
This is an exploratory, matched-discretization diagnostic; it is not the
planned N=100/all-400 primary evaluation.
"""
from __future__ import annotations

import json
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from run_economou_cstr_supervised_hds import PolicyValue
from run_economou_cstr_two_stage_vs_kkt_only import Config, ROOT, dump_json, evaluate_method
from screen_economou_cstr_30x30 import EconomouScreenConfig


TRAINING = ROOT / "kkt_collocation/results/exploratory_cosine_adaptive_projection_cstr_n40_v2/cstr/seed20260771"
BASELINE = ROOT / "kkt_collocation/results/economou_cstr_two_stage_vs_kkt_only_n40_rk10_ca030_050_t410_420_s_sk_single"
LABELS = ROOT / "kkt_collocation/results/economou_cstr_reduced_kkt_n40_rk10_ca030_050_t410_420"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=TRAINING)
    parser.add_argument("--methods", nargs="+", default=("S-u", "S-u+K"))
    args = parser.parse_args()
    training = args.training.resolve()
    try:
        training_seed = int(training.name.removeprefix("seed"))
    except ValueError:
        training_seed = None
    label_summary = json.loads((LABELS / "summary.json").read_text(encoding="utf-8"))
    values = dict(label_summary["config"])
    for key in ("ti_bounds_K", "flow_bounds", "ca_initial_range", "temperature_initial_range_K"):
        values[key] = tuple(values[key])
    cstr = EconomouScreenConfig(**values)
    test = np.load(BASELINE / "test_initial_conditions.npy")
    references = json.loads((BASELINE / "cold_start_references.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(lambda_grid_size=31, hds_workers=4)

    output: dict[str, object] = {
        "status": "running",
        "protocol": {
            "training": f"N=40 KKT-conflict projection training, seed {training_seed}",
            "test_reference": "frozen matched N=40 cold-start NLP reference from the prior single-seed study",
            "gap_definition": "100*(J_network_plus_HDS-J_cold_NLP)/abs(J_cold_NLP)",
            "test_count": int(len(test)),
            "hds_statement": "continuous-time numerical audit evidence under declared model/numerics; not a real-system absolute safety guarantee",
        },
        "methods": {},
    }
    for method in args.methods:
        checkpoint = torch.load(training / f"{method}.pth", map_location=device, weights_only=False)
        model = PolicyValue(cstr).to(device)
        model.load_state_dict(checkpoint["model"])
        result = evaluate_method(
            method,
            model,
            np.asarray(checkpoint["state_mean"]),
            np.asarray(checkpoint["state_std"]),
            test,
            cstr,
            cfg,
            training,
            references,
        )
        output["methods"][method] = result
        dump_json(training / "evaluation_summary.json", output)
    output["status"] = "completed"
    dump_json(training / "evaluation_summary.json", output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
