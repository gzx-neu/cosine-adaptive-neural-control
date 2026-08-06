"""Evaluate one saved no-rollout VDP checkpoint on its frozen 400-point cohort."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from kkt_collocation.run_vdp_ablation import AblationConfig, KKTPolicyValueNetwork, TrainConfig, constraint, constraint_derivative, predict, terminal_cost, vdp_ode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    out = ROOT / "kkt_collocation" / "results" / "vdp_no_rollout_w0"
    checkpoint = torch.load(out / f"vdp_seed{args.seed}.pth", map_location="cpu", weights_only=False)
    cfg = AblationConfig(**checkpoint["config"])
    model = KKTPolicyValueNetwork(TrainConfig()); model.load_state_dict(checkpoint["model"]); model.eval()
    source = ROOT / "kkt_collocation" / "results" / f"final_multiseed_vdp900_penalty_seed{args.seed}"
    states = np.load(source / "test_states.npy")
    controls, inference = predict(model, np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"]), states, torch.device("cpu"))
    with (ROOT / "kkt_collocation" / "results" / "jiang_fu_matched400_comparison" / "per_point_seed.csv").open(newline="", encoding="utf-8") as handle:
        refs = [r for r in csv.DictReader(handle) if r["problem"] == "VDP" and int(r["seed"]) == 20260751]
    ref_objective = np.asarray([float(r["jiang_objective"]) for r in refs])
    corrector = HDSLambdaCorrector(vdp_ode, constraint, constraint_derivative, (-.3, 1.), HDSLambdaConfig(grid_size=31, max_step_fraction=100.0))
    rows = []
    for i, (state, nominal) in enumerate(zip(states, controls)):
        raw_peak = corrector.audit(state, nominal, cfg.duration)
        raw_cost = terminal_cost(state, nominal, corrector, cfg.duration)
        start = time.perf_counter(); outcome = corrector.correct(state, nominal, cfg.duration); elapsed = time.perf_counter()-start
        if not outcome.accepted: raise RuntimeError(f"HDS dispatch at {i}")
        applied_peak = corrector.audit(state, outcome.controls, cfg.duration)
        applied_cost = terminal_cost(state, outcome.controls, corrector, cfg.duration)
        rows.append({"training_seed":args.seed,"sample_index":i,"nominal_hds_max_g":raw_peak,"applied_hds_max_g":applied_peak,
                     "nominal_cost":raw_cost,"applied_cost":applied_cost,"reference_objective":ref_objective[i],
                     "relative_objective_difference_percent":abs(applied_cost-ref_objective[i])/max(abs(ref_objective[i]),1e-12)*100,
                     "corrected_segments":int(sum(s.corrected for s in outcome.segments)),"filter_seconds":elapsed,"inference_seconds":inference})
        if (i+1)%100==0: print(f"seed {args.seed}: {i+1}/400", flush=True)
    path = out / f"per_sample_seed{args.seed}.csv"
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    rel=np.asarray([r["relative_objective_difference_percent"] for r in rows])
    summary={"seed":args.seed,"selected_branch":checkpoint["selected_branch"],"nominal_violation_rate_percent":float(100*np.mean([r["nominal_hds_max_g"]>1e-8 for r in rows])),
             "accepted_peak_max":float(max(r["applied_hds_max_g"] for r in rows)),"mean_corrected_segments":float(np.mean([r["corrected_segments"] for r in rows])),
             "relative_objective_difference_percent":{"mean":float(rel.mean()),"sample_std":float(rel.std(ddof=1))}}
    (out/f"summary_seed{args.seed}.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == "__main__": main()
