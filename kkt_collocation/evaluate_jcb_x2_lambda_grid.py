"""Re-evaluate one frozen JCB x2 policy under a different lambda-grid size."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from run_jcb2d_jiang_valc import Config, Policy, g, gdot, initial, ode, objective_np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambda-grid", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg0 = checkpoint["config"]
    cfg = Config(**cfg0)
    model = Policy(cfg.zoh_steps)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mean, std = np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"])

    source_rows = list(csv.DictReader(args.test_csv.open(encoding="utf-8")))
    points = np.asarray([[float(row["x1_0"]), float(row["x2_0"])] for row in source_rows])
    with torch.no_grad():
        _, ut = model(torch.tensor((points - mean) / std, dtype=torch.float32))
    controls = ut.numpy()
    reference = loadmat(args.reference)
    p_ref = np.asarray(reference["initialStates"], dtype=float)
    j_ref = np.asarray(reference["objectives"], dtype=float).reshape(-1)
    if not np.allclose(points, p_ref):
        raise ValueError("The frozen test points do not match the cold-start reference points.")

    corrector = HDSLambdaCorrector(
        ode, g, gdot, (cfg.u_min, cfg.u_max),
        HDSLambdaConfig(grid_size=args.lambda_grid, safety_margin=cfg.margin,
                        max_step_fraction=200.0),
    )
    rows = []
    for i, (p, u) in enumerate(zip(points, controls)):
        raw = corrector.audit(initial(p), u, cfg.dt)
        outcome = corrector.correct(initial(p), u, cfg.dt)
        if not outcome.accepted:
            raise RuntimeError(f"Frozen policy fell back at test index {i}.")
        applied = outcome.controls
        j_nom, j_app = objective_np(p, u, cfg), objective_np(p, applied, cfg)
        rows.append({
            "index": i, "x1_0": p[0], "x2_0": p[1], "raw_hds_max_g": raw,
            "applied_hds_max_g": corrector.audit(initial(p), applied, cfg.dt),
            "nominal_objective": j_nom, "applied_objective": j_app,
            "reference_objective": j_ref[i],
            "relative_objective_difference_percent": 100 * abs(j_app - j_ref[i]) / max(abs(j_ref[i]), 1e-8),
            "corrected_segments": int(sum(segment.corrected for segment in outcome.segments)),
        })
    with (args.output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    rel = np.asarray([row["relative_objective_difference_percent"] for row in rows])
    delta = np.asarray([row["applied_objective"] - row["nominal_objective"] for row in rows])
    report = {
        "frozen_model": str(args.model), "lambda_grid": args.lambda_grid, "points": len(rows),
        "hds_acceptance_rate_percent": 100.0,
        "fallback_rate_percent": 0.0,
        "raw_violation_rate_percent": 100 * float(np.mean([row["raw_hds_max_g"] > 1e-8 for row in rows])),
        "raw_max_g": float(max(row["raw_hds_max_g"] for row in rows)),
        "accepted_max_g": float(max(row["applied_hds_max_g"] for row in rows)),
        "mean_corrected_segments": float(np.mean([row["corrected_segments"] for row in rows])),
        "relative_objective_difference_percent": {"mean": float(rel.mean()), "sample_std": float(rel.std(ddof=1)), "maximum": float(rel.max())},
        "hds_objective_change": {"mean": float(delta.mean()), "sample_std": float(delta.std(ddof=1)), "maximum": float(delta.max())},
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
