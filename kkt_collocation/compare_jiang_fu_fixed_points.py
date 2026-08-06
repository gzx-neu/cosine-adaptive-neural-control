"""Pointwise comparison of Adaptive+HDS against Jiang--Fu Algorithm 1."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from kkt_collocation.run_penicillin_ablation import DT, UMAX, Policy
from kkt_collocation.run_penicillin_ablation import g as pen_g
from kkt_collocation.run_penicillin_ablation import gdot as pen_gdot
from kkt_collocation.run_penicillin_ablation import ode as pen_ode
from kkt_collocation.run_penicillin_ablation import predict as pen_predict
from kkt_collocation.run_penicillin_ablation import terminal_product
from kkt_collocation.run_vdp_ablation import constraint as vdp_g
from kkt_collocation.run_vdp_ablation import constraint_derivative as vdp_gdot
from kkt_collocation.run_vdp_ablation import predict as vdp_predict
from kkt_collocation.run_vdp_ablation import terminal_cost, vdp_ode
from kkt_collocation.train_vdp_kkt_policy import KKTPolicyValueNetwork, TrainConfig


def load_jiang_rows(path: Path, scenario: str) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {}
    for row in rows:
        if row["scenario"] != scenario:
            continue
        if row["solver_success"].lower() not in {"1", "true"} or row["hds_safe"].lower() not in {"1", "true"}:
            raise RuntimeError(f"Jiang--Fu reference is not certified at {row['problem']} {row['point_id']}")
        selected[(row["problem"], row["point_id"])] = row
    return selected


def corrected_result(corrector, initial: np.ndarray, controls: np.ndarray, duration: float, problem: str):
    started = time.perf_counter()
    correction = corrector.correct(initial, controls, duration)
    filter_seconds = time.perf_counter() - started
    if not correction.accepted:
        return {
            "accepted": False, "fallback": True, "filter_seconds": filter_seconds,
            "hds_gmax": float("nan"), "objective": float("nan"), "corrected_segments": float("nan"),
        }
    applied = np.asarray(correction.controls, dtype=float)
    hds_gmax = corrector.audit(initial, applied, duration)
    objective = (terminal_cost(initial, applied, corrector, duration) if problem == "VDP"
                 else -terminal_product(float(initial[1]), applied, corrector))
    return {
        "accepted": True, "fallback": False, "filter_seconds": filter_seconds,
        "hds_gmax": hds_gmax, "objective": objective,
        "corrected_segments": sum(segment.corrected for segment in correction.segments),
    }


def evaluate_vdp(directory: Path, references, device: torch.device) -> list[dict[str, object]]:
    checkpoint = torch.load(directory / "models.pth", map_location=device, weights_only=False)
    seed = int(checkpoint["config"]["seed"])
    model = KKTPolicyValueNetwork(TrainConfig()).to(device)
    model.load_state_dict(checkpoint["S"])
    model.eval()
    items = sorted((key, row) for key, row in references.items() if key[0] == "VDP")
    states = np.asarray([[float(row["x1_0"]), float(row["x2_0"]), 0.0] for _, row in items])
    mean, std = checkpoint["normalization"]["S"]
    controls, inference_seconds = vdp_predict(model, np.asarray(mean), np.asarray(std), states, device)
    corrector = HDSLambdaCorrector(
        vdp_ode, vdp_g, vdp_gdot, (-0.3, 1.0),
        HDSLambdaConfig(grid_size=31, max_step_fraction=100.0),
    )
    rows = []
    for ((_, point_id), reference), initial, nominal in zip(items, states, controls):
        result = corrected_result(corrector, initial, nominal, 0.5, "VDP")
        jiang_objective = float(reference["independent_objective"])
        result.update({
            "problem": "VDP", "point_id": point_id, "seed": seed, "adaptive_model": "S",
            "inference_seconds": inference_seconds,
            "total_deployment_seconds": inference_seconds + float(result["filter_seconds"]),
            "jiang_solve_seconds": float(reference["solve_seconds"]),
            "jiang_objective": jiang_objective, "jiang_hds_gmax": float(reference["hds_gmax"]),
            "relative_objective_difference": abs(float(result["objective"]) - jiang_objective) / abs(jiang_objective),
        })
        rows.append(result)
    return rows


def evaluate_pen(directory: Path, references, device: torch.device) -> list[dict[str, object]]:
    checkpoint = torch.load(directory / "models.pth", map_location=device, weights_only=False)
    seed = int(checkpoint["config"]["seed"])
    model = Policy().to(device)
    model.load_state_dict(checkpoint["true_KKT"])
    model.eval()
    items = sorted((key, row) for key, row in references.items() if key[0] == "Penicillin")
    x2 = np.asarray([float(row["x2_0"]) for _, row in items])
    states = np.column_stack((np.ones(len(x2)), x2, np.full(len(x2), 0.001), np.full(len(x2), 250.0)))
    mean, std = checkpoint["normalization"]["true_KKT"]
    controls, inference_seconds = pen_predict(model, float(mean), float(std), x2, device)
    corrector = HDSLambdaCorrector(
        pen_ode, pen_g, pen_gdot, (0.0, UMAX),
        HDSLambdaConfig(grid_size=31, max_step_fraction=100.0),
    )
    rows = []
    for ((_, point_id), reference), initial, nominal in zip(items, states, controls):
        result = corrected_result(corrector, initial, nominal, DT, "Penicillin")
        jiang_objective = float(reference["independent_objective"])
        result.update({
            "problem": "Penicillin", "point_id": point_id, "seed": seed,
            "adaptive_model": "S+true-KKT", "inference_seconds": inference_seconds,
            "total_deployment_seconds": inference_seconds + float(result["filter_seconds"]),
            "jiang_solve_seconds": float(reference["solve_seconds"]),
            "jiang_objective": jiang_objective, "jiang_hds_gmax": float(reference["hds_gmax"]),
            "relative_objective_difference": abs(float(result["objective"]) - jiang_objective) / abs(jiang_objective),
        })
        rows.append(result)
    return rows


def seed_then_problem_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    per_problem: dict[str, list[dict[str, object]]] = defaultdict(list)
    for problem in sorted({str(row["problem"]) for row in rows}):
        for seed in sorted({int(row["seed"]) for row in rows if row["problem"] == problem}):
            group = [row for row in rows if row["problem"] == problem and int(row["seed"]) == seed]
            per_problem[problem].append({
                "seed": seed,
                "points": len(group),
                "accepted_rate": float(np.mean([bool(row["accepted"]) for row in group])),
                "fallback_rate": float(np.mean([bool(row["fallback"]) for row in group])),
                "adaptive_total_seconds": float(np.mean([float(row["total_deployment_seconds"]) for row in group])),
                "adaptive_filter_seconds": float(np.mean([float(row["filter_seconds"]) for row in group])),
                "adaptive_max_hds_g": float(max(float(row["hds_gmax"]) for row in group)),
                "adaptive_mean_objective": float(np.mean([float(row["objective"]) for row in group])),
                "relative_objective_difference": float(np.mean([float(row["relative_objective_difference"]) for row in group])),
                "corrected_segments": float(np.mean([float(row["corrected_segments"]) for row in group])),
                "jiang_solve_seconds": float(np.mean([float(row["jiang_solve_seconds"]) for row in group])),
                "jiang_max_hds_g": float(max(float(row["jiang_hds_gmax"]) for row in group)),
                "jiang_mean_objective": float(np.mean([float(row["jiang_objective"]) for row in group])),
            })

    output = {}
    for problem, seeds in per_problem.items():
        aggregate = {"training_seeds": [item["seed"] for item in seeds], "points_per_seed": seeds[0]["points"]}
        for key in seeds[0]:
            if key in {"seed", "points"}:
                continue
            values = np.asarray([float(item[key]) for item in seeds])
            aggregate[key] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "per_seed": values.tolist(),
            }
        output[problem] = aggregate
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jiang-results", type=Path,
        default=ROOT / "kkt_collocation" / "results" / "jiang_fu_algorithm1_baseline" / "per_point.csv",
    )
    parser.add_argument(
        "--scenario", default="unified_fixed_points",
        help="Scenario label from the audited Jiang--Fu CSV to evaluate on identical points.",
    )
    parser.add_argument(
        "--vdp-inputs", nargs="+", type=Path,
        default=[ROOT / "kkt_collocation" / "results" / f"final_multiseed_vdp900_penalty_seed{seed}"
                 for seed in (20260751, 20260752, 20260753)],
    )
    parser.add_argument(
        "--penicillin-inputs", nargs="+", type=Path,
        default=[ROOT / "kkt_collocation" / "results" / f"final_multiseed_penicillin400_penalty_seed{seed}"
                 for seed in (20260761, 20260762, 20260763)],
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "kkt_collocation" / "results" / "jiang_fu_fixed_point_comparison",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    references = load_jiang_rows(args.jiang_results, args.scenario)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict[str, object]] = []
    if any(problem == "VDP" for problem, _ in references):
        for directory in args.vdp_inputs:
            rows.extend(evaluate_vdp(directory, references, device))
    if any(problem == "Penicillin" for problem, _ in references):
        for directory in args.penicillin_inputs:
            rows.extend(evaluate_pen(directory, references, device))
    if not rows:
        raise RuntimeError(f"No certified references found for scenario {args.scenario!r}")

    with (args.output / "per_point_seed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "comparison": "Adaptive+HDS versus Jiang--Fu Algorithm 1 on identical points and problem settings",
        "scenario": args.scenario,
        "timing_note": "Jiang--Fu is a cold MATLAB solve; Adaptive time is inference plus required HDS correction/audit",
        "objective_note": "relative difference uses the independently integrated Jiang--Fu objective at the same point",
        "problems": seed_then_problem_summary(rows),
    }
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
