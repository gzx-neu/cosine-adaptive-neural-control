"""Compare discrete trajectory checks with event-located HDS auditing.

For each fixed test trajectory, the script independently evaluates ZOH
endpoints, 10 uniform samples per ZOH, 100 uniform samples per ZOH, and the
event-located HDS maximum.  It reports false-safe decisions relative to HDS,
peak underestimation, and audit time over all three training seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from kkt_collocation.run_vdp_ablation import constraint as vdp_g, constraint_derivative as vdp_gdot, predict as vdp_predict, vdp_ode
from kkt_collocation.train_vdp_kkt_policy import KKTPolicyValueNetwork, TrainConfig
from kkt_collocation.run_penicillin_ablation import DT, UMAX, Policy, g as pen_g, gdot as pen_gdot, ode as pen_ode, predict as pen_predict

_PROBLEM = ""
_CORRECTOR = None


def _init_worker(problem: str) -> None:
    global _PROBLEM, _CORRECTOR
    _PROBLEM = problem
    if problem == "vdp":
        _CORRECTOR = HDSLambdaCorrector(vdp_ode, vdp_g, vdp_gdot, (-.3, 1.), HDSLambdaConfig(max_step_fraction=100.))
    else:
        _CORRECTOR = HDSLambdaCorrector(pen_ode, pen_g, pen_gdot, (0., UMAX), HDSLambdaConfig(max_step_fraction=100.))


def _sampled_peak(initial: np.ndarray, controls: np.ndarray, samples_per_segment: int) -> tuple[float, float]:
    ode = vdp_ode if _PROBLEM == "vdp" else pen_ode
    constraint = vdp_g if _PROBLEM == "vdp" else pen_g
    duration = .5 if _PROBLEM == "vdp" else DT
    state = np.asarray(initial, dtype=float)
    peak = constraint(state)
    start = time.perf_counter()
    for control in controls:
        times = np.linspace(0., duration, samples_per_segment+1)
        sol = solve_ivp(lambda t, z: ode(t, z, float(control)), (0., duration), state,
                        t_eval=times, rtol=1e-9, atol=1e-11, max_step=duration)
        if not sol.success:
            raise RuntimeError(sol.message)
        peak = max(peak, max(constraint(sol.y[:, j]) for j in range(sol.y.shape[1])))
        state = sol.y[:, -1]
    return float(peak), time.perf_counter()-start


def _one(task):
    if _CORRECTOR is None:
        raise RuntimeError("worker not initialized")
    seed, index, initial, controls = task
    row = {"training_seed": seed, "sample_index": index}
    for name, samples in (("zoh_endpoints", 1), ("uniform_10", 10), ("uniform_100", 100)):
        row[f"{name}_peak"], row[f"{name}_seconds"] = _sampled_peak(initial, controls, samples)
    start = time.perf_counter()
    duration = .5 if _PROBLEM == "vdp" else DT
    row["hds_peak"] = float(_CORRECTOR.audit(np.asarray(initial), np.asarray(controls), duration))
    row["hds_seconds"] = time.perf_counter()-start
    return row


def _load_controls(problem: str, directory: Path, device) -> tuple[int, np.ndarray, np.ndarray]:
    report = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    seed = int(report["config"]["seed"])
    checkpoint = torch.load(directory/"models.pth", map_location=device, weights_only=False)
    if problem == "vdp":
        branch = "KKT" if report["adaptive_gate"]["selected_branch"] != "S" else "S"
        model = KKTPolicyValueNetwork(TrainConfig()).to(device)
        model.load_state_dict(checkpoint[branch]); model.eval()
        mean, std = checkpoint["normalization"][branch]
        states = np.load(directory/"test_states.npy")
        controls, _ = vdp_predict(model, np.asarray(mean), np.asarray(std), states, device)
        return seed, states, controls
    branch = "true_KKT" if report["adaptive_gate"]["selected_branch"] != "S" else "S"
    model = Policy().to(device); model.load_state_dict(checkpoint[branch]); model.eval()
    mean, std = checkpoint["normalization"][branch]
    x2 = np.load(directory/"test_x2.npy")
    controls, _ = pen_predict(model, float(mean), float(std), x2, device)
    states = np.column_stack((np.ones(len(x2)), x2, np.full(len(x2), .001), np.full(len(x2), 250.)))
    return seed, states, controls


def _aggregate(rows: list[dict]) -> dict:
    report = {"samples": len(rows), "hds_violation_rate": float(np.mean([r["hds_peak"] > 1e-8 for r in rows])), "audits": {}}
    hds = np.asarray([r["hds_peak"] for r in rows])
    for name in ("zoh_endpoints", "uniform_10", "uniform_100"):
        sampled = np.asarray([r[f"{name}_peak"] for r in rows])
        false_safe = (sampled <= 1e-8) & (hds > 1e-8)
        report["audits"][name] = {
            "declared_safe_rate": float(np.mean(sampled <= 1e-8)),
            "false_safe_rate_vs_hds": float(false_safe.mean()),
            "false_safe_count": int(false_safe.sum()),
            "maximum_peak_underestimation": float(np.max(hds-sampled)),
            "mean_peak_underestimation": float(np.mean(hds-sampled)),
            "mean_seconds": float(np.mean([r[f"{name}_seconds"] for r in rows])),
        }
    report["audits"]["HDS"] = {"false_safe_rate_vs_hds": 0., "false_safe_count": 0,
                                        "mean_seconds": float(np.mean([r["hds_seconds"] for r in rows]))}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("vdp", "penicillin"), required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = []
    for directory in args.inputs:
        seed, states, controls = _load_controls(args.problem, directory, device)
        tasks.extend((seed, index, state, control) for index, (state, control) in enumerate(zip(states, controls)))
    if args.workers == 1:
        _init_worker(args.problem); rows = [_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(args.problem,)) as pool:
            rows = list(pool.map(_one, tasks, chunksize=4))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/"per_trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = {"problem": args.problem, "training_seeds": sorted(set(r["training_seed"] for r in rows)),
              "comparison": _aggregate(rows),
              "note": "All sampling audits use high-accuracy integration; HDS additionally locates stationary constraint events rather than relying on a fixed grid."}
    (args.output/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
