"""Evaluate an instance-wise, pre-execution S-versus-KKT HDS selector.

For every measured initial condition, the supervised and KKT-refined policy
each generate one complete ZOH sequence.  Both sequences are independently
audited and, if necessary, corrected by the same event-located HDS procedure.
Among numerically accepted candidates, the selector deploys the corrected
sequence with the better objective under the declared model.  The selector
does not consult an NLP objective at deployment; frozen cold-start NLP values
are read only for the final evaluation report.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from run_penicillin_ablation import (DT as PEN_DT, Policy as PenicillinPolicy, UMAX as PEN_UMAX,
                                     g as pen_g, gdot as pen_gdot, ode as pen_ode,
                                     predict as pen_predict, terminal_product)
from run_vdp_ablation import (constraint as vdp_g, constraint_derivative as vdp_gdot,
                              terminal_cost, vdp_ode, predict as vdp_predict)
from train_vdp_kkt_policy import KKTPolicyValueNetwork, TrainConfig


_CORRECTOR: HDSLambdaCorrector | None = None
_PROBLEM: str | None = None


def _worker_init(problem: str) -> None:
    global _CORRECTOR, _PROBLEM
    _PROBLEM = problem
    if problem == "vdp":
        _CORRECTOR = HDSLambdaCorrector(vdp_ode, vdp_g, vdp_gdot, (-0.3, 1.0),
                                        HDSLambdaConfig(grid_size=31, max_step_fraction=100.0))
    else:
        _CORRECTOR = HDSLambdaCorrector(pen_ode, pen_g, pen_gdot, (0.0, PEN_UMAX),
                                        HDSLambdaConfig(grid_size=31, max_step_fraction=100.0))


def _evaluate_one(task: tuple[int, np.ndarray, np.ndarray, np.ndarray, float]) -> dict[str, Any]:
    if _CORRECTOR is None or _PROBLEM is None:
        raise RuntimeError("selector worker was not initialized")
    index, state, u_s, u_k, reference = task
    duration = 0.5 if _PROBLEM == "vdp" else PEN_DT

    def evaluate(label: str, controls: np.ndarray) -> dict[str, Any]:
        raw_peak = _CORRECTOR.audit(state, controls, duration)
        start = time.perf_counter()
        result = _CORRECTOR.correct(state, controls, duration)
        correction_seconds = time.perf_counter() - start
        if not result.accepted:
            return {"label": label, "accepted": False, "raw_peak": raw_peak,
                    "objective": np.nan, "applied_peak": np.nan,
                    "corrected_segments": int(sum(s.corrected for s in result.segments)),
                    "correction_seconds": correction_seconds}
        applied = result.controls
        if _PROBLEM == "vdp":
            objective = terminal_cost(state, applied, _CORRECTOR, duration)
        else:
            objective = -terminal_product(float(state[1]), applied, _CORRECTOR)
        return {"label": label, "accepted": True, "raw_peak": raw_peak,
                "objective": objective, "applied_peak": _CORRECTOR.audit(state, applied, duration),
                "corrected_segments": int(sum(s.corrected for s in result.segments)),
                "correction_seconds": correction_seconds}

    s, k = evaluate("S", u_s), evaluate("KKT", u_k)
    candidates = [candidate for candidate in (s, k) if candidate["accepted"]]
    if not candidates:
        return {"sample_index": index, "selected_branch": "fallback", "accepted": False,
                "reference_objective": reference, "relative_objective_difference_percent": np.nan,
                "selector_seconds": s["correction_seconds"] + k["correction_seconds"],
                **{f"{prefix}_{key}": value for prefix, item in (("s", s), ("kkt", k)) for key, value in item.items()}}
    chosen = min(candidates, key=lambda candidate: candidate["objective"])
    rel = abs(chosen["objective"] - reference) / max(abs(reference), 1e-12) * 100.0
    return {"sample_index": index, "selected_branch": chosen["label"], "accepted": True,
            "reference_objective": reference, "selected_objective": chosen["objective"],
            "selected_hds_max_g": chosen["applied_peak"],
            "selected_corrected_segments": chosen["corrected_segments"],
            "relative_objective_difference_percent": rel,
            "selector_seconds": s["correction_seconds"] + k["correction_seconds"],
            **{f"{prefix}_{key}": value for prefix, item in (("s", s), ("kkt", k)) for key, value in item.items()}}


def _references(problem: str) -> np.ndarray:
    path = ROOT / "kkt_collocation" / "results" / "jiang_fu_matched400_comparison" / "per_point_seed.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["problem"].lower() == problem.lower()]
    # The comparison export stores exactly one shared frozen 400-point sequence
    # for each problem; all policy seeds use those same initial conditions.
    if len(rows) != 400:
        raise RuntimeError(f"Expected 400 {problem} NLP references, found {len(rows)}")
    return np.asarray([float(row["jiang_objective"]) for row in rows], dtype=float)


def _evaluate_tasks(problem: str, states: np.ndarray, us: np.ndarray, uk: np.ndarray,
                    references: np.ndarray, workers: int) -> list[dict[str, Any]]:
    tasks = [(i, np.asarray(state, dtype=float), np.asarray(s, dtype=float), np.asarray(k, dtype=float), float(ref))
             for i, (state, s, k, ref) in enumerate(zip(states, us, uk, references))]
    if workers == 1:
        _worker_init(problem)
        return [_evaluate_one(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(problem,)) as pool:
        return list(pool.map(_evaluate_one, tasks, chunksize=1))


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = np.asarray([row["selected_branch"] for row in rows])
    accepted = np.asarray([row["accepted"] for row in rows], dtype=bool)
    return {
        "samples": len(rows), "accepted_rate": float(accepted.mean()),
        "fallback_rate": float(1.0 - accepted.mean()),
        "selected_S_rate": float(np.mean(selected == "S")),
        "selected_KKT_rate": float(np.mean(selected == "KKT")),
        "mean_relative_objective_difference_percent": float(np.nanmean([row["relative_objective_difference_percent"] for row in rows])),
        "max_selected_hds_g": float(np.nanmax([row.get("selected_hds_max_g", np.nan) for row in rows])),
        "mean_selected_corrected_segments": float(np.nanmean([row.get("selected_corrected_segments", np.nan) for row in rows])),
        "mean_selector_seconds": float(np.mean([row["selector_seconds"] for row in rows])),
        "mean_S_correction_seconds": float(np.mean([row["s_correction_seconds"] for row in rows])),
        "mean_KKT_correction_seconds": float(np.mean([row["kkt_correction_seconds"] for row in rows])),
    }


def _vdp(seed: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = ROOT / "kkt_collocation" / "results" / "validation_selected_vdp_penicillin" / "vdp" / f"seed{seed}"
    checkpoint = torch.load(root / "candidates.pth", map_location=device, weights_only=False)
    s, k = KKTPolicyValueNetwork(TrainConfig()).to(device).eval(), KKTPolicyValueNetwork(TrainConfig()).to(device).eval()
    s.load_state_dict(checkpoint["S"]); k.load_state_dict(checkpoint["KKT"])
    source = ROOT / "kkt_collocation" / "results" / f"final_multiseed_vdp900_penalty_seed{seed}"
    states = np.load(source / "test_states.npy")
    us, _ = vdp_predict(s, np.asarray(checkpoint["normalization"]["S"][0]), np.asarray(checkpoint["normalization"]["S"][1]), states, device)
    uk, _ = vdp_predict(k, np.asarray(checkpoint["normalization"]["KKT"][0]), np.asarray(checkpoint["normalization"]["KKT"][1]), states, device)
    return states, us, uk


def _penicillin(seed: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = ROOT / "kkt_collocation" / "results" / f"final_multiseed_penicillin400_penalty_seed{seed}"
    checkpoint = torch.load(source / "models.pth", map_location=device, weights_only=False)
    s, k = PenicillinPolicy().to(device).eval(), PenicillinPolicy().to(device).eval()
    s.load_state_dict(checkpoint["S"]); k.load_state_dict(checkpoint["true_KKT"])
    x2 = np.load(source / "test_x2.npy")
    us, _ = pen_predict(s, float(checkpoint["normalization"]["S"][0]), float(checkpoint["normalization"]["S"][1]), x2, device)
    uk, _ = pen_predict(k, float(checkpoint["normalization"]["true_KKT"][0]), float(checkpoint["normalization"]["true_KKT"][1]), x2, device)
    states = np.column_stack((np.ones(len(x2)), x2, np.full(len(x2), .001), np.full(len(x2), 250.0)))
    return states, us, uk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("vdp", "penicillin", "both"), default="both")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "dual_candidate_hds_selector")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_report: dict[str, Any] = {"rule": "audit and correct S and KKT; deploy accepted candidate with smaller model objective", "vdp": [], "penicillin": []}
    for problem, seeds, loader in (("vdp", (20260751, 20260752, 20260753), _vdp),
                                  ("penicillin", (20260761, 20260762, 20260763), _penicillin)):
        if args.problem not in (problem, "both"):
            continue
        refs = _references("VDP" if problem == "vdp" else "Penicillin")
        for seed in seeds:
            print(f"[{problem}] dual-candidate selector, seed {seed}", flush=True)
            states, us, uk = loader(seed, device)
            rows = _evaluate_tasks(problem, states, us, uk, refs, args.workers)
            out = args.output / problem; out.mkdir(parents=True, exist_ok=True)
            path = out / f"seed{seed}.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            all_report[problem].append({"seed": seed, **_summarise(rows)})
    (args.output / "summary.json").write_text(json.dumps(all_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(all_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
