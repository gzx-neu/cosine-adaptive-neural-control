"""Serially retime conservative HDS correction for every frozen seed."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kkt_collocation.update_matched_comparison_conservative import corrector


SEEDS = {
    "VDP": (20260751, 20260752, 20260753),
    "Penicillin": (20260761, 20260762, 20260763),
    "CSTR": (20260718, 20260725, 20260726),
}


def stats(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=tuple(SEEDS), required=True)
    parser.add_argument("--margin", type=float, default=1e-6)
    parser.add_argument("--root", type=Path, default=ROOT / "kkt_collocation" / "results" / "conservative_margin_1e6")
    args = parser.parse_args()
    per_seed = {}
    seed_means = []
    for seed in SEEDS[args.problem]:
        directory = args.root / f"{args.problem.lower()}_seed{seed}"
        data = np.load(directory / "population_controls.npz")
        states, nominal, expected = data["initial_states"], data["nominal_controls"], data["applied_controls"]
        evaluator, duration = corrector(args.problem, args.margin)
        timings = []
        for index, (state, control) in enumerate(zip(states, nominal)):
            started = time.perf_counter(); outcome = evaluator.correct(state, control, duration); timings.append(time.perf_counter() - started)
            if not outcome.accepted or not np.allclose(outcome.controls, expected[index], rtol=0.0, atol=1e-12):
                raise RuntimeError(f"{args.problem} seed {seed} mismatch at {index}")
        values = np.asarray(timings)
        per_seed[str(seed)] = stats(values)
        seed_means.append(values.mean())
        print(f"{args.problem} seed {seed}: {values.mean():.6f} +/- {values.std(ddof=1):.6f} s", flush=True)
    report = {"problem": args.problem, "safety_margin": args.margin, "per_seed": per_seed,
              "aggregate_across_seed_means": stats(np.asarray(seed_means))}
    (args.root / f"{args.problem.lower()}_retiming_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
