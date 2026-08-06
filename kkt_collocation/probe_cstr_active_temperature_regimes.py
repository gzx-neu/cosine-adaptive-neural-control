"""Screen safety-limited operating domains for the existing CSTR benchmark.

This diagnostic does not train a network or change any manuscript artefact.  It
only checks whether a proposed initial-state domain yields feasible offline
solutions whose temperature path constraint is actually relevant.  The same
mass/energy balance, transcription, and event-located HDS audit as
``run_cstr_full_simulation.py`` are used.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kkt_collocation.run_cstr_full_simulation import (
    CSTRConfig,
    CSTRTranscription,
    grid_states,
    make_corrector,
)


def candidate_config(limit: float) -> CSTRConfig:
    """Keep a five-kelvin initial safety buffer while varying the safety limit."""
    return replace(
        CSTRConfig(),
        temperature_max=limit,
        temperature_range=(345.0, limit - 5.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--limits", type=float, nargs="+", default=(360.0, 362.0, 365.0))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "cstr_active_regime_screen",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | bool | str]] = []
    for limit in args.limits:
        cfg = candidate_config(float(limit))
        transcription = CSTRTranscription(cfg)
        corrector = make_corrector(cfg)
        for index, state in enumerate(grid_states(cfg, args.grid_size)):
            try:
                record = transcription.solve(state)
                peak = corrector.audit(state, record["optimal_controls"], cfg.zoh_duration)
                rows.append(
                    {
                        "temperature_limit_K": cfg.temperature_max,
                        "CA0": float(state[0]),
                        "T0_K": float(state[1]),
                        "solved": True,
                        "objective": float(record["objective"]),
                        "hds_peak_g_K": float(peak),
                        "label_hds_corrected": bool(record["label_hds_corrected"]),
                        "mean_cooling": float(np.mean(record["optimal_controls"])),
                        "max_cooling": float(np.max(record["optimal_controls"])),
                        "solve_seconds": float(record["solve_seconds"]),
                        "error": "",
                    }
                )
            except Exception as error:  # diagnostics must retain failed points
                rows.append(
                    {
                        "temperature_limit_K": cfg.temperature_max,
                        "CA0": float(state[0]),
                        "T0_K": float(state[1]),
                        "solved": False,
                        "objective": np.nan,
                        "hds_peak_g_K": np.nan,
                        "label_hds_corrected": False,
                        "mean_cooling": np.nan,
                        "max_cooling": np.nan,
                        "solve_seconds": np.nan,
                        "error": str(error),
                    }
                )
            print(f"Tmax={limit:.1f} K, point {index + 1}/{args.grid_size ** 2}", flush=True)

    fields = list(rows[0])
    with (args.output / "per_point.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict[str, float | int]] = []
    for limit in args.limits:
        selected = [row for row in rows if row["temperature_limit_K"] == float(limit)]
        solved = [row for row in selected if bool(row["solved"])]
        peaks = np.asarray([float(row["hds_peak_g_K"]) for row in solved], dtype=float)
        summary.append(
            {
                "temperature_limit_K": float(limit),
                "initial_temperature_min_K": 345.0,
                "initial_temperature_max_K": float(limit) - 5.0,
                "points": len(selected),
                "solved": len(solved),
                "failed": len(selected) - len(solved),
                "active_within_0p1K": int(np.sum(peaks >= -0.1)) if peaks.size else 0,
                "mean_peak_g_K": float(np.mean(peaks)) if peaks.size else np.nan,
                "max_peak_g_K": float(np.max(peaks)) if peaks.size else np.nan,
                "hds_repaired_labels": int(sum(bool(row["label_hds_corrected"]) for row in solved)),
            }
        )
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.output / "protocol.txt").write_text(
        "Diagnostic only: the CSTR dynamics and solver are unchanged. Each candidate "
        "uses T(0) <= Tmax - 5 K so every tested initial state is safe.\n"
        + str(asdict(candidate_config(float(args.limits[0])))),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
