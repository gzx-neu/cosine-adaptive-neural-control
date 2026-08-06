"""Convert a v7.3 MATLAB Jiang--Fu VDP-CVP20 export without altering its grids."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


FIELDS = (
    "index", "initial_state", "success", "exit_flag", "objective", "controls",
    "solve_seconds", "outer_iterations", "final_constraint_grid", "path_multipliers",
    "lower_bound_multipliers", "upper_bound_multipliers", "final_nlp_iterations",
)


def dereference(file: h5py.File, refs: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(file[reference]).squeeze() for reference in refs.reshape(-1)]


def read_records(path: Path) -> dict[str, list[np.ndarray]]:
    with h5py.File(path, "r") as handle:
        if "records" not in handle:
            raise ValueError(f"{path} has no records struct")
        records = handle["records"]
        missing = set(FIELDS).difference(records)
        if missing:
            raise ValueError(f"Missing fields: {sorted(missing)}")
        return {field: dereference(handle, records[field][...]) for field in FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    records = read_records(args.input)
    n = len(records["index"])
    if n != 400:
        raise ValueError(f"Expected 400 records, got {n}")
    index = np.asarray(records["index"], dtype=int)
    success = np.asarray(records["success"], dtype=bool)
    states = np.vstack(records["initial_state"]).astype(np.float64)
    controls = np.vstack(records["controls"]).astype(np.float64)
    objective = np.asarray(records["objective"], dtype=np.float64)
    solve_seconds = np.asarray(records["solve_seconds"], dtype=np.float64)
    grids = [np.asarray(value, dtype=np.float64).reshape(-1) for value in records["final_constraint_grid"]]
    multipliers = [np.asarray(value, dtype=np.float64).reshape(-1) for value in records["path_multipliers"]]
    if not np.array_equal(index, np.arange(n)) or not success.all() or controls.shape != (400, 20):
        raise ValueError("Expected ordered 400/400-successful CVP20 records")
    if any(len(grid) != len(multiplier) + 1 for grid, multiplier in zip(grids, multipliers)):
        raise ValueError("Each Jiang--Fu path multiplier vector must match its final-grid intervals")
    max_grid = max(map(len, grids))
    max_multiplier = max(map(len, multipliers))
    grid_padded = np.full((n, max_grid), np.nan)
    multiplier_padded = np.full((n, max_multiplier), np.nan)
    for row, (grid, multiplier) in enumerate(zip(grids, multipliers)):
        grid_padded[row, :len(grid)] = grid
        multiplier_padded[row, :len(multiplier)] = multiplier
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "native_grid_labels.npz",
        initial_state=states, optimal_controls=controls, objective=objective,
        solve_seconds=solve_seconds, final_constraint_grid_padded=grid_padded,
        final_constraint_grid_length=np.asarray([len(item) for item in grids], dtype=int),
        path_multipliers_padded=multiplier_padded,
        path_multiplier_length=np.asarray([len(item) for item in multipliers], dtype=int),
        lower_bound_multipliers=np.vstack(records["lower_bound_multipliers"]),
        upper_bound_multipliers=np.vstack(records["upper_bound_multipliers"]),
    )
    report = {
        "source": str(args.input), "records": n, "successes": int(success.sum()),
        "control_dimension": int(controls.shape[1]), "objective_direction": "minimize terminal x3",
        "mean_cold_solve_seconds": float(solve_seconds.mean()),
        "final_constraint_grid_length": {"min": int(min(map(len, grids))), "max": int(max(map(len, grids)))},
        "path_multiplier_length": {"min": int(min(map(len, multipliers))), "max": int(max(map(len, multipliers)))},
        "native_grid_note": "Each multiplier remains paired with its own Jiang--Fu final adaptive upper-bound interval grid; it must not be treated as a fixed-node RK multiplier.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
