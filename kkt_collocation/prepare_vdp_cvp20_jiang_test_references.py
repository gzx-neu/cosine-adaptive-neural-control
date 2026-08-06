"""Merge existing VDP Jiang--Fu CVP20 cold-test shards without re-solving."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kkt_collocation.convert_vdp_cvp20_jiang_mat import read_records  # noqa: E402


def scalar(value: np.ndarray, dtype=float):
    return dtype(np.asarray(value).reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--test-states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    states = np.load(args.test_states)
    if states.shape != (400, 3):
        raise ValueError(f"Expected frozen test states (400,3), got {states.shape}")
    rows: dict[int, dict] = {}
    files = sorted(args.shards.glob("test_shard*.mat"))
    if len(files) != 4:
        raise ValueError(f"Expected four test shards, found {len(files)}")
    for path in files:
        records = read_records(path)
        for i in range(len(records["index"])):
            index = scalar(records["index"][i], int)
            if index in rows:
                raise ValueError(f"duplicate test index {index}")
            row_state = np.asarray(records["initial_state"][i], dtype=float).reshape(-1)
            controls = np.asarray(records["controls"][i], dtype=float).reshape(-1)
            success = bool(scalar(records["success"][i], int))
            if not success or row_state.shape != (3,) or controls.shape != (20,):
                raise ValueError(f"invalid CVP20 record at index {index}")
            rows[index] = {"state": row_state, "objective": scalar(records["objective"][i]),
                           "solve_seconds": scalar(records["solve_seconds"][i]), "controls": controls,
                           "outer_iterations": scalar(records["outer_iterations"][i], int)}
    if sorted(rows) != list(range(400)):
        raise ValueError("test shards do not cover exactly indices 0..399")
    ordered = [rows[i] for i in range(400)]
    solved_states = np.vstack([row["state"] for row in ordered])
    max_state_error = float(np.max(np.abs(solved_states - states)))
    if max_state_error > 1e-12:
        raise ValueError(f"CVP20 shard state order does not match frozen test states; max error={max_state_error:.3e}")
    objectives = np.asarray([row["objective"] for row in ordered], dtype=float)
    controls = np.vstack([row["controls"] for row in ordered])
    seconds = np.asarray([row["solve_seconds"] for row in ordered], dtype=float)
    args.output.mkdir(parents=True)
    np.save(args.output / "reference_objectives.npy", objectives)
    np.save(args.output / "reference_controls.npy", controls)
    with (args.output / "per_point_cold_reference.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_index", "objective", "cold_solve_seconds", "outer_iterations"))
        writer.writeheader()
        for i, row in enumerate(ordered):
            writer.writerow({"sample_index": i, "objective": row["objective"], "cold_solve_seconds": row["solve_seconds"], "outer_iterations": row["outer_iterations"]})
    report = {"source_shards": [str(path) for path in files], "samples": 400, "zoh_steps": 20,
              "cold_start_statement": "Existing Jiang--Fu CVP20 cold-start solves were reused; no NLP was re-run.",
              "independent_hds_audit_performed": False,
              "state_order_max_abs_error": max_state_error,
              "mean_cold_solve_seconds": float(seconds.mean()),
              "objective": {"mean": float(objectives.mean()), "min": float(objectives.min()), "max": float(objectives.max())}}
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
