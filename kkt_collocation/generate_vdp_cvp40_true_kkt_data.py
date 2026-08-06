"""Generate independent CVP40 VDP labels with direct-transcription KKT duals.

This is an isolated CVP40 branch.  It keeps the VDP convention: controls are
the only NLP variables and ``trust-constr`` supplies the finite-dimensional
path-constraint multipliers.  The labels are not continuous-time multipliers.
"""
from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from generate_vdp_cvp_reduced_data import Config, ReducedVDPCVP, grid_states


_WORKER: ReducedVDPCVP | None = None


def _init(cfg: Config) -> None:
    global _WORKER
    _WORKER = ReducedVDPCVP(cfg, audit_references=True)


def _solve(state: np.ndarray) -> dict:
    if _WORKER is None:
        raise RuntimeError("VDP worker was not initialized")
    return _WORKER.solve(np.asarray(state, dtype=float))


def _solve_indexed(payload: tuple[int, np.ndarray]) -> tuple[int, dict]:
    index, state = payload
    return index, _solve(state)


def _records(data: dict) -> list[dict]:
    rows = []
    for i in range(len(data["initial_state"])):
        rows.append({
            "index": i,
            "success": True,
            "initial_state": data["initial_state"][i].tolist(),
            "controls": data["optimal_controls"][i].tolist(),
            "objective": float(data["objective"][i]),
            "path_duals": data["path_duals"][i].tolist(),
            "event_located_max_g": float(data["hds_gmax"][i]),
            "solve_seconds": float(data["solve_seconds"][i]),
            "solver_optimality": float(data["optimality"][i]),
            "solver_constraint_violation": float(data["constraint_violation"][i]),
            "cold_start_attempts": int(data["cold_start_attempts"][i]),
            "dual_source": "trust-constr discretized-path multiplier",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--resume", action="store_true",
                        help="Resume the same frozen grid from records.jsonl after an interrupted run.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {args.output}; use --resume only for this exact frozen grid")

    # Match the original VDP constraint margin, but use the requested CVP40
    # controls and the CSTR-style ten RK4 substeps per ZOH segment.
    cfg = replace(Config(), zoh_steps=40, substeps_per_zoh=10,
                  collocation_safety_margin=1e-4)
    states = grid_states(args.grid_size)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    states_path = args.output / "initial_states.npy"
    if states_path.exists():
        saved_states = np.asarray(np.load(states_path), dtype=float)
        if saved_states.shape != states.shape or not np.array_equal(saved_states, states):
            raise ValueError("--resume grid does not exactly match the saved initial_states.npy")
    else:
        np.save(states_path, states)
    records_path = args.output / "records.jsonl"
    completed: dict[int, dict] = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[int(row["index"])] = row
    pending = [(index, state) for index, state in enumerate(states) if index not in completed]
    with records_path.open("a", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(cfg,)) as pool:
            futures = [pool.submit(_solve_indexed, payload) for payload in pending]
            for future in as_completed(futures):
                index, row = future.result()
                row["index"] = index
                serialized = {
                    "index": index, "success": True,
                    "initial_state": np.asarray(row["initial_state"]).tolist(),
                    "controls": np.asarray(row["optimal_controls"]).tolist(),
                    "objective": float(row["objective"]),
                    "path_duals": np.asarray(row["path_duals"]).tolist(),
                    "event_located_max_g": float(row["hds_gmax"]),
                    "solve_seconds": float(row["solve_seconds"]),
                    "solver_optimality": float(row["optimality"]),
                    "solver_constraint_violation": float(row["constraint_violation"]),
                    "cold_start_attempts": int(row["cold_start_attempts"]),
                    "dual_source": "trust-constr discretized-path multiplier",
                }
                handle.write(json.dumps(serialized) + "\n")
                handle.flush()
                completed[index] = row
                print(f"{len(completed)}/{len(states)} index={index} t={row['solve_seconds']:.2f}s", flush=True)
    if len(completed) != len(states):
        raise RuntimeError(f"Generation stopped with {len(completed)}/{len(states)} labels checkpointed")
    rows = [completed[index] for index in range(len(states))]
    if any(float(row["hds_gmax"]) > 1e-8 for row in rows):
        bad = max(float(row["hds_gmax"]) for row in rows)
        raise RuntimeError(f"A CVP40 VDP label failed the event-located audit: {bad:.3e}")

    data = {key: np.asarray([row[key] for row in rows]) for key in
            ("initial_state", "optimal_controls", "objective", "path_duals", "hds_gmax",
             "solve_seconds", "cold_start_attempts", "optimality", "constraint_violation")}
    data["config"] = asdict(cfg)
    data["description"] = (
        "VDP CVP40 reduced-space RK4 transcription. Path duals are returned by "
        "trust-constr for the finite-dimensional discrete NLP, not continuous-time multipliers."
    )
    data["cold_start_protocol"] = (
        "fixed all-0.5 controls, then fixed all-0.2 and all-0.8 retries only; "
        "no neighbouring label or policy warm start"
    )
    with (args.output / "labels.pkl").open("wb") as handle:
        pickle.dump(data, handle)
    summary = {
        "purpose": "Independent CVP40 VDP KKT labels.",
        "labels_requested": len(rows), "labels_successful": len(rows), "labels_failed": 0,
        "config": asdict(cfg), "initial_state_layout": f"{args.grid_size}x{args.grid_size} endpoint grid",
        "mean_solve_seconds": float(np.mean(data["solve_seconds"])),
        "max_event_located_g": float(np.max(data["hds_gmax"])),
        "mean_solver_optimality": float(np.mean(data["optimality"])),
        "max_solver_optimality": float(np.max(data["optimality"])),
        "multiplier_interpretation": "Finite-dimensional path multipliers returned by trust-constr for the reduced RK4 transcription; not continuous-time multipliers.",
        "cold_start_protocol": data["cold_start_protocol"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
