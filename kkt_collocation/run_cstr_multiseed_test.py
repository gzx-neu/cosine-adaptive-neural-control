"""Three-seed CSTR training and 3 x 400 frozen in-domain evaluation.

The 144 direct-RK4 labels are held fixed.  Three independent network
initializations are trained from those labels; each branch is selected by its
own HDS validation gate and evaluated on a separate 400-point LHS test cohort.
Rows are checkpointed immediately, so this computationally expensive audit can
resume safely after interruption without changing completed observations.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.adaptive_kkt_gate import AdaptiveKKTThresholds, audit_raw_hds_peaks  # noqa: E402
from kkt_collocation.run_cstr_full_simulation import (  # noqa: E402
    CSTRConfig, Policy, lhs_states, make_corrector, objective, predict,
    summarise, train_branch,
)


TRAIN_SEEDS = (20260718, 20260725, 20260726)
TEST_SEEDS = (20260724, 20260825, 20260826)
FIELDS = ("method", "training_seed", "sample_index", "CA0", "T0", "nominal_hds_max_g",
          "applied_hds_max_g", "nominal_objective", "applied_objective", "objective_change",
          "accepted", "fallback", "corrected_segments", "mean_abs_lambda_minus_one",
          "inference_seconds", "filter_seconds")


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def cast_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("CA0", "T0", "nominal_hds_max_g", "applied_hds_max_g", "nominal_objective",
                    "applied_objective", "objective_change", "mean_abs_lambda_minus_one",
                    "inference_seconds", "filter_seconds"):
            row[key] = float(row[key])
        for key in ("sample_index", "corrected_segments"):
            row[key] = int(row[key])
        if "training_seed" in row:
            row["training_seed"] = int(row["training_seed"])
        for key in ("accepted", "fallback"):
            row[key] = str(row[key]).strip().lower() == "true"
    return rows


def mean_sd(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    return {"mean": float(data.mean()), "sample_std": float(data.std(ddof=1))}


def bootstrap_seed_one(output: Path) -> list[dict]:
    """Reuse the completed 400-point frozen seed-20260718 audit exactly."""
    source = ROOT / "kkt_collocation" / "results" / "cstr_extended_test400" / "per_sample.csv"
    if not source.exists():
        return []
    rows = cast_rows(source)
    for row in rows:
        row["training_seed"] = TRAIN_SEEDS[0]
    return rows


def train_or_load(seed: int, cfg: CSTRConfig, data: dict, output: Path) -> tuple[Policy, np.ndarray, np.ndarray, dict]:
    checkpoint_path = output / f"cstr_seed{seed}.pth"
    metadata_path = output / f"cstr_seed{seed}_gate.json"
    if checkpoint_path.exists() and metadata_path.exists():
        stored = load_checkpoint(checkpoint_path)
        model = Policy(cfg); model.load_state_dict(stored["model"])
        return model, np.asarray(stored["mean"]), np.asarray(stored["std"]), json.loads(metadata_path.read_text(encoding="utf-8"))

    supervised, mean, std, _ = train_branch(data, cfg, kkt=False, epochs=300, seed=seed)
    validation = lhs_states(cfg, 60, seed + 1000)
    controls, _ = predict(supervised, mean, std, validation)
    corrector = make_corrector(cfg)
    gate = audit_raw_hds_peaks(
        np.asarray([corrector.audit(x, u, cfg.zoh_duration) for x, u in zip(validation, controls)]),
        AdaptiveKKTThresholds(allowed_violation_rate=.05, rate_normalized_violation=.025,
                              allowed_normalized_peak_violation=.03,
                              engineering_constraint_scale=5., numerical_violation_tolerance=1e-8),
    )
    selected = "Supervised"
    model = supervised
    if gate.kkt_refinement_required:
        model, mean, std, _ = train_branch(data, cfg, kkt=True, epochs=30, seed=seed + 500,
                                            base=copy.deepcopy(supervised))
        selected = "KKT-refined"
    metadata = {"training_seed": seed, "validation_seed": seed + 1000,
                "selected_branch": selected, "gate": asdict(gate)}
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "config": asdict(cfg),
                "selected_branch": selected}, checkpoint_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model, mean, std, metadata


def evaluate_seed(seed: int, test_seed: int, model: Policy, mean: np.ndarray, std: np.ndarray,
                  cfg: CSTRConfig, existing: list[dict], writer: csv.DictWriter, handle) -> None:
    states = lhs_states(cfg, 400, test_seed)
    controls, inference = predict(model, mean, std, states)
    done = sum(row["training_seed"] == seed for row in existing)
    corrector = make_corrector(cfg)
    for index in range(done, len(states)):
        state, nominal = states[index], controls[index]
        raw_peak = corrector.audit(state, nominal, cfg.zoh_duration)
        raw_objective = objective(state, nominal, cfg)
        start = time.perf_counter(); outcome = corrector.correct(state, nominal, cfg.zoh_duration)
        elapsed = time.perf_counter() - start
        if not outcome.accepted:
            raise RuntimeError(f"CSTR seed {seed}: HDS fallback at test sample {index}")
        applied_peak = corrector.audit(state, outcome.controls, cfg.zoh_duration)
        applied_objective = objective(state, outcome.controls, cfg)
        lambdas = np.asarray([segment.lambda_value for segment in outcome.segments if segment.lambda_value is not None])
        row = {"method": "Adaptive+HDS", "training_seed": seed, "sample_index": index,
               "CA0": state[0], "T0": state[1], "nominal_hds_max_g": raw_peak,
               "applied_hds_max_g": applied_peak, "nominal_objective": raw_objective,
               "applied_objective": applied_objective, "objective_change": applied_objective - raw_objective,
               "accepted": True, "fallback": False,
               "corrected_segments": int(sum(segment.corrected for segment in outcome.segments)),
               "mean_abs_lambda_minus_one": float(np.mean(np.abs(lambdas - 1.0))) if len(lambdas) else 0.0,
               "inference_seconds": inference, "filter_seconds": elapsed}
        writer.writerow(row); handle.flush(); existing.append(row)
        if (index + 1) % 20 == 0:
            print(f"seed {seed}: completed {index + 1}/400", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "cstr_full_simulation_900" / "cstr_labels.pkl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "cstr_multiseed_test1200_900_clean")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with args.labels.open("rb") as handle:
        data = pickle.load(handle)
    cfg = CSTRConfig(**data["config"])
    csv_path = args.output / "per_sample.csv"
    existing = cast_rows(csv_path)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for seed, test_seed in zip(TRAIN_SEEDS, TEST_SEEDS):
            if seed == TRAIN_SEEDS[0] and sum(row["training_seed"] == seed for row in existing) == 400:
                continue
            model, mean, std, _ = train_or_load(seed, cfg, data, args.output)
            evaluate_seed(seed, test_seed, model, mean, std, cfg, existing, writer, handle)

    rows = cast_rows(csv_path)
    if len(rows) != 1200:
        print(f"checkpointed {len(rows)}/1200 rows; rerun this script to resume.")
        return
    per_seed = {str(seed): summarise([row for row in rows if row["training_seed"] == seed]) for seed in TRAIN_SEEDS}
    metrics = ("raw_violation_rate_percent", "raw_peak_max_K", "accepted_rate_percent", "fallback_rate_percent",
               "corrected_segments_mean", "accepted_peak_max_K", "mean_objective_change", "mean_inference_ms", "mean_hds_ms")
    report = {"model": "representative literature-informed mechanistic CSTR; not plant data",
              "config": asdict(cfg), "labels": len(data["initial_state"]), "training_seeds": list(TRAIN_SEEDS),
              "test_seeds": list(TEST_SEEDS), "test_points_per_seed": 400, "test_points_total": 1200,
              "protocol": f"Fixed {len(data['initial_state'])}-label dataset; three independently initialized neural trainings; independent 400-point LHS test cohort per seed; event-located HDS--lambda auditing.",
              "per_seed": per_seed,
              "aggregate": {metric: mean_sd([per_seed[str(seed)][metric] for seed in TRAIN_SEEDS]) for metric in metrics}}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
