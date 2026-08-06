"""Fair single-seed S vs direct-KKT vs S+K continuation ablation.

This script is deliberately separate from the paper-result drivers.  It
reuses their labels, held-out splits, bounded policies, 31-candidate HDS
corrector, and direct-transcription KKT residual, but writes only to
``results/two_stage_vs_kkt_only``.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.adaptive_event_hds import (
    AdaptiveEventHDSConfig,
    AdaptiveEventHDSCorrector,
)
from offline_safe_control.kkt_regularization import augmented_lagrangian_kkt_residual
from kkt_collocation.run_penicillin_ablation import (
    Config as PenicillinTrainConfig, DT as PEN_DT, N as PEN_N, Policy as PenicillinPolicy,
    UMAX as PEN_UMAX, g as pen_g, gdot as pen_gdot, ode as pen_ode, rollout as pen_rollout,
    terminal_product,
)
from kkt_collocation.run_vdp_ablation import (
    constraint as vdp_g, constraint_derivative as vdp_gdot, terminal_cost, vdp_ode,
)
from kkt_collocation.train_vdp_kkt_policy import (
    KKTPolicyValueNetwork, TrainConfig as VDPTrainConfig, differentiable_rollout,
)


_WORKER_CORRECTOR: AdaptiveEventHDSCorrector | None = None
_WORKER_NAME: str | None = None
_WORKER_DURATION: float | None = None


def _evaluation_worker_init(name: str, lambda_grid_size: int) -> None:
    """One HDS corrector per process: keeps a 400-point audit below the job time limit."""
    global _WORKER_CORRECTOR, _WORKER_NAME, _WORKER_DURATION
    _WORKER_NAME = name
    if name == "vdp":
        _WORKER_DURATION = 0.5
        _WORKER_CORRECTOR = AdaptiveEventHDSCorrector(
            vdp_ode, vdp_g, vdp_gdot, (-0.3, 1.0),
            AdaptiveEventHDSConfig(grid_size=lambda_grid_size),
        )
    else:
        _WORKER_DURATION = PEN_DT
        _WORKER_CORRECTOR = AdaptiveEventHDSCorrector(
            pen_ode, pen_g, pen_gdot, (0.0, PEN_UMAX),
            AdaptiveEventHDSConfig(grid_size=lambda_grid_size),
        )


def _evaluate_one_worker(task: tuple[int, np.ndarray, np.ndarray, float, float, float, float]) -> dict:
    if _WORKER_CORRECTOR is None or _WORKER_NAME is None or _WORKER_DURATION is None:
        raise RuntimeError("HDS worker was not initialized")
    index, state, nominal, reference, inference_seconds, reference_solve_seconds, reference_hds_gmax = task
    initial = np.asarray(state, float) if _WORKER_NAME == "vdp" else np.array([1.0, float(state), 0.001, 250.0])
    nominal_g = float(_WORKER_CORRECTOR.audit(initial, nominal, _WORKER_DURATION))
    nominal_objective = (terminal_cost(initial, nominal, _WORKER_CORRECTOR, _WORKER_DURATION)
                         if _WORKER_NAME == "vdp" else -terminal_product(float(state), nominal, _WORKER_CORRECTOR))
    started = time.perf_counter()
    outcome = _WORKER_CORRECTOR.correct(initial, nominal, _WORKER_DURATION)
    correction_seconds = time.perf_counter() - started
    accepted = bool(outcome.accepted)
    applied = outcome.controls if accepted else nominal
    applied_g = float(_WORKER_CORRECTOR.audit(initial, applied, _WORKER_DURATION)) if accepted else np.nan
    applied_objective = ((terminal_cost(initial, applied, _WORKER_CORRECTOR, _WORKER_DURATION)
                         if _WORKER_NAME == "vdp" else -terminal_product(float(state), applied, _WORKER_CORRECTOR)) if accepted else np.nan)
    denom = abs(reference) if np.isfinite(reference) and abs(reference) > 1e-12 else np.nan
    return {"sample_index": index, "nominal_hds_max_g": nominal_g, "applied_hds_max_g": applied_g,
            "accepted": accepted, "fallback": not accepted, "nominal_objective": nominal_objective,
            "applied_objective": applied_objective, "nlp_reference_objective": reference,
            "cold_reference_solve_seconds": reference_solve_seconds,
            "cold_reference_hds_gmax": reference_hds_gmax,
            "nominal_relative_objective_gap": (nominal_objective - reference) / denom if np.isfinite(denom) else np.nan,
            "applied_relative_objective_gap": (applied_objective - reference) / denom if np.isfinite(denom) else np.nan,
            "objective_change": applied_objective - nominal_objective if accepted else np.nan,
            "corrected_segments": int(sum(segment.corrected for segment in outcome.segments)),
            "inference_seconds": inference_seconds, "hds_correction_seconds": correction_seconds,
            "total_seconds": inference_seconds + correction_seconds}


@dataclass(frozen=True)
class ExperimentConfig:
    supervised_epochs: int = 200
    continuation_epochs: int = 20
    supervised_learning_rate: float = 1e-3
    continuation_learning_rate: float = 1e-5
    kkt_weight_vdp: float = 1e-3
    kkt_weight_penicillin: float = 1e-2
    augmented_penalty: float = 10.0
    anchor_weight: float = 1.0
    lambda_grid_size: int = 31
    # Deliberately zero: this ablation uses exactly the requested pure S loss.
    rollout_consistency_weight: float = 0.0

    @property
    def total_epochs(self) -> int:
        return self.supervised_epochs + self.continuation_epochs


def json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def dump_json(path: Path, content: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2, default=json_default)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().detach().cpu())


class Problem:
    """Thin adapter retaining each problem's established model and rollout."""

    def __init__(self, name: str, device: torch.device, seed: int, cfg: ExperimentConfig):
        self.name, self.device, self.seed, self.cfg = name, device, seed, cfg
        self.duration = 0.5 if name == "vdp" else PEN_DT
        self.nodes = 101 if name == "vdp" else 801
        # The frozen Jiang--Fu export is the single cold-start reference for
        # all 400 test points.  Each row is already independently audited.
        self.reference_records: dict[int, dict[str, float]] = {}
        if name == "vdp":
            self.data_path = ROOT / "kkt_collocation/data/vdp_kkt_20x20.pkl"
            with self.data_path.open("rb") as handle:
                data = pickle.load(handle)
            self.initial = torch.tensor(np.asarray(data["initial_state"], np.float32), device=device)
            self.u_ref = torch.tensor(np.asarray(data["optimal_controls"], np.float32), device=device)
            self.j_ref = torch.tensor(np.asarray(data["objective"], np.float32).reshape(-1, 1), device=device)
            self.mu_ref = torch.tensor(np.asarray(data["path_duals"], np.float32), device=device)
            self.train_cfg = VDPTrainConfig(epochs=cfg.total_epochs)
            split_dir = ROOT / "kkt_collocation/results/final_multiseed_vdp900_penalty_seed20260751"
            self.validation_states = np.load(split_dir / "validation_states.npy")
            self.test_states = np.load(split_dir / "test_states.npy")
            self.reference_records = self._load_jiang_fu_references("VDP", "vdp")
        else:
            self.data_path = ROOT / "kkt_collocation/data/penicillin_kkt_400_true_duals.pkl"
            with self.data_path.open("rb") as handle:
                data = pickle.load(handle)
            x2 = np.asarray(data["initial_state"], np.float32)[:, 1]
            self.initial = torch.tensor(x2, device=device)
            self.u_ref = torch.tensor(np.asarray(data["optimal_controls"], np.float32), device=device)
            self.j_ref = torch.tensor(np.asarray(data["objective"], np.float32).reshape(-1, 1), device=device)
            self.mu_ref = torch.tensor(np.asarray(data["path_duals"], np.float32), device=device)
            if self.mu_ref.shape[1] != 801:
                raise ValueError("penicillin true-dual labels must have 801 path nodes")
            self.train_cfg = PenicillinTrainConfig(epochs=cfg.total_epochs, substeps=80, rollout_weight=0.0,
                                                   seed=seed, kkt_weight=cfg.kkt_weight_penicillin,
                                                   penalty=cfg.augmented_penalty)
            split_dir = ROOT / "kkt_collocation/results/final_multiseed_penicillin400_penalty_seed20260761"
            self.validation_states = np.load(split_dir / "validation_x2.npy")
            self.test_states = np.load(split_dir / "test_x2.npy")
            self.reference_records = self._load_jiang_fu_references("Penicillin", "pen")

        if self.u_ref.shape[0] != 400:
            raise ValueError(f"Expected exactly 400 labels, got {self.u_ref.shape[0]}")
        self.mean = self.input_values().mean(dim=0)
        self.std = self.input_values().std(dim=0).clamp_min(1e-8)
        self.j_mean = self.j_ref.mean()
        self.j_std = self.j_ref.std().clamp_min(1e-8)

    @staticmethod
    def _load_jiang_fu_references(problem_name: str, point_prefix: str) -> dict[int, dict[str, float]]:
        """Load the archived matched-400 *cold* references in frozen point order."""
        path = ROOT / "kkt_collocation/results/jiang_fu_matched400_comparison/per_point_seed.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["problem"] == problem_name]
        if len(rows) != 400:
            raise ValueError(f"Expected 400 Jiang--Fu references for {problem_name}, found {len(rows)}")
        records: dict[int, dict[str, float]] = {}
        for index, row in enumerate(rows):
            expected_id = f"{point_prefix}_{index + 1:03d}"
            if row["point_id"] != expected_id:
                raise ValueError(f"Frozen {problem_name} reference ordering mismatch: {row['point_id']} != {expected_id}")
            records[index] = {
                "objective": float(row["jiang_objective"]),
                "solve_seconds": float(row["jiang_solve_seconds"]),
                "hds_gmax": float(row["jiang_hds_gmax"]),
            }
        return records

    def input_values(self) -> torch.Tensor:
        return self.initial[:, :2] if self.name == "vdp" else self.initial[:, None]

    def normalized_train_input(self) -> torch.Tensor:
        return (self.input_values() - self.mean) / self.std

    def make_model(self) -> nn.Module:
        return KKTPolicyValueNetwork(self.train_cfg).to(self.device) if self.name == "vdp" else PenicillinPolicy().to(self.device)

    def rollout(self, controls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.name == "vdp":
            return differentiable_rollout(self.initial, controls, self.train_cfg)
        return pen_rollout(self.initial, controls, self.train_cfg.substeps)

    def kkt_weight(self) -> float:
        return self.cfg.kkt_weight_vdp if self.name == "vdp" else self.cfg.kkt_weight_penicillin

    def predict(self, model: nn.Module, states: np.ndarray) -> tuple[np.ndarray, float]:
        features = states[:, :2] if self.name == "vdp" else states[:, None]
        x = torch.tensor((features - self.mean.detach().cpu().numpy()) / self.std.detach().cpu().numpy(), dtype=torch.float32, device=self.device)
        start = time.perf_counter()
        with torch.no_grad():
            _, controls = model(x)
        return controls.cpu().numpy(), (time.perf_counter() - start) / len(states)

    def corrector(self) -> AdaptiveEventHDSCorrector:
        if self.name == "vdp":
            return AdaptiveEventHDSCorrector(
                vdp_ode, vdp_g, vdp_gdot, (-0.3, 1.0),
                AdaptiveEventHDSConfig(grid_size=self.cfg.lambda_grid_size),
            )
        return AdaptiveEventHDSCorrector(
            pen_ode, pen_g, pen_gdot, (0.0, PEN_UMAX),
            AdaptiveEventHDSConfig(grid_size=self.cfg.lambda_grid_size),
        )

    def initial_state(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, float) if self.name == "vdp" else np.array([1.0, float(value), 0.001, 250.0])

    def cost(self, value: np.ndarray, controls: np.ndarray, corrector: AdaptiveEventHDSCorrector) -> float:
        if self.name == "vdp":
            return terminal_cost(self.initial_state(value), controls, corrector, self.duration)
        return -terminal_product(float(value), controls, corrector)


def loss_terms(problem: Problem, model: nn.Module, *, need_kkt: bool, anchor: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    predicted_j, controls = model(problem.normalized_train_input())
    control_mse = nn.functional.mse_loss(controls, problem.u_ref)
    objective_mse = nn.functional.mse_loss(predicted_j, (problem.j_ref - problem.j_mean) / problem.j_std)
    supervised = control_mse + 0.1 * objective_mse
    result = {"control_mse": control_mse, "objective_mse": objective_mse, "supervised": supervised}
    if need_kkt:
        objective, path_g = problem.rollout(controls)
        raw = augmented_lagrangian_kkt_residual(objective, controls, path_g, problem.mu_ref, problem.cfg.augmented_penalty)
        # This established scale normalization is identical in K-only and S+K.
        normalized = raw.total if problem.name == "vdp" else raw.total / raw.total.detach().clamp_min(1.0)
        result.update({"kkt_raw": raw.total, "kkt_loss": normalized,
                       "stationarity": raw.stationarity, "primal": raw.primal_feasibility,
                       "complementarity": raw.complementarity})
    if anchor is not None:
        result["anchor"] = nn.functional.mse_loss(controls, anchor)
    return result


def train_method(problem: Problem, method: str, base_state: dict, output: Path) -> tuple[nn.Module, dict, list[dict]]:
    model = problem.make_model()
    model.load_state_dict(base_state)
    history: list[dict] = []
    failed, reason = False, None
    start = time.perf_counter()

    # S+K uses a frozen prediction of the stage-one policy only as its stated
    # anchor.  K-only receives neither controls/objectives nor an anchor.
    stages = [("supervised", problem.cfg.total_epochs, False, None, problem.cfg.supervised_learning_rate)] if method == "S" else []
    if method == "S+K":
        stages.append(("supervised", problem.cfg.supervised_epochs, False, None, problem.cfg.supervised_learning_rate))
    if method == "K-only":
        stages.append(("kkt_only", problem.cfg.total_epochs, True, None, problem.cfg.supervised_learning_rate))
    anchor = None
    for stage, epochs, use_kkt, stage_anchor, learning_rate in stages:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(1, epochs + 1):
            try:
                terms = loss_terms(problem, model, need_kkt=use_kkt, anchor=stage_anchor)
                loss = terms["kkt_loss"] if method == "K-only" else terms["supervised"]
                if use_kkt and method != "K-only":
                    loss = loss + problem.kkt_weight() * terms["kkt_loss"]
                if stage_anchor is not None:
                    loss = loss + problem.cfg.anchor_weight * terms["anchor"]
                if not finite_tensor(loss):
                    raise FloatingPointError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if epoch == 1 or epoch == epochs or epoch % 20 == 0:
                    history.append({"stage": stage, "epoch": epoch, "loss": float(loss.detach()),
                                    **{k: float(v.detach()) for k, v in terms.items()}})
            except (FloatingPointError, RuntimeError) as exc:
                failed, reason = True, f"{type(exc).__name__}: {exc}"
                break
        if failed:
            break
        if method == "S+K" and stage == "supervised":
            with torch.no_grad():
                _, anchor = model(problem.normalized_train_input())
            stages.append(("continuation", problem.cfg.continuation_epochs, True, anchor.detach(), problem.cfg.continuation_learning_rate))
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.enable_grad():
        metrics = loss_terms(problem, model, need_kkt=True)
    final = {"completed": not failed, "numerical_failure": failed, "failure_reason": reason,
             "train_seconds": elapsed, "final_control_mse": float(metrics["control_mse"].detach()),
             "final_objective_mse": float(metrics["objective_mse"].detach()),
             "final_kkt_residual": float(metrics["kkt_raw"].detach()),
             "kkt_residual_components": {k: float(metrics[k].detach()) for k in ("stationarity", "primal", "complementarity")}}
    torch.save({"model": model.state_dict(), "method": method, "problem": problem.name,
                "normalization": {"mean": problem.mean.detach().cpu(), "std": problem.std.detach().cpu(),
                                  "objective_mean": problem.j_mean.detach().cpu(), "objective_std": problem.j_std.detach().cpu()},
                "training": final}, output / f"{method}.pth")
    return model, final, history


def summarize_rows(rows: list[dict]) -> dict:
    # The established penicillin driver also evaluates independent HDS audits
    # in workers.  This changes no numerical setting and keeps all 400 audits.
    value = lambda key: np.asarray([row[key] for row in rows], float)
    accepted = value("accepted").astype(bool)
    finite = lambda key: float(np.nanmean(value(key)))
    summary = {"samples": len(rows), "reference_samples": int(np.isfinite(value("nlp_reference_objective")).sum()),
               "nominal_relative_objective_gap": finite("nominal_relative_objective_gap"),
               "nominal_violation_rate": float(np.mean(value("nominal_hds_max_g") > 1e-8)),
               "nominal_max_g": float(np.max(value("nominal_hds_max_g"))),
               "mean_nominal_objective_gap": finite("nominal_relative_objective_gap"),
               "continuous_time_audit_acceptance_rate": float(np.mean(accepted)),
               "offline_optimizer_fallback_rate": float(1.0 - np.mean(accepted)),
               "final_max_g": float(np.nanmax(value("applied_hds_max_g"))),
               "mean_corrected_segments": finite("corrected_segments"),
               "hds_relative_objective_gap": finite("applied_relative_objective_gap"),
               "mean_hds_objective_change": finite("objective_change"),
               "mean_inference_seconds": finite("inference_seconds"),
               "mean_audit_seconds": finite("hds_correction_seconds"), "mean_total_seconds": finite("total_seconds"),
               "mean_cold_reference_solve_seconds": finite("cold_reference_solve_seconds"),
               "mean_cold_reference_hds_gmax": finite("cold_reference_hds_gmax"),
               "mean_cold_reference_speedup": float(np.nanmean(value("cold_reference_solve_seconds") / value("total_seconds")))}
    return summary


def evaluate(problem: Problem, method: str, model: nn.Module, infer_seconds: float,
             start: int = 0, stop: int | None = None, workers: int = 1) -> tuple[list[dict], dict]:
    stop = len(problem.test_states) if stop is None else stop
    indices = np.arange(start, stop)
    states = problem.test_states[indices]
    controls, infer_seconds = problem.predict(model, states)
    tasks = [(int(index), state, nominal,
              problem.reference_records.get(int(index), {}).get("objective", np.nan), infer_seconds,
              problem.reference_records.get(int(index), {}).get("solve_seconds", np.nan),
              problem.reference_records.get(int(index), {}).get("hds_gmax", np.nan))
             for index, state, nominal in zip(indices, states, controls)]
    if workers == 1:
        _evaluation_worker_init(problem.name, problem.cfg.lambda_grid_size)
        rows = [_evaluate_one_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_evaluation_worker_init,
                                 initargs=(problem.name, problem.cfg.lambda_grid_size)) as pool:
            rows = list(pool.map(_evaluate_one_worker, tasks, chunksize=1))
    for row in rows:
        row["method"] = method
    return rows, summarize_rows(rows)


def run_problem(name: str, seed: int, root_output: Path, cfg: ExperimentConfig) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    problem = Problem(name, device, seed, cfg)
    output = root_output / f"{name}_seed{seed}"
    output.mkdir(parents=True, exist_ok=False)
    config = {"experiment": asdict(cfg), "problem": name, "seed": seed, "device": str(device),
              "torch_version": torch.__version__, "numpy_version": np.__version__, "python_version": sys.version,
              "platform": platform.platform(), "label_source": str(problem.data_path),
              "split_source": "current independent validation/test arrays from the existing fixed multiseed main experiment",
              "cold_reference_source": "results/jiang_fu_matched400_comparison/per_point_seed.csv (frozen pointwise Jiang--Fu cold solves)",
              "validation_samples": len(problem.validation_states), "test_samples": len(problem.test_states),
              "kkt_multiplier_note": "KKT multipliers are finite-dimensional KKT multipliers of the discretized-transcription NLP, not continuous-time multipliers.",
              "k_only_note": "K-only uses reference discretized-NLP multipliers but no reference controls or objective values in its loss.",
              "hds_note": "HDS is continuous-time numerical audit evidence under the declared model and numerical settings; it is not an absolute safety guarantee."}
    dump_json(output / "config.json", config)
    np.save(output / "validation_states.npy", problem.validation_states)
    np.save(output / "test_states.npy", problem.test_states)

    set_seed(seed)
    base = problem.make_model()
    base_state = copy.deepcopy(base.state_dict())
    methods, histories = {}, {}
    for method in ("S", "K-only", "S+K"):
        print(f"[{name} seed {seed}] training {method}", flush=True)
        model, training, history = train_method(problem, method, base_state, output)
        rows, deployed = evaluate(problem, method, model, 0.0)
        methods[method] = {"training": training, "deployment": deployed}
        histories[method] = history
        with (output / f"test_sample_log_{method}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    dump_json(output / "training_log.json", histories)
    validation = {"same_fixed_validation_set_saved_to": "validation_states.npy",
                  "selection_statement": "No hyperparameters were selected or changed using test results; fixed configuration was applied to all methods."}
    dump_json(output / "validation_summary.json", validation)
    result = {"config": config, "methods": methods,
              "reference_gap_note": "Relative objective gaps use the frozen Jiang--Fu matched-400 cold-start reference pointwise; all 400 test points are included."}
    dump_json(output / "summary.json", result)
    return {"problem": name, "seed": seed, "directory": str(output), "methods": methods}


def prepare_stage_problem(name: str, seed: int, root_output: Path, cfg: ExperimentConfig) -> tuple[Problem, Path, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    problem = Problem(name, device, seed, cfg)
    output = root_output / f"{name}_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    config = {"experiment": asdict(cfg), "problem": name, "seed": seed, "device": str(device),
              "torch_version": torch.__version__, "numpy_version": np.__version__, "python_version": sys.version,
              "platform": platform.platform(), "label_source": str(problem.data_path),
              "split_source": "current independent validation/test arrays from the existing fixed multiseed main experiment",
              "cold_reference_source": "results/jiang_fu_matched400_comparison/per_point_seed.csv (frozen pointwise Jiang--Fu cold solves)",
              "validation_samples": len(problem.validation_states), "test_samples": len(problem.test_states),
              "kkt_multiplier_note": "KKT multipliers are finite-dimensional KKT multipliers of the discretized-transcription NLP, not continuous-time multipliers.",
              "k_only_note": "K-only uses reference discretized-NLP multipliers but no reference controls or objective values in its loss.",
              "hds_note": "HDS is continuous-time numerical audit evidence under the declared model and numerical settings; it is not an absolute safety guarantee."}
    dump_json(output / "config.json", config)
    np.save(output / "validation_states.npy", problem.validation_states)
    np.save(output / "test_states.npy", problem.test_states)
    return problem, output, config


def stage_train(problem: Problem, output: Path, method: str) -> None:
    set_seed(problem.seed)
    base = problem.make_model()
    model, training, history = train_method(problem, method, copy.deepcopy(base.state_dict()), output)
    del model
    dump_json(output / f"training_log_{method}.json", {"training": training, "history": history})


def stage_konly_chunk(problem: Problem, output: Path, start: int, stop: int) -> None:
    """Resume the *same* Adam trajectory for K-only across terminal-sized jobs."""
    if start < 0 or stop <= start or stop > problem.cfg.total_epochs:
        raise ValueError("invalid K-only epoch chunk")
    progress_path = output / "K-only_progress.pth"
    if start == 0:
        set_seed(problem.seed)
        model = problem.make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=problem.cfg.supervised_learning_rate)
        history: list[dict] = []
    else:
        progress = torch.load(progress_path, map_location=problem.device, weights_only=False)
        if int(progress["completed_epochs"]) != start:
            raise ValueError("K-only resume start does not match saved optimizer trajectory")
        model = problem.make_model(); model.load_state_dict(progress["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=problem.cfg.supervised_learning_rate)
        optimizer.load_state_dict(progress["optimizer"])
        history = progress["history"]
    model.train(); started = time.perf_counter()
    for epoch in range(start + 1, stop + 1):
        terms = loss_terms(problem, model, need_kkt=True)
        loss = terms["kkt_loss"]
        if not finite_tensor(loss):
            raise FloatingPointError(f"non-finite K-only loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if epoch == 1 or epoch == stop or epoch % 20 == 0:
            history.append({"stage": "kkt_only", "epoch": epoch, "loss": float(loss.detach()),
                            **{key: float(value.detach()) for key, value in terms.items()}})
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "completed_epochs": stop,
                "history": history}, progress_path)
    if stop == problem.cfg.total_epochs:
        model.eval()
        with torch.enable_grad():
            metrics = loss_terms(problem, model, need_kkt=True)
        training = {"completed": True, "numerical_failure": False, "failure_reason": None,
                    "train_seconds": None, "final_control_mse": float(metrics["control_mse"].detach()),
                    "final_objective_mse": float(metrics["objective_mse"].detach()),
                    "final_kkt_residual": float(metrics["kkt_raw"].detach()),
                    "kkt_residual_components": {key: float(metrics[key].detach()) for key in ("stationarity", "primal", "complementarity")},
                    "chunked_execution_note": "Optimizer state was preserved exactly across fixed execution chunks."}
        torch.save({"model": model.state_dict(), "method": "K-only", "problem": problem.name,
                    "normalization": {"mean": problem.mean.detach().cpu(), "std": problem.std.detach().cpu(),
                                      "objective_mean": problem.j_mean.detach().cpu(), "objective_std": problem.j_std.detach().cpu()},
                    "training": training}, output / "K-only.pth")
        dump_json(output / "training_log_K-only.json", {"training": training, "history": history})
    print(f"[{problem.name}] K-only epochs {start + 1}-{stop} completed in {time.perf_counter() - started:.2f}s", flush=True)


def mark_konly_failure(problem: Problem, output: Path, reason: str) -> None:
    """Persist the last finite checkpoint and make a failed run auditable."""
    progress = torch.load(output / "K-only_progress.pth", map_location=problem.device, weights_only=False)
    model = problem.make_model(); model.load_state_dict(progress["model"]); model.eval()
    with torch.enable_grad():
        metrics = loss_terms(problem, model, need_kkt=True)
    training = {"completed": False, "numerical_failure": True, "failure_reason": reason,
                "last_finite_epoch": int(progress["completed_epochs"]), "train_seconds": None,
                "final_control_mse": float(metrics["control_mse"].detach()),
                "final_objective_mse": float(metrics["objective_mse"].detach()),
                "final_kkt_residual": float(metrics["kkt_raw"].detach()),
                "kkt_residual_components": {key: float(metrics[key].detach()) for key in ("stationarity", "primal", "complementarity")}}
    torch.save({"model": model.state_dict(), "method": "K-only", "problem": problem.name,
                "training": training}, output / "K-only.pth")
    dump_json(output / "training_log_K-only.json", {"training": training, "history": progress["history"]})


def stage_evaluate(problem: Problem, output: Path, method: str, start: int, stop: int) -> None:
    checkpoint = torch.load(output / f"{method}.pth", map_location=problem.device, weights_only=False)
    model = problem.make_model(); model.load_state_dict(checkpoint["model"]); model.eval()
    rows, partial = evaluate(problem, method, model, 0.0, start, stop, workers=1)
    with (output / f"test_sample_log_{method}_part_{start:03d}_{stop:03d}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    dump_json(output / f"evaluation_{method}_part_{start:03d}_{stop:03d}.json", partial)


def aggregate_stage(output: Path, config: dict) -> dict:
    methods = {}
    all_rows = []
    for method in ("S", "K-only", "S+K"):
        paths = sorted(output.glob(f"test_sample_log_{method}_part_*.csv"))
        if not paths:
            training = json.load((output / f"training_log_{method}.json").open(encoding="utf-8"))["training"]
            if training.get("numerical_failure"):
                methods[method] = {"training": training, "deployment": {"not_evaluable": True, "reason": "training numerical failure before a 220-epoch policy was available"}}
                continue
            raise FileNotFoundError(f"No evaluation chunks for {method}")
        rows = []
        for path in paths:
            for row in csv.DictReader(path.open(encoding="utf-8")):
                converted = {"method": row["method"]}
                for key, value in row.items():
                    if key == "method":
                        continue
                    converted[key] = (value == "True") if key in ("accepted", "fallback") else float(value)
                rows.append(converted)
        indices = [int(row["sample_index"]) for row in rows]
        if sorted(indices) != list(range(config["test_samples"])):
            raise ValueError(f"{method} chunks do not cover each test sample exactly once")
        training = json.load((output / f"training_log_{method}.json").open(encoding="utf-8"))["training"]
        methods[method] = {"training": training, "deployment": summarize_rows(rows)}
        all_rows.extend(rows)
    all_rows.sort(key=lambda row: (row["method"], row["sample_index"]))
    if all_rows:
        with (output / "test_sample_log.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    validation = {"same_fixed_validation_set_saved_to": "validation_states.npy",
                  "selection_statement": "No hyperparameters were selected or changed using test results; fixed configuration was applied to all methods."}
    dump_json(output / "validation_summary.json", validation)
    result = {"config": config, "methods": methods,
              "reference_gap_note": "Relative objective gaps use the frozen Jiang--Fu matched-400 cold-start reference pointwise; all 400 test points are included."}
    dump_json(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("vdp", "penicillin", "both"), default="both")
    parser.add_argument("--vdp-seed", type=int, default=20260751)
    parser.add_argument("--penicillin-seed", type=int, default=20260761)
    parser.add_argument("--output", type=Path, default=ROOT / "kkt_collocation/results/two_stage_vs_kkt_only")
    parser.add_argument("--phase", choices=("full", "train", "konly-chunk", "konly-until-end", "konly-failure", "evaluate", "aggregate"), default="full")
    parser.add_argument("--method", choices=("S", "K-only", "S+K", "all"), default="all")
    parser.add_argument("--test-start", type=int, default=0)
    parser.add_argument("--test-stop", type=int)
    args = parser.parse_args()
    cfg = ExperimentConfig()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase != "full":
        names = (("vdp", args.vdp_seed),) if args.problem == "vdp" else (("penicillin", args.penicillin_seed),)
        if args.problem == "both":
            names = (("vdp", args.vdp_seed), ("penicillin", args.penicillin_seed))
        for name, seed in names:
            problem, output, config = prepare_stage_problem(name, seed, args.output, cfg)
            methods = ("S", "K-only", "S+K") if args.method == "all" else (args.method,)
            if args.phase == "train":
                for method in methods:
                    print(f"[{name} seed {seed}] training {method}", flush=True); stage_train(problem, output, method)
            elif args.phase == "konly-chunk":
                if args.method not in ("K-only", "all"):
                    raise ValueError("konly-chunk requires --method K-only")
                stop = cfg.total_epochs if args.test_stop is None else args.test_stop
                stage_konly_chunk(problem, output, args.test_start, stop)
            elif args.phase == "konly-until-end":
                progress_path = output / "K-only_progress.pth"
                if not progress_path.exists():
                    raise FileNotFoundError("Run the initial K-only chunk before resuming to the end")
                completed = int(torch.load(progress_path, map_location="cpu", weights_only=False)["completed_epochs"])
                while completed < cfg.total_epochs:
                    next_stop = min(completed + 4, cfg.total_epochs)
                    stage_konly_chunk(problem, output, completed, next_stop)
                    completed = next_stop
            elif args.phase == "konly-failure":
                mark_konly_failure(problem, output, "FloatingPointError: non-finite K-only loss at epoch 18")
            elif args.phase == "evaluate":
                stop = len(problem.test_states) if args.test_stop is None else args.test_stop
                for method in methods:
                    print(f"[{name} seed {seed}] evaluating {method}: {args.test_start}:{stop}", flush=True)
                    stage_evaluate(problem, output, method, args.test_start, stop)
            else:
                result = aggregate_stage(output, config)
                combined = {}
                for candidate in args.output.glob("*_seed*"):
                    summary_path = candidate / "summary.json"
                    if summary_path.exists():
                        combined[candidate.name] = json.load(summary_path.open(encoding="utf-8"))
                dump_json(args.output / "single_seed_summary.json", {"runs": combined})
                print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
        return
    runs = []
    if args.problem in ("vdp", "both"):
        runs.append(run_problem("vdp", args.vdp_seed, args.output, cfg))
    if args.problem in ("penicillin", "both"):
        runs.append(run_problem("penicillin", args.penicillin_seed, args.output, cfg))
    dump_json(args.output / "single_seed_summary.json", {"runs": runs})
    print(json.dumps({"runs": runs}, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
