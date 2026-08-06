"""Independent Rayleigh S versus S+K single-seed experiment on RK10 labels."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
# The existing Rayleigh pilot requires this on Windows when CasADi/SciPy and
# PyTorch are imported in spawned reference-audit workers.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from torch import nn

from generate_rayleigh_reduced_kkt_data import (RayleighConfig, ReducedRayleigh,
                                                 rayleigh_ode, segment_peak)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    seed: int = 20260771
    supervised_epochs: int = 200
    continuation_epochs: int = 20
    supervised_lr: float = 1e-3
    continuation_lr: float = 1e-5
    kkt_weight: float = 1e-2
    augmented_penalty: float = 10.0
    anchor_weight: float = 1.0
    rollout_consistency_weight: float = 0.0
    lambda_grid_size: int = 31
    validation_count: int = 60
    test_count: int = 100
    hds_workers: int = 4
    serial_evaluation: bool = False

    @property
    def total_epochs(self):
        return self.supervised_epochs + self.continuation_epochs


class PolicyValue(nn.Module):
    def __init__(self, cfg: RayleighConfig):
        super().__init__()
        self.cfg = cfg
        self.body = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(),
                                  nn.Linear(256, 128), nn.ReLU())
        self.value, self.control = nn.Linear(128, 1), nn.Linear(128, cfg.zoh_steps)

    def forward(self, inputs):
        z = self.body(inputs)
        low, high = self.cfg.control_bounds
        return self.value(z), low + (high - low) * torch.sigmoid(self.control(z))


def json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_labels(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["index"]))
    if len(rows) != 900 or not all(row.get("success") for row in rows):
        raise ValueError("Expected 900 successful Rayleigh labels")
    states = np.asarray([row["initial_state"][:2] for row in rows], float)
    controls = np.asarray([row["controls"] for row in rows], float)
    objective = np.asarray([row["objective"] for row in rows], float)
    path_duals = np.asarray([row["path_duals"] for row in rows], float)
    bound_duals = np.asarray([row["bound_duals"] for row in rows], float)
    residual = np.asarray([row["kkt_stationarity_norm"] for row in rows], float)
    n = controls.shape[1]
    if controls.shape != (900, n) or path_duals.shape != (900, n * 11) or bound_duals.shape != (900, 2, n):
        raise ValueError(f"Unexpected label shapes: {controls.shape}, {path_duals.shape}, {bound_duals.shape}")
    return states, controls, objective, path_duals, bound_duals, residual


def rollout(initial, controls, cfg):
    """Exact reduced RK10 transcription including t_k+ path nodes."""
    state = torch.cat((initial, torch.zeros((len(initial), 1), dtype=initial.dtype, device=initial.device)), dim=1)
    g, h = [], cfg.dt
    for k in range(cfg.zoh_steps):
        u = controls[:, k]
        g.append(u + state[:, 0] / 6.0 + cfg.node_margin)

        def rhs(x):
            x1, x2, _ = x[:, 0], x[:, 1], x[:, 2]
            return torch.stack((x2,
                                4.0 * u - x1 - x2 * (7.0 * x2.square() / 50.0 - 7.0 / 5.0),
                                x1.square() + u.square()), dim=1)
        for _ in range(cfg.substeps_per_zoh):
            k1 = rhs(state); k2 = rhs(state + .5 * h * k1)
            k3 = rhs(state + .5 * h * k2); k4 = rhs(state + h * k3)
            state = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            g.append(u + state[:, 0] / 6.0 + cfg.node_margin)
    return state[:, 2], torch.stack(g, dim=1)


def kkt_terms(initial, controls, path_duals, bound_duals, cfg, rho):
    objective, g = rollout(initial, controls, cfg)
    low, high = cfg.control_bounds
    mu_lo, mu_hi = bound_duals[:, 0], bound_duals[:, 1]
    violation = torch.relu(g)
    lagrangian = (objective + (path_duals * g).sum(1)
                  + (mu_lo * (low - controls)).sum(1) + (mu_hi * (controls - high)).sum(1)
                  + .5 * rho * violation.square().sum(1))
    gradient = torch.autograd.grad(lagrangian.sum(), controls, create_graph=True)[0]
    stationarity = gradient.square().mean()
    primal = violation.square().mean()
    comp_path = (path_duals * g).square().mean()
    comp_bounds = ((mu_lo * (low - controls)).square().mean()
                   + (mu_hi * (controls - high)).square().mean())
    return {"total": stationarity + primal + comp_path + comp_bounds,
            "stationarity": stationarity, "primal": primal,
            "complementarity_path": comp_path, "complementarity_bounds": comp_bounds}


def lhs(cfg, count, seed):
    rng = np.random.default_rng(seed)
    x1 = cfg.x1_initial_range[0] + (rng.permutation(count) + rng.random(count)) / count * (cfg.x1_initial_range[1] - cfg.x1_initial_range[0])
    x2 = cfg.x2_initial_range[0] + (rng.permutation(count) + rng.random(count)) / count * (cfg.x2_initial_range[1] - cfg.x2_initial_range[0])
    return np.c_[x1, x2]


def trajectory_objective(state0, controls, cfg):
    from scipy.integrate import solve_ivp
    state = np.array([state0[0], state0[1], 0.0], float)
    for u in controls:
        sol = solve_ivp(lambda t, x: rayleigh_ode(t, x, float(u)), (0, cfg.zoh_duration), state,
                        method="DOP853", rtol=1e-10, atol=1e-12)
        if not sol.success: raise RuntimeError(sol.message)
        state = sol.y[:, -1]
    return float(state[2])


def lambda_candidates(control, cfg, grid):
    low, high = cfg.control_bounds
    normalized = np.clip((float(control) - low) / (high - low), 0.0, 1.0)
    lmax = 1.0 / normalized if normalized > 1e-10 else 1.0
    values = np.unique(np.r_[np.linspace(0.0, lmax, grid), 1.0])
    return [(v, low + np.clip(v * normalized, 0.0, 1.0) * (high - low))
            for v in sorted(values[(values >= 0) & (values <= lmax + 1e-12)], key=lambda v: (abs(v - 1.0), v))]


def hds_correct(state0, controls, cfg, grid):
    initial = np.array([float(state0[0]), float(state0[1]), 0.0])
    state, out = initial.copy(), np.asarray(controls, float).copy()
    raw_peak, changed, lambdas = -np.inf, 0, []
    for k in range(cfg.zoh_steps):
        peak, nxt = segment_peak(state, float(out[k]), cfg); raw_peak = max(raw_peak, peak)
        if peak <= 1e-8:
            state = nxt; lambdas.append(1.0); continue
        found = None
        for lam, candidate in lambda_candidates(out[k], cfg, grid):
            p, candidate_next = segment_peak(state, candidate, cfg)
            if p <= 1e-8:
                found = (lam, candidate, candidate_next); break
        if found is None:
            return {"accepted": False, "controls": out, "raw_peak": raw_peak, "final_peak": np.nan, "corrected_segments": changed, "lambdas": lambdas}
        lam, out[k], state = found; changed += 1; lambdas.append(float(lam))
    check, final_peak = initial.copy(), -np.inf
    for u in out:
        p, check = segment_peak(check, float(u), cfg); final_peak = max(final_peak, p)
    return {"accepted": final_peak <= 1e-8, "controls": out, "raw_peak": raw_peak, "final_peak": final_peak, "corrected_segments": changed, "lambdas": lambdas}


def _ref_worker(payload):
    index, state, config = payload; cfg = RayleighConfig(**config); started = time.perf_counter()
    try:
        result = ReducedRayleigh(cfg).solve(np.asarray(state, float))
        # A matched cold reference is defined by the same reduced-space RK10
        # discrete NLP. Event-located continuous-time checking is optional
        # diagnostic evidence and must not reject a successful discrete solve.
        return {"index": index, "initial_state": state, "solver_success": True, "solve_seconds": result["solve_seconds"],
                "objective": trajectory_objective(np.asarray(state, float), np.asarray(result["controls"], float), cfg),
                "audit_accepted": None, "audit_max_g": None,
                "kkt_stationarity_norm": result["kkt_stationarity_norm"], "worker_seconds": time.perf_counter() - started}
    except Exception as exc:
        return {"index": index, "initial_state": state, "solver_success": False, "audit_accepted": False,
                "solve_seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}"}


def _audit_worker(payload):
    index, state, controls, config, grid = payload; cfg = RayleighConfig(**config); started = time.perf_counter()
    try:
        raw = np.asarray(controls, float); state0 = np.asarray(state, float)
        # Evaluate the nominal sequence separately, not after earlier corrections.
        s, nominal_peak = np.array([state0[0], state0[1], 0.0]), -np.inf
        for u in raw:
            p, s = segment_peak(s, float(u), cfg); nominal_peak = max(nominal_peak, p)
        nominal_obj = trajectory_objective(state0, raw, cfg)
        corrected = hds_correct(state0, raw, cfg, grid)
        applied = trajectory_objective(state0, corrected["controls"], cfg) if corrected["accepted"] else np.nan
        return {"index": index, "initial_state": state, "accepted": bool(corrected["accepted"]), "fallback": not bool(corrected["accepted"]),
                "nominal_max_g": nominal_peak, "final_max_g": corrected["final_peak"], "nominal_objective": nominal_obj,
                "hds_objective": applied, "corrected_segments": int(corrected["corrected_segments"]),
                "hds_seconds": time.perf_counter() - started}
    except Exception as exc:
        return {"index": index, "initial_state": state, "accepted": False, "fallback": True, "error": f"{type(exc).__name__}: {exc}", "hds_seconds": time.perf_counter() - started}


def finite_mean(values):
    x = np.asarray(values, float); x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def evaluate(name, model, mean, std, states, refs, cfg, exp, output):
    device = next(model.parameters()).device
    inputs = torch.tensor((states - mean) / std, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(10): model(inputs)
        begun = time.perf_counter(); _, predicted = model(inputs)
        inference = (time.perf_counter() - begun) / len(states)
    controls = predicted.cpu().numpy()
    payload = [(i, states[i].tolist(), controls[i].tolist(), asdict(cfg), exp.lambda_grid_size) for i in range(len(states))]
    if exp.serial_evaluation:
        rows = []
        for count, task in enumerate(payload, start=1):
            rows.append(_audit_worker(task))
            if count % 10 == 0 or count == len(payload):
                print(f"{name} HDS {count}/{len(payload)}", flush=True)
    else:
        # CVP60 cold-start solves can retain substantial native solver state on
        # Windows. Isolate every audit task so a long evaluation cannot exhaust
        # a reused spawned worker; this changes process lifetime, not numerics.
        with mp.get_context("spawn").Pool(exp.hds_workers, maxtasksperchild=1) as pool:
            rows = list(pool.imap_unordered(_audit_worker, payload, chunksize=1))
    rows.sort(key=lambda row: row["index"]); ref_map = {r["index"]: r for r in refs}
    for row in rows:
        ref = ref_map[row["index"]]; row["reference_objective"] = ref.get("objective")
        # The direct-transcription reference is retained when its discrete NLP
        # solve succeeds. Its event-located audit is a diagnostic, not a reason
        # to discard an otherwise valid discrete reduced-space reference.
        reference_ok = bool(ref.get("solver_success")); row["reference_available"] = reference_ok
        denominator = max(abs(ref["objective"]), 1e-12) if reference_ok else np.nan
        row["relative_nominal_gap_percent"] = 100 * (row["nominal_objective"] - ref["objective"]) / denominator if reference_ok else np.nan
        row["relative_hds_gap_percent"] = 100 * (row["hds_objective"] - ref["objective"]) / denominator if reference_ok and row["accepted"] else np.nan
        row["inference_seconds"] = inference; row["total_predeployment_seconds"] = inference + row["hds_seconds"]
    fields = sorted({key for row in rows for key in row})
    with (output / f"{name}_test_sample_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    hds_gaps = [r.get("relative_hds_gap_percent", np.nan) for r in rows]
    accepted = np.asarray([r["accepted"] for r in rows], bool)
    return {"accepted_network_samples": int(accepted.sum()), "fallback_samples": int((~accepted).sum()),
            "nominal_violation_rate_percent": float(100 * np.mean(np.asarray([r.get("nominal_max_g", np.nan) for r in rows]) > 1e-8)),
            "nominal_max_g": float(np.nanmax([r.get("nominal_max_g", np.nan) for r in rows])),
            "final_max_g": finite_mean([r.get("final_max_g", np.nan) for r in rows]),
            "mean_nominal_relative_gap_percent": finite_mean([r.get("relative_nominal_gap_percent", np.nan) for r in rows]),
            "mean_hds_relative_gap_percent": finite_mean(hds_gaps),
            "std_hds_relative_gap_percent": float(np.nanstd(hds_gaps)), "median_hds_relative_gap_percent": float(np.nanmedian(hds_gaps)),
            "p95_hds_relative_gap_percent": float(np.nanpercentile(hds_gaps, 95)),
            "mean_corrected_segments": finite_mean([r.get("corrected_segments", np.nan) for r in rows]),
            "corrected_trajectory_rate_percent": float(100 * np.mean(np.asarray([r.get("corrected_segments", 0) for r in rows]) > 0)),
            "corrected_control_segment_rate_percent": float(100 * np.sum([r.get("corrected_segments", 0) for r in rows]) / (cfg.zoh_steps * len(rows))),
            "mean_hds_objective_change": finite_mean([r.get("hds_objective", np.nan) - r.get("nominal_objective", np.nan) for r in rows]),
            "mean_inference_seconds": inference, "mean_hds_seconds": finite_mean([r["hds_seconds"] for r in rows]),
            "mean_total_predeployment_seconds": finite_mean([r["total_predeployment_seconds"] for r in rows])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=ROOT / "kkt_collocation/results/rayleigh_reduced_kkt_n20_rk10_30x30_margin1e5/records.jsonl")
    ap.add_argument("--output", type=Path, default=ROOT / "kkt_collocation/results/rayleigh_two_stage_vs_s_single")
    ap.add_argument("--seed", type=int, default=20260771,
                    help="Training/initialization seed. Does not change the frozen evaluation split.")
    ap.add_argument("--split-seed", type=int, default=20260771,
                    help="Seed defining the frozen validation and test initial conditions.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hds-workers", type=int, default=4)
    ap.add_argument("--anchor-weight", type=float, default=1.0,
                    help="Continuation anchor coefficient; fixed for both the saved config and training run.")
    ap.add_argument("--control-only-supervision", action="store_true",
                    help="Use only normalized control MSE in supervised stages; objective labels remain unused.")
    ap.add_argument("--unified-short-continuation", action="store_true",
                    help="Use the exploratory S-u=210 versus S-u+K=200+10 protocol (alpha=1e-2, anchor=1).")
    ap.add_argument("--serial-evaluation", action="store_true",
                    help="Run cold references and HDS audits in the parent process with progress logs.")
    ap.add_argument("--train-only", action="store_true",
                    help="Save the S and S+K training ablation without cold-reference or HDS evaluation.")
    ap.add_argument("--reference-only", action="store_true",
                    help="Generate only matched reduced-space cold-start references for frozen test states.")
    ap.add_argument("--test-states", type=Path,
                    help="Optional frozen .npy test-state file; required when referencing an existing training run.")
    ap.add_argument("--evaluate-checkpoints", type=Path,
                    help="Directory containing saved S.pth and S+K.pth checkpoints to evaluate without retraining.")
    ap.add_argument("--references", type=Path,
                    help="Matched cold-start reference JSON generated by --reference-only.")
    args = ap.parse_args()
    if args.output.exists(): raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    if args.hds_workers < 1:
        raise ValueError("--hds-workers must be positive")
    exp = (Config(seed=args.seed, supervised_epochs=2, continuation_epochs=1, validation_count=4,
                  test_count=4, hds_workers=min(2, args.hds_workers), anchor_weight=args.anchor_weight,
                  serial_evaluation=args.serial_evaluation) if args.smoke
           else Config(seed=args.seed, hds_workers=args.hds_workers, anchor_weight=args.anchor_weight,
                       serial_evaluation=args.serial_evaluation))
    if args.unified_short_continuation:
        exp = Config(seed=args.seed, hds_workers=args.hds_workers, serial_evaluation=args.serial_evaluation,
                     supervised_epochs=200, continuation_epochs=10, kkt_weight=1e-2, anchor_weight=1.0)
    set_seed(exp.seed)
    labels = args.labels.resolve(); summary = json.loads((labels.parent / "summary.json").read_text(encoding="utf-8"))
    cdict = dict(summary["config"])
    for k in ("control_bounds", "x1_initial_range", "x2_initial_range"): cdict[k] = tuple(cdict[k])
    cfg = RayleighConfig(**cdict)
    states, uref, jref, path, bounds, recorded = load_labels(labels)
    validation = lhs(cfg, exp.validation_count, args.split_seed + 1)
    test = (np.asarray(np.load(args.test_states), float) if args.test_states else
            lhs(cfg, exp.test_count, args.split_seed + 2))
    if test.shape != (exp.test_count, 2):
        raise ValueError(f"Expected frozen test states of shape {(exp.test_count, 2)}, got {test.shape}")
    np.save(args.output / "validation_initial_conditions.npy", validation); np.save(args.output / "test_initial_conditions.npy", test)
    # Exact full-KKT teacher audit in float64.
    x64 = torch.tensor(states, dtype=torch.float64); u64 = torch.tensor(uref, dtype=torch.float64, requires_grad=True)
    teacher = kkt_terms(x64, u64, torch.tensor(path, dtype=torch.float64), torch.tensor(bounds, dtype=torch.float64), cfg, exp.augmented_penalty)
    rms = float(torch.sqrt(teacher["stationarity"]).detach())
    if not np.isfinite(rms) or rms > 1e-3: raise RuntimeError(f"Teacher self-check failed: {rms:.3e}")
    teacher_summary = {"recorded_stationarity_norm_mean": float(recorded.mean()), "recorded_stationarity_norm_max": float(recorded.max()),
                       "torch_rms_stationarity": rms, "torch_total_kkt_residual": float(teacher["total"].detach()),
                       "interpretation": "finite-dimensional reduced-RK10 transcription KKT quantities; not continuous-time multipliers"}
    dump(args.output / "teacher_kkt_self_check.json", teacher_summary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, ur, jr = (torch.tensor(states, dtype=torch.float32, device=device), torch.tensor(uref, dtype=torch.float32, device=device), torch.tensor(jref[:, None], dtype=torch.float32, device=device))
    pr, br = torch.tensor(path, dtype=torch.float32, device=device), torch.tensor(bounds, dtype=torch.float32, device=device)
    mean, std = states.mean(0), states.std(0).clip(1e-6); nx = torch.tensor((states - mean) / std, dtype=torch.float32, device=device)
    jmean, jstd = jr.mean(), jr.std().clamp_min(1e-6); low, high = cfg.control_bounds; span = high - low

    def terms(model, need_kkt, anchor=None):
        predj, predu = model(nx); control = nn.functional.mse_loss((predu - low) / span, (ur - low) / span)
        objective = nn.functional.mse_loss(predj, (jr - jmean) / jstd)
        out = {"control_mse": control, "objective_mse": objective,
               "supervised": control if args.control_only_supervision else control + .1 * objective}
        if need_kkt: out.update({"kkt_" + k: v for k, v in kkt_terms(x, predu, pr, br, cfg, exp.augmented_penalty).items()})
        if anchor is not None: out["anchor"] = nn.functional.mse_loss((predu - low) / span, (anchor - low) / span)
        return out

    def train(method):
        set_seed(exp.seed); model = PolicyValue(cfg).to(device); base = copy.deepcopy(model.state_dict()); model.load_state_dict(base)
        stages = [("supervised", exp.total_epochs, False, None, exp.supervised_lr)] if method == "S" else [("supervised", exp.supervised_epochs, False, None, exp.supervised_lr)]
        history, failure, started = [], None, time.perf_counter()
        for stage, epochs, need_kkt, anchor, lr in stages:
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for epoch in range(1, epochs + 1):
                try:
                    record = terms(model, need_kkt, anchor); loss = record["supervised"]
                    if need_kkt:
                        min_scale = torch.finfo(record["kkt_total"].dtype).tiny if args.control_only_supervision else 1.0
                        loss = loss + exp.kkt_weight * record["kkt_total"] / record["kkt_total"].detach().clamp_min(min_scale)
                    if anchor is not None: loss = loss + exp.anchor_weight * record["anchor"]
                    if not torch.isfinite(loss): raise FloatingPointError("non-finite loss")
                    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                    history.append({"stage": stage, "epoch": epoch, "loss": float(loss.detach()), **{k: float(v.detach()) for k, v in record.items()}})
                    if epoch == 1 or epoch % 20 == 0 or epoch == epochs: print(f"{method} {stage} {epoch}/{epochs} loss={float(loss.detach()):.3e}", flush=True)
                except (RuntimeError, FloatingPointError) as exc: failure = f"{type(exc).__name__}: {exc}"; break
            if failure: break
            if method == "S+K" and stage == "supervised":
                with torch.no_grad(): _, frozen = model(nx)
                stages.append(("continuation", exp.continuation_epochs, True, frozen.detach(), exp.continuation_lr))
        final = terms(model, True)
        rec = {"completed": failure is None, "failure": failure, "seconds": time.perf_counter() - started,
               "control_mse_normalized": float(final["control_mse"].detach()), "objective_mse_normalized": float(final["objective_mse"].detach()),
               "kkt_residual": float(final["kkt_total"].detach()), "kkt_stationarity": float(final["kkt_stationarity"].detach()),
               "kkt_primal": float(final["kkt_primal"].detach()), "kkt_complementarity_path": float(final["kkt_complementarity_path"].detach()), "kkt_complementarity_bounds": float(final["kkt_complementarity_bounds"].detach())}
        torch.save({"model": model.state_dict(), "state_mean": mean, "state_std": std, "training": rec, "config": asdict(exp)}, args.output / f"{method}.pth")
        dump(args.output / f"{method}_training_log.json", {"training": rec, "history": history}); return model, rec

    if args.train_only:
        methods = {}
        for method in ("S", "S+K"):
            print(f"Training {method}", flush=True)
            _, training = train(method)
            methods[method] = {"training": training}
        final = {"status": "completed_training_only", "config": asdict(exp), "split_seed": args.split_seed,
                 "label_source": str(labels), "teacher_kkt_self_check": teacher_summary, "methods": methods,
                 "deferred_evaluation": "Cold-start NLP references and HDS/lambda evaluation were intentionally deferred."}
        dump(args.output / "summary.json", final)
        rows = ["| Method | Training stable | Control MSE | Objective MSE | Full KKT residual |",
                "|---|---|---:|---:|---:|"]
        for name, result in methods.items():
            tr = result["training"]
            rows.append(f"| {name} | {tr['completed']} | {tr['control_mse_normalized']:.3e} | {tr['objective_mse_normalized']:.3e} | {tr['kkt_residual']:.3e} |")
        (args.output / "summary_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(json.dumps(final, indent=2, default=json_default), flush=True)
        return

    if args.reference_only:
        payload = [(i, test[i].tolist(), asdict(cfg)) for i in range(len(test))]
        refs = []
        # Serial mode is deliberately available for high-CVP cold starts,
        # avoiding Windows native-solver worker accumulation while checkpoint
        # progress remains visible in the terminal.
        if exp.serial_evaluation:
            for count, task in enumerate(payload, start=1):
                refs.append(_ref_worker(task))
                if count % 10 == 0 or count == len(payload):
                    dump(args.output / "cold_start_references_partial.json", refs)
                    print(f"Cold references {count}/{len(payload)}", flush=True)
        else:
            context = mp.get_context("spawn")
            with context.Pool(exp.hds_workers, maxtasksperchild=1) as pool:
                refs = list(pool.imap_unordered(_ref_worker, payload, chunksize=1))
        refs.sort(key=lambda row: row["index"])
        dump(args.output / "cold_start_references.json", refs)
        successful = [row for row in refs if row.get("solver_success")]
        final = {"status": "completed_reference_only", "config": asdict(exp), "split_seed": args.split_seed,
                 "label_source": str(labels), "test_states": str(args.test_states) if args.test_states else None,
                 "reference": {"count": len(refs), "solver_successful": len(successful),
                               "event_audit_performed": False,
                               "mean_cold_start_seconds": finite_mean([r.get("solve_seconds", np.nan) for r in successful]),
                               "protocol": "zero-control cold start, reduced-space RK10 direct transcription; no warm start; event audit diagnostic only"}}
        dump(args.output / "summary.json", final)
        print(json.dumps(final, indent=2, default=json_default), flush=True)
        return

    if args.evaluate_checkpoints:
        if not args.references:
            raise ValueError("--evaluate-checkpoints requires --references")
        references = json.loads(args.references.read_text(encoding="utf-8"))
        if len(references) != len(test):
            raise ValueError("Reference count does not match the frozen test-state count")
        methods = {}
        for method in ("S", "S+K"):
            checkpoint = torch.load(args.evaluate_checkpoints / f"{method}.pth", map_location=device, weights_only=False)
            model = PolicyValue(cfg).to(device); model.load_state_dict(checkpoint["model"])
            training = checkpoint["training"]
            methods[method] = {"training": training,
                               "deployment": evaluate(method, model, mean, std, test, references, cfg, exp, args.output)}
        ref_ok = [r for r in references if r.get("solver_success")]
        ref_seconds = finite_mean([r.get("solve_seconds", np.nan) for r in ref_ok])
        for result in methods.values():
            deployment = result["deployment"]
            deployment["mean_cold_nlp_seconds"] = ref_seconds
            deployment["speedup_vs_cold_nlp"] = ref_seconds / deployment["mean_total_predeployment_seconds"]
        final = {"status": "completed_evaluation_only", "config": asdict(exp), "split_seed": args.split_seed,
                 "training_protocol": {"control_only_supervision": args.control_only_supervision,
                                       "unified_short_continuation": args.unified_short_continuation,
                                       "objective_labels_used_for_loss": not args.control_only_supervision},
                 "label_source": str(labels), "test_states": str(args.test_states) if args.test_states else None,
                 "reference_source": str(args.references),
                 "reference": {"count": len(references), "solver_successful": len(ref_ok),
                               "mean_cold_start_seconds": ref_seconds,
                               "protocol": "zero-control cold start, reduced-space RK10 direct transcription; no warm start"},
                 "methods": methods,
                 "hds_statement": "Continuous-time numerical audit evidence under the declared model and numerical settings; not a real-system absolute safety guarantee."}
        dump(args.output / "summary.json", final)
        rows = ["| Method | Nominal gap (%) | HDS gap (%) | Nominal violation | Corrected segments | Accepted / fallback | KKT residual | Speedup |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, result in methods.items():
            d, tr = result["deployment"], result["training"]
            rows.append(f"| {name} | {d['mean_nominal_relative_gap_percent']:.3f} | {d['mean_hds_relative_gap_percent']:.3f} | {d['nominal_violation_rate_percent']:.1f}% | {d['mean_corrected_segments']:.2f} | {d['accepted_network_samples']} / {d['fallback_samples']} | {tr['kkt_residual']:.3e} | {d['speedup_vs_cold_nlp']:.2f}x |")
        (args.output / "summary_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(json.dumps(final, indent=2, default=json_default), flush=True)
        return

    context = mp.get_context("spawn"); payload = [(i, test[i].tolist(), asdict(cfg)) for i in range(len(test))]
    if exp.serial_evaluation:
        refs = []
        for count, task in enumerate(payload, start=1):
            refs.append(_ref_worker(task))
            if count % 10 == 0 or count == len(payload):
                print(f"Cold references {count}/{len(payload)}", flush=True)
    else:
        # See the analogous audit pool above: reference NLP solves are isolated
        # task-by-task to avoid native-state accumulation for high-CVP runs.
        with context.Pool(exp.hds_workers, maxtasksperchild=1) as pool:
            refs = list(pool.imap_unordered(_ref_worker, payload, chunksize=1))
    refs.sort(key=lambda row: row["index"]); dump(args.output / "cold_start_references.json", refs)
    methods = {}
    for method in ("S", "S+K"):
        print(f"Training {method}", flush=True); model, training = train(method)
        methods[method] = {"training": training, "deployment": evaluate(method, model, mean, std, test, refs, cfg, exp, args.output)}
    # The reduced-QP cold reference worker records event audit as diagnostic
    # only; availability for an objective comparison is solver success.
    ref_ok = [r for r in refs if r.get("solver_success")]; ref_seconds = finite_mean([r.get("solve_seconds", np.nan) for r in ref_ok])
    for result in methods.values():
        d = result["deployment"]; d["mean_cold_nlp_seconds"] = ref_seconds; d["speedup_vs_cold_nlp"] = ref_seconds / d["mean_total_predeployment_seconds"]
    final = {"status": "completed", "config": asdict(exp), "split_seed": args.split_seed,
             "training_protocol": {"control_only_supervision": args.control_only_supervision,
                                   "unified_short_continuation": args.unified_short_continuation,
                                   "objective_labels_used_for_loss": not args.control_only_supervision},
             "label_source": str(labels), "teacher_kkt_self_check": teacher_summary,
             "reference": {"count": len(refs), "successful_and_audited": len(ref_ok), "mean_cold_start_seconds": ref_seconds}, "methods": methods,
             "hds_statement": "Continuous-time numerical audit evidence under the declared model and numerical settings; not a real-system absolute safety guarantee."}
    dump(args.output / "summary.json", final)
    rows = ["| Method | Nominal gap (%) | HDS gap (%) | Nominal violation | Corrected segments | Accepted / fallback | KKT residual | Speedup |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in methods.items():
        d, tr = result["deployment"], result["training"]
        rows.append(f"| {name} | {d['mean_nominal_relative_gap_percent']:.3f} | {d['mean_hds_relative_gap_percent']:.3f} | {d['nominal_violation_rate_percent']:.1f}% | {d['mean_corrected_segments']:.2f} | {d['accepted_network_samples']} / {d['fallback_samples']} | {tr['kkt_residual']:.3e} | {d['speedup_vs_cold_nlp']:.2f}x |")
    (args.output / "summary_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, default=json_default), flush=True)


if __name__ == "__main__": main()
