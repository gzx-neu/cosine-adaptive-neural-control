"""Independent old-domain CSTR path-only KKT ablation.

This intentionally omits box-bound multipliers from the loss while preserving
the sigmoid control parameterization.  It is a diagnostic ablation matching
the older VDP/penicillin path-only loss structure, not a complete CSTR KKT
system.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing as mp
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

import run_economou_cstr_two_stage_vs_kkt_only as full
from run_economou_cstr_supervised_hds import PolicyValue, lhs_states
from screen_economou_cstr_30x30 import EconomouScreenConfig

ROOT = Path(__file__).resolve().parents[1]


def path_only_terms(initial, flat, path_duals, cfg, rho):
    """VDP/penicillin-style path-only residual for g(u) <= 0.

    The plus signs are the minimization-KKT convention used by the existing
    VDP/penicillin implementation: J + mu^T g + rho/2 ||relu(g)||^2.
    """
    objective, g = full.rollout_flat(initial, flat, cfg)
    mu = torch.clamp(path_duals, min=0.0)
    violation = torch.relu(g)
    augmented = objective + (mu * g).sum(1) + .5 * rho * violation.square().sum(1)
    gradient = torch.autograd.grad(augmented.sum(), flat, create_graph=True)[0]
    stationarity = gradient.square().mean()
    primal = violation.square().mean()
    complementarity = (mu * g).square().mean()
    return {"total": stationarity + primal + complementarity,
            "stationarity": stationarity, "primal": primal,
            "complementarity": complementarity, "objective": objective, "path_g": g}


def add_correction_statistics(summary, log_path, reference_seconds):
    rows = list(csv.DictReader(log_path.open(encoding="utf-8")))
    corrected = np.asarray([float(row.get("corrected_segments", 0)) for row in rows])
    summary["corrected_trajectory_rate_percent"] = float(100 * np.mean(corrected > 0))
    summary["corrected_control_segment_rate_percent"] = float(100 * corrected.sum() / (10 * len(rows)))
    total = summary.get("mean_total_predeployment_seconds")
    summary["mean_cold_nlp_seconds"] = reference_seconds
    summary["speedup_vs_cold_nlp"] = reference_seconds / total if total and total > 0 else None
    return summary


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=full.json_default), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path,
                    default=ROOT / "kkt_collocation/results/economou_cstr_reduced_kkt10_30x30/records.jsonl")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "kkt_collocation/results/economou_cstr_path_only_kkt_single")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--supervised-epochs", type=int, default=200)
    ap.add_argument("--continuation-epochs", type=int, default=20)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    if args.smoke:
        cfg = full.Config(supervised_epochs=2, continuation_epochs=1, validation_count=4, test_count=4, hds_workers=2)
    else:
        if args.supervised_epochs < 1 or args.continuation_epochs < 1:
            raise ValueError("Both training-stage lengths must be positive")
        cfg = full.Config(supervised_epochs=args.supervised_epochs, continuation_epochs=args.continuation_epochs)
    full.set_seed(cfg.seed)

    labels = args.labels.resolve()
    label_summary = json.loads((labels.parent / "summary.json").read_text(encoding="utf-8"))
    cdict = dict(label_summary["config"])
    for key in ("ti_bounds_K", "flow_bounds", "ca_initial_range", "temperature_initial_range_K"):
        cdict[key] = tuple(cdict[key])
    cstr = EconomouScreenConfig(**cdict)
    if cstr.substeps_per_zoh != 10:
        raise ValueError("This path-only driver is fixed to RK10 labels")
    states, u_ref, j_ref, path, bounds, recorded_self = full.load_labels(labels)
    validation = lhs_states(cstr, cfg.validation_count, cfg.seed + 1)
    test = lhs_states(cstr, cfg.test_count, cfg.seed + 2)
    np.save(args.output / "validation_initial_conditions.npy", validation)
    np.save(args.output / "test_initial_conditions.npy", test)

    # Diagnostic only: full teacher KKT must be small; path-only is not
    # expected to be zero when a control box bound is active.
    x64 = torch.tensor(states, dtype=torch.float64)
    flat64 = torch.tensor(u_ref.reshape(len(u_ref), -1), dtype=torch.float64, requires_grad=True)
    path64 = torch.tensor(path, dtype=torch.float64)
    bounds64 = torch.tensor(bounds, dtype=torch.float64)
    full_teacher = full.kkt_terms(x64, flat64, path64, bounds64, cstr, cfg.augmented_penalty)
    full_rms = float(torch.sqrt(full_teacher["stationarity"]).detach())
    if not np.isfinite(full_rms) or full_rms > 1e-3:
        raise RuntimeError(f"Full teacher self-check failed: {full_rms:.3e}")
    flat64_path = torch.tensor(u_ref.reshape(len(u_ref), -1), dtype=torch.float64, requires_grad=True)
    path_teacher = path_only_terms(x64, flat64_path, path64, cstr, cfg.augmented_penalty)
    teacher_summary = {
        "recorded_full_stationarity_norm_mean": float(recorded_self.mean()),
        "recorded_full_stationarity_norm_max": float(recorded_self.max()),
        "recomputed_full_teacher_rms_stationarity": full_rms,
        "recomputed_full_teacher_total": float(full_teacher["total"].detach()),
        "path_only_teacher_rms_stationarity": float(torch.sqrt(path_teacher["stationarity"]).detach()),
        "path_only_teacher_total": float(path_teacher["total"].detach()),
        "interpretation": "path-only residual is an ablation matching the VDP/penicillin loss structure, not the complete CSTR box-constrained KKT system.",
    }
    dump(args.output / "teacher_kkt_self_check.json", teacher_summary)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(states, dtype=torch.float32, device=device)
    ur = torch.tensor(u_ref, dtype=torch.float32, device=device)
    jr = torch.tensor(j_ref[:, None], dtype=torch.float32, device=device)
    pr = torch.tensor(path, dtype=torch.float32, device=device)
    mean = states[:, [0, 2]].mean(0)
    std = states[:, [0, 2]].std(0).clip(1e-6)
    nx = torch.tensor((states[:, [0, 2]] - mean) / std, dtype=torch.float32, device=device)
    jmean, jstd = jr.mean(), jr.std().clamp_min(1e-6)
    low = torch.tensor([cstr.ti_bounds_K[0], cstr.flow_bounds[0]], dtype=torch.float32, device=device)
    span = torch.tensor([cstr.ti_bounds_K[1] - cstr.ti_bounds_K[0],
                         cstr.flow_bounds[1] - cstr.flow_bounds[0]], dtype=torch.float32, device=device)

    def terms(model, need_path, anchor=None):
        pred_j, pred_u = model(nx)
        control = nn.functional.mse_loss((pred_u - low) / span, (ur - low) / span)
        objective = nn.functional.mse_loss(pred_j, (jr - jmean) / jstd)
        result = {"control_mse": control, "objective_mse": objective, "supervised": control + .1 * objective}
        if need_path:
            raw = path_only_terms(x, pred_u.reshape(len(pred_u), -1), pr, cstr, cfg.augmented_penalty)
            result.update({"path_kkt_" + key: value for key, value in raw.items() if key not in ("objective", "path_g")})
        if anchor is not None:
            result["anchor"] = nn.functional.mse_loss((pred_u - low) / span, (anchor - low) / span)
        return result

    def train(method):
        full.set_seed(cfg.seed)
        model = PolicyValue(cstr).to(device)
        initial_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(initial_state)
        history, failure = [], None
        started = time.perf_counter()
        if method == "S":
            stages = [("supervised", cfg.total_epochs, False, None, cfg.supervised_lr)]
        elif method == "K-only-path":
            stages = [("path_kkt_only", cfg.total_epochs, True, None, cfg.supervised_lr)]
        else:
            stages = [("supervised", cfg.supervised_epochs, False, None, cfg.supervised_lr)]
        for stage, epochs, need_path, anchor, lr in stages:
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for epoch in range(1, epochs + 1):
                try:
                    record = terms(model, need_path, anchor)
                    if method == "K-only-path":
                        loss = record["path_kkt_total"] / record["path_kkt_total"].detach().clamp_min(1.0)
                    else:
                        loss = record["supervised"]
                        if need_path:
                            loss = loss + cfg.kkt_weight * record["path_kkt_total"] / record["path_kkt_total"].detach().clamp_min(1.0)
                        if anchor is not None:
                            loss = loss + cfg.anchor_weight * record["anchor"]
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite loss")
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                    history.append({"stage": stage, "epoch": epoch, "loss": float(loss.detach()),
                                    **{key: float(value.detach()) for key, value in record.items()}})
                    if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                        print(f"{method} {stage} {epoch}/{epochs} loss={float(loss.detach()):.3e}", flush=True)
                except (FloatingPointError, RuntimeError) as exc:
                    failure = f"{type(exc).__name__}: {exc}"; break
            if failure:
                break
            if method == "S+K-path" and stage == "supervised":
                with torch.no_grad():
                    _, frozen = model(nx)
                stages.append(("path_continuation", cfg.continuation_epochs, True, frozen.detach(), cfg.continuation_lr))
        final = terms(model, True)
        training = {"completed": failure is None, "failure": failure, "seconds": time.perf_counter() - started,
                    "control_mse_normalized": float(final["control_mse"].detach()),
                    "objective_mse_normalized": float(final["objective_mse"].detach()),
                    "path_only_kkt_residual": float(final["path_kkt_total"].detach()),
                    "path_only_stationarity": float(final["path_kkt_stationarity"].detach()),
                    "path_only_primal": float(final["path_kkt_primal"].detach()),
                    "path_only_complementarity": float(final["path_kkt_complementarity"].detach())}
        torch.save({"model": model.state_dict(), "state_mean": mean, "state_std": std,
                    "method": method, "training": training, "config": asdict(cfg)}, args.output / f"{method}.pth")
        dump(args.output / f"{method}_training_log.json", {"training": training, "history": history})
        return model, training

    # Frozen validation and test references use no labels, policies, or warm starts.
    context = mp.get_context("spawn")
    def references_for(states_to_solve, directory):
        directory.mkdir(parents=True, exist_ok=True)
        payload = [(i, state.tolist(), asdict(cstr)) for i, state in enumerate(states_to_solve)]
        with context.Pool(cfg.hds_workers) as pool:
            refs = list(pool.imap_unordered(full._reference_worker, payload, chunksize=1))
        refs.sort(key=lambda row: row["index"])
        dump(directory / "cold_start_references.json", {"rows": refs})
        return refs
    validation_dir = args.output / "validation"
    validation_refs, test_refs = references_for(validation, validation_dir), references_for(test, args.output)
    ref_seconds = full._mean([row.get("solve_seconds", np.nan) for row in test_refs if row.get("solver_success") and row.get("audit_accepted")])

    methods, validation_metrics = {}, {}
    for method in ("S", "K-only-path", "S+K-path"):
        print(f"Training {method}", flush=True)
        model, training = train(method)
        vsummary = full.evaluate_method(method, model, mean, std, validation, cstr, cfg, validation_dir, validation_refs)
        dsummary = full.evaluate_method(method, model, mean, std, test, cstr, cfg, args.output, test_refs)
        add_correction_statistics(vsummary, validation_dir / f"{method}_test_sample_log.csv",
                                  full._mean([r.get("solve_seconds", np.nan) for r in validation_refs if r.get("solver_success") and r.get("audit_accepted")]))
        add_correction_statistics(dsummary, args.output / f"{method}_test_sample_log.csv", ref_seconds)
        methods[method] = {"training": training, "deployment": dsummary}
        validation_metrics[method] = vsummary
        dump(args.output / "summary.json", {"status": "running", "config": asdict(cfg), "methods": methods})

    # Validation-only deployment decision; K-only is intentionally excluded.
    candidates = [(name, data) for name, data in validation_metrics.items() if name in ("S", "S+K-path") and data["fallback_samples"] == 0]
    selected = min(candidates, key=lambda item: item[1]["mean_hds_relative_gap_percent"])[0] if candidates else None
    existing_full = ROOT / "kkt_collocation/results/economou_cstr_two_stage_vs_kkt_only_rk10_single_formal/summary.json"
    full_hds = None
    if existing_full.exists():
        full_hds = json.loads(existing_full.read_text(encoding="utf-8"))["methods"]["S+K"]["deployment"]["mean_hds_relative_gap_percent"]
    conclusion = {
        "path_only_beats_S_on_test": methods["S+K-path"]["deployment"]["mean_hds_relative_gap_percent"] < methods["S"]["deployment"]["mean_hds_relative_gap_percent"],
        "full_KKT_S_plus_K_hds_gap_percent": full_hds,
        "path_only_S_plus_K_hds_gap_percent": methods["S+K-path"]["deployment"]["mean_hds_relative_gap_percent"],
        "validation_selected_deployment_method": selected,
        "selection_rule": "zero HDS fallback first; then lower validation HDS-corrected relative objective gap between S and S+K-path only.",
        "interpretation": "A path-only result below S would isolate box multipliers as a plausible cause; a result not below S would show that box multipliers alone do not explain the CSTR behavior.",
    }
    ref_ok = [row for row in test_refs if row.get("solver_success") and row.get("audit_accepted")]
    final = {"status": "completed", "config": asdict(cfg), "label_source": str(labels),
             "initial_domain": {"CA": list(cstr.ca_initial_range), "T_K": list(cstr.temperature_initial_range_K)},
             "teacher_self_check": teacher_summary, "validation": validation_metrics, "test": methods,
             "reference": {"count": len(test_refs), "successful_and_audited": len(ref_ok), "mean_cold_start_seconds": ref_seconds},
             "conclusion": conclusion,
             "path_only_statement": "path-only residual is an ablation matching the VDP/penicillin loss structure, not the complete CSTR box-constrained KKT system.",
             "hds_statement": "Continuous-time numerical audit evidence under the declared model and numerical settings; not a real-system absolute safety guarantee."}
    dump(args.output / "summary.json", final)
    lines = ["| Method | Nominal gap (%) | HDS gap (%) | Nominal violation | Corrected trajectories | Corrected segments | Accepted / fallback | Path-only KKT | Speedup |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, result in methods.items():
        d, tr = result["deployment"], result["training"]
        lines.append(f"| {name} | {d['mean_nominal_relative_gap_percent']:.3f} | {d['mean_hds_relative_gap_percent']:.3f} | {d['nominal_violation_rate_percent']:.1f}% | {d['corrected_trajectory_rate_percent']:.1f}% | {d['mean_corrected_segments']:.2f} | {d['accepted_network_samples']} / {d['fallback_samples']} | {tr['path_only_kkt_residual']:.3e} | {d['speedup_vs_cold_nlp']:.2f}x |")
    (args.output / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output / "conclusion.md").write_text(json.dumps(conclusion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, default=full.json_default), flush=True)


if __name__ == "__main__":
    main()
