"""Train S, direct K-only, and S+K on all 900 N=100 CSTR labels.

This driver intentionally performs *training only*: it reserves no labels and
does not run HDS or any cold-start NLP reference.  A later, externally fixed
400-point reference set can therefore evaluate all three saved checkpoints
without having been touched during training.
"""
from __future__ import annotations

import copy
import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from run_economou_cstr_supervised_hds import PolicyValue
from run_economou_cstr_two_stage_vs_kkt_only import (
    Config, ROOT, dump_json, kkt_terms, load_labels, set_seed,
)
from screen_economou_cstr_30x30 import EconomouScreenConfig


LABELS = ROOT / "kkt_collocation/results/economou_cstr_reduced_kkt_n100_rk10_margin0_ca030_050_t410_420/records.jsonl"
OUTPUT = ROOT / "kkt_collocation/results/economou_cstr_n100_all900_s_sk_konly_training_seed20260771"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--methods", nargs="+", choices=("S", "S+K", "K-only"), default=("S", "S+K", "K-only"))
    parser.add_argument("--supervised-epochs", type=int, default=200)
    parser.add_argument("--continuation-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260771)
    args = parser.parse_args()
    output_dir = args.output
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite independent directory: {output_dir}")
    output_dir.mkdir(parents=True)
    cfg = Config(seed=args.seed, supervised_epochs=args.supervised_epochs, continuation_epochs=args.continuation_epochs)
    label_summary = json.loads((LABELS.parent / "summary.json").read_text(encoding="utf-8"))
    label_config = dict(label_summary["config"])
    for field in ("ti_bounds_K", "flow_bounds", "ca_initial_range", "temperature_initial_range_K"):
        label_config[field] = tuple(label_config[field])
    cstr = EconomouScreenConfig(**label_config)
    states, u_ref, j_ref, path_duals, bound_duals, label_stationarity = load_labels(LABELS)
    if len(states) != 900 or cstr.zoh_steps != 100 or cstr.substeps_per_zoh != 10:
        raise ValueError("Expected the complete 900-label N=100 RK10 CSTR data set")

    # Confirm that the exact finite-dimensional transcription sees the label
    # controls as KKT-consistent before using their multipliers in any loss.
    x64 = torch.tensor(states, dtype=torch.float64)
    u64 = torch.tensor(u_ref.reshape(900, -1), dtype=torch.float64, requires_grad=True)
    d64 = torch.tensor(path_duals, dtype=torch.float64)
    b64 = torch.tensor(bound_duals, dtype=torch.float64)
    teacher = kkt_terms(x64, u64, d64, b64, cstr, cfg.augmented_penalty)
    teacher_rms = float(torch.sqrt(teacher["stationarity"]).detach())
    if not np.isfinite(teacher_rms) or teacher_rms > 1e-3:
        raise RuntimeError(f"Label KKT self-check failed: RMS stationarity={teacher_rms:.3e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(states, dtype=torch.float32, device=device)
    ur = torch.tensor(u_ref, dtype=torch.float32, device=device)
    jr = torch.tensor(j_ref[:, None], dtype=torch.float32, device=device)
    dr = torch.tensor(path_duals, dtype=torch.float32, device=device)
    br = torch.tensor(bound_duals, dtype=torch.float32, device=device)
    state_mean = states[:, [0, 2]].mean(0)
    state_std = states[:, [0, 2]].std(0).clip(1e-6)
    normalized_x = torch.tensor((states[:, [0, 2]] - state_mean) / state_std, dtype=torch.float32, device=device)
    objective_mean, objective_std = jr.mean(), jr.std().clamp_min(1e-6)
    low = torch.tensor([cstr.ti_bounds_K[0], cstr.flow_bounds[0]], dtype=torch.float32, device=device)
    span = torch.tensor([70.0, 1.0], dtype=torch.float32, device=device)

    def compute(model: PolicyValue, need_kkt: bool, anchor=None):
        prediction_j, prediction_u = model(normalized_x)
        control_mse = nn.functional.mse_loss((prediction_u - low) / span, (ur - low) / span)
        objective_mse = nn.functional.mse_loss(prediction_j, (jr - objective_mean) / objective_std)
        result = {"control_mse": control_mse, "objective_mse": objective_mse,
                  "supervised": control_mse + .1 * objective_mse}
        if need_kkt:
            terms = kkt_terms(x, prediction_u.reshape(len(x), -1), dr, br, cstr, cfg.augmented_penalty)
            result.update({f"kkt_{name}": value for name, value in terms.items()
                           if name not in ("objective", "path_g")})
        if anchor is not None:
            result["anchor"] = nn.functional.mse_loss((prediction_u - low) / span, (anchor - low) / span)
        return result

    # One captured initialization is loaded into every branch.  This is
    # stronger than merely setting the same random seed per branch.
    set_seed(cfg.seed)
    prototype = PolicyValue(cstr).to(device)
    initial_state = copy.deepcopy(prototype.state_dict())

    def run(method: str):
        model = PolicyValue(cstr).to(device)
        model.load_state_dict(initial_state)
        history: list[dict] = []
        failure = None
        started = time.perf_counter()
        if method == "S":
            stages = [("supervised", cfg.total_epochs, False, None, cfg.supervised_lr)]
        elif method == "S+K":
            stages = [("supervised", cfg.supervised_epochs, False, None, cfg.supervised_lr)]
        elif method == "K-only":
            # Direct KKT branch shares the same 200/20 LR calendar.  It has no
            # supervision or anchor in either phase.
            stages = [("kkt_direct", cfg.supervised_epochs, True, None, cfg.supervised_lr),
                      ("kkt_direct_decay", cfg.continuation_epochs, True, None, cfg.continuation_lr)]
        else:
            raise ValueError(method)
        for stage, epochs, need_kkt, anchor, lr in stages:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            for epoch in range(1, epochs + 1):
                try:
                    record = compute(model, need_kkt, anchor)
                    if method == "K-only":
                        loss = record["kkt_total"] / record["kkt_total"].detach().clamp_min(1.0)
                    else:
                        loss = record["supervised"]
                        if need_kkt:
                            loss = loss + cfg.kkt_weight * record["kkt_total"] / record["kkt_total"].detach().clamp_min(1.0)
                        if anchor is not None:
                            loss = loss + cfg.anchor_weight * record["anchor"]
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    history.append({"stage": stage, "epoch": epoch, "loss": float(loss.detach()),
                                    **{key: float(value.detach()) for key, value in record.items()}})
                    if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                        print(f"{method} {stage} {epoch}/{epochs} loss={float(loss.detach()):.3e}", flush=True)
                except (RuntimeError, FloatingPointError) as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                    break
            if failure:
                break
            if method == "S+K" and stage == "supervised":
                with torch.no_grad():
                    _, frozen = model(normalized_x)
                stages.append(("continuation", cfg.continuation_epochs, True, frozen.detach(), cfg.continuation_lr))
        with torch.enable_grad():
            final = compute(model, True)
        training = {"completed": failure is None, "failure": failure,
                    "seconds": time.perf_counter() - started,
                    "control_mse_normalized": float(final["control_mse"].detach()),
                    "objective_mse_normalized": float(final["objective_mse"].detach()),
                    "kkt_residual": float(final["kkt_total"].detach()),
                    "kkt_stationarity": float(final["kkt_stationarity"].detach()),
                    "kkt_primal": float(final["kkt_primal"].detach()),
                    "kkt_complementarity_path": float(final["kkt_complementarity_path"].detach()),
                    "kkt_complementarity_bounds": float(final["kkt_complementarity_bounds"].detach())}
        torch.save({"model": model.state_dict(), "state_mean": state_mean, "state_std": state_std,
                    "config": asdict(cfg), "training": training}, output_dir / f"{method}.pth")
        dump_json(output_dir / f"{method}_training_log.json", {"training": training, "history": history})
        return training

    methods = {}
    for method in args.methods:
        print(f"Training {method}", flush=True)
        methods[method] = run(method)
    output = {
        "status": "completed",
        "protocol": "All 900 labels used for training only; no held-out labels, HDS evaluation, or cold-start references were used.",
        "label_source": str(LABELS),
        "label_protocol": "900 independent cold-start RK10 reduced-space labels; finite-dimensional discretized-NLP multipliers only.",
        "config": asdict(cfg),
        "methods": methods,
        "teacher_kkt_self_check": {"recorded_stationarity_norm_mean": float(label_stationarity.mean()),
                                    "torch_rms_stationarity": teacher_rms,
                                    "torch_total_kkt_residual": float(teacher["total"].detach())},
        "reproducibility": {"seed": cfg.seed, "torch": torch.__version__, "numpy": np.__version__,
                              "platform": platform.platform(), "device": str(device)},
        "later_evaluation_requirement": "Use one externally fixed 400-point cold-start reference set for all three checkpoints; do not use it for model selection."
    }
    dump_json(output_dir / "summary.json", output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
