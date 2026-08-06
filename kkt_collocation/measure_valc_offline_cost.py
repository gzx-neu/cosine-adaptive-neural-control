"""Measure the offline cost of the frozen VALC training protocol.

This script deliberately reuses the final label sets and the exact selected
training branches.  It does not run any HDS test evaluation.  The resulting
break-even count is an implementation-specific amortization estimate, not a
hardware-independent complexity claim.
"""
from __future__ import annotations

import copy
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kkt_collocation.run_vdp_ablation import AblationConfig, load_data as load_vdp, train_policy as train_vdp
from kkt_collocation.run_penicillin_true_kkt_ablation import load_true_kkt_data
from kkt_collocation.run_penicillin_ablation import Config as PenicillinConfig, train as train_penicillin
from kkt_collocation.run_cstr_full_simulation import CSTRConfig, train_branch


def sample_sd(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"mean_seconds": float(a.mean()), "sample_std_seconds": float(a.std(ddof=1))}


def main() -> None:
    out = ROOT / "kkt_collocation" / "results" / "eai_extension"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict[str, object] = {"device": str(device), "protocol": {}}

    # VDP: the gate retained S for all three final training seeds.
    data_path = ROOT / "kkt_collocation" / "data" / "vdp_kkt_30x30_warm_s10_margin1e3.pkl"
    initial, controls, objectives, duals, _ = load_vdp(data_path, device)
    vdp_times = []
    for seed in (20260751, 20260752, 20260753):
        cfg = AblationConfig(epochs=200, seed=seed, lambda_grid_size=31)
        start = time.perf_counter(); train_vdp(initial, controls, objectives, duals, False, cfg, device); vdp_times.append(time.perf_counter() - start)
    with data_path.open("rb") as handle:
        vdp_labels = pickle.load(handle)
    report["protocol"]["VDP"] = {
        "selected_branch": "supervised", "label_count": int(len(vdp_labels["initial_state"])),
        "label_generation_seconds": float(np.asarray(vdp_labels["solve_seconds"], float).sum()),
        "training": sample_sd(vdp_times),
    }

    # Penicillin: the gate selected the true-KKT refinement for every seed.
    data_path = ROOT / "kkt_collocation" / "data" / "penicillin_kkt_400_true_duals.pkl"
    x2, controls, objectives, duals, _ = load_true_kkt_data(data_path)
    pen_times = []
    for seed in (20260761, 20260762, 20260763):
        cfg = PenicillinConfig(epochs=200, kkt_epochs=20, kkt_weight=1e-2, substeps=80, grid_size=31, rollout_weight=0.0, seed=seed)
        start = time.perf_counter()
        supervised, mean, std = train_penicillin(x2, controls, objectives, duals, False, cfg, device)
        normalized = torch.tensor(((x2 - mean) / std)[:, None], dtype=torch.float32, device=device)
        with torch.no_grad():
            _, anchor = supervised(normalized)
        train_penicillin(x2, controls, objectives, duals, True, cfg, device, model=copy.deepcopy(supervised),
                         epochs=20, learning_rate=cfg.kkt_learning_rate, anchor_controls=anchor.cpu().numpy())
        pen_times.append(time.perf_counter() - start)
    with data_path.open("rb") as handle:
        pen_labels = pickle.load(handle)
    report["protocol"]["Penicillin"] = {
        "selected_branch": "supervised plus true-KKT refinement", "label_count": int(len(pen_labels["initial_state"])),
        "label_generation_seconds": float(np.asarray(pen_labels["solve_seconds"], float).sum()),
        "training": sample_sd(pen_times),
    }

    # CSTR: retain the frozen seed-specific branch decision (S, S+KKT, S).
    labels_path = ROOT / "kkt_collocation" / "results" / "cstr_full_simulation_900" / "cstr_labels.pkl"
    with labels_path.open("rb") as handle:
        cstr_labels = pickle.load(handle)
    cfg = CSTRConfig(**cstr_labels["config"])
    cstr_times = []
    selected = {20260718: False, 20260725: True, 20260726: False}
    for seed, use_kkt in selected.items():
        start = time.perf_counter()
        supervised, _, _, _ = train_branch(cstr_labels, cfg, kkt=False, epochs=300, seed=seed)
        if use_kkt:
            train_branch(cstr_labels, cfg, kkt=True, epochs=30, seed=seed + 500, base=copy.deepcopy(supervised))
        cstr_times.append(time.perf_counter() - start)
    report["protocol"]["CSTR"] = {
        "selected_branch_per_seed": {str(k): "KKT-refined" if v else "supervised" for k, v in selected.items()},
        "label_count": int(len(cstr_labels["initial_state"])),
        "label_generation_seconds": float(np.asarray(cstr_labels["solve_seconds"], float).sum()),
        "training": sample_sd(cstr_times),
    }

    # Matched online costs come from the already frozen strict-baseline study.
    online = {"VDP": (0.348, 0.115), "Penicillin": (6.367, 0.156), "CSTR": (0.460, 0.099)}
    for name, (baseline, valc) in online.items():
        item = report["protocol"][name]
        total = item["label_generation_seconds"] + item["training"]["mean_seconds"]
        item["matched_baseline_seconds"] = baseline
        item["valc_seconds"] = valc
        item["estimated_break_even_instances"] = int(np.ceil(total / (baseline - valc)))

    path = out / "offline_cost_break_even.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
