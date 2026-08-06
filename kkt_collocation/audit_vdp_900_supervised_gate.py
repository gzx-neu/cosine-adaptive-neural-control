"""Independent HDS gate audit for the original 30x30 VDP supervised model.

This script intentionally evaluates only the original 900-label supervised
policy.  It establishes whether the adaptive gate selects the no-KKT branch
when supervision is sufficiently dense.  A fair Always-KKT comparison still
requires direct-NLP dual labels on this same 30x30 initial-state grid.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.adaptive_kkt_gate import AdaptiveKKTThresholds, audit_raw_hds_peaks  # noqa: E402
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector  # noqa: E402
from run_vdp_ablation import constraint, constraint_derivative, lhs_states, vdp_ode  # noqa: E402


class OriginalVDPNet(nn.Module):
    """Architecture used by ``VDP绘图/纯监督900点.py``."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(2, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.head_J = nn.Linear(256, 1)
        self.head_u = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.head_J(features), self.head_u(features)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "VDP绘图" / "vdp_joint_baseline_800epoch_30x30.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "kkt_collocation" / "results" / "vdp_900_supervised_gate")
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--gate-rate", type=float, default=0.05)
    parser.add_argument("--gate-rate-normalized-violation", type=float, default=0.025)
    parser.add_argument("--gate-normalized-peak", type=float, default=0.03)
    parser.add_argument("--gate-engineering-scale", type=float, default=0.4)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = OriginalVDPNet()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    states = lhs_states(args.validation_samples, args.seed)
    normalized = torch.tensor((states[:, :2] - checkpoint["X_mean"]) / checkpoint["X_std"], dtype=torch.float32)
    with torch.no_grad():
        _, controls = model(normalized)
    controls = controls.numpy()
    if np.any(controls < -0.3 - 1e-8) or np.any(controls > 1.0 + 1e-8):
        raise RuntimeError("The original 900-point policy produced an out-of-bound control")
    corrector = HDSLambdaCorrector(
        vdp_ode, constraint, constraint_derivative, (-0.3, 1.0),
        HDSLambdaConfig(grid_size=31, max_step_fraction=100.0),
    )
    peaks = np.asarray([corrector.audit(state, control, 0.5) for state, control in zip(states, controls)])
    thresholds = AdaptiveKKTThresholds(
        allowed_violation_rate=args.gate_rate,
        rate_normalized_violation=args.gate_rate_normalized_violation,
        allowed_normalized_peak_violation=args.gate_normalized_peak,
        engineering_constraint_scale=args.gate_engineering_scale,
    )
    audit = audit_raw_hds_peaks(peaks, thresholds)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "gate_validation_states.npy", states)
    np.save(args.output / "gate_validation_hds_peaks.npy", peaks)
    with (args.output / "gate_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("index", "y1_0", "y2_0", "raw_hds_max_g"))
        writer.writeheader()
        writer.writerows({"index": i, "y1_0": x[0], "y2_0": x[1], "raw_hds_max_g": peak} for i, (x, peak) in enumerate(zip(states, peaks)))
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "validation_seed": args.seed,
        "thresholds": asdict(thresholds),
        "supervised_raw_validation": asdict(audit),
        "adaptive_selected_branch": "S+KKT" if audit.kkt_refinement_required else "S",
        "control_range": [float(controls.min()), float(controls.max())],
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
