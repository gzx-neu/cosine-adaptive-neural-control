"""Sensitivity analysis of the frozen raw-HDS validation gate.

Only the supervised policy and the already independent validation set are
used.  No test point is touched.  The scan pairs the severe-violation-rate
threshold and the normalized maximum-violation threshold over 2--5 percent;
the allowed sample fraction remains fixed at 5 percent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_safe_control.adaptive_kkt_gate import AdaptiveKKTThresholds, audit_raw_hds_peaks
from offline_safe_control.hds_lambda_corrector import HDSLambdaConfig, HDSLambdaCorrector
from kkt_collocation.run_penicillin_ablation import (  # noqa: E402
    DT as PEN_DURATION,
    Policy as PenicillinPolicy,
    UMAX as PEN_UMAX,
    g as pen_constraint,
    gdot as pen_constraint_derivative,
    ode as pen_ode,
)
from kkt_collocation.run_vdp_ablation import (  # noqa: E402
    constraint as vdp_constraint,
    constraint_derivative as vdp_constraint_derivative,
    vdp_ode,
)
from kkt_collocation.train_vdp_kkt_policy import KKTPolicyValueNetwork, TrainConfig


def _vdp_peaks(result_dir: Path) -> tuple[np.ndarray, float]:
    states = np.load(result_dir / "validation_states.npy")
    checkpoint = torch.load(result_dir / "models.pth", map_location="cpu", weights_only=False)
    model = KKTPolicyValueNetwork(TrainConfig()).eval()
    model.load_state_dict(checkpoint["S"])
    mean, std = (np.asarray(value, dtype=np.float32) for value in checkpoint["normalization"]["S"])
    with torch.no_grad():
        _, controls = model(torch.tensor((states[:, :2] - mean) / std, dtype=torch.float32))
    corrector = HDSLambdaCorrector(
        vdp_ode, vdp_constraint, vdp_constraint_derivative, (-0.3, 1.0),
        HDSLambdaConfig(grid_size=31, max_step_fraction=100.0),
    )
    peaks = np.asarray([corrector.audit(state, control, 0.5) for state, control in zip(states, controls.numpy())])
    return peaks, 0.4


def _penicillin_peaks(result_dir: Path) -> tuple[np.ndarray, float]:
    x2 = np.load(result_dir / "validation_x2.npy")
    checkpoint = torch.load(result_dir / "models.pth", map_location="cpu", weights_only=False)
    model = PenicillinPolicy().eval()
    model.load_state_dict(checkpoint["S"])
    mean, std = checkpoint["normalization"]["S"]
    with torch.no_grad():
        _, controls = model(torch.tensor(((x2 - float(mean)) / float(std))[:, None], dtype=torch.float32))
    corrector = HDSLambdaCorrector(
        pen_ode, pen_constraint, pen_constraint_derivative, (0.0, PEN_UMAX),
        HDSLambdaConfig(grid_size=31, max_step_fraction=100.0),
    )
    states = np.column_stack((np.ones(len(x2)), x2, np.full(len(x2), .001), np.full(len(x2), 250.0)))
    peaks = np.asarray([corrector.audit(state, control, PEN_DURATION) for state, control in zip(states, controls.numpy())])
    return peaks, 0.5


def _scan(peaks: np.ndarray, scale: float, fractions: list[float]) -> dict:
    rows = []
    for rate_threshold in fractions:
        for peak_threshold in fractions:
            result = audit_raw_hds_peaks(peaks, AdaptiveKKTThresholds(
                numerical_violation_tolerance=1e-8,
                allowed_violation_rate=0.05,
                rate_normalized_violation=rate_threshold,
                allowed_normalized_peak_violation=peak_threshold,
                engineering_constraint_scale=scale,
            ))
            rows.append({
                "severe_rate_threshold": rate_threshold,
                "peak_threshold": peak_threshold,
                "observed_severe_rate": result.violation_rate,
                "observed_normalized_peak": result.normalized_peak_violation,
                "trigger_by_rate": result.trigger_by_rate,
                "trigger_by_peak": result.trigger_by_peak,
                "trigger": result.kkt_refinement_required,
            })
    triggers = [row["trigger"] for row in rows]
    return {
        "validation_samples": int(len(peaks)),
        "engineering_constraint_scale": scale,
        "maximum_raw_violation": float(np.max(np.maximum(peaks, 0.0))),
        "maximum_normalized_violation": float(np.max(np.maximum(peaks, 0.0)) / scale),
        "threshold_fractions": fractions,
        "rows": rows,
        "all_combinations_trigger": bool(all(triggers)),
        "no_combination_triggers": bool(not any(triggers)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=[.02, .025, .03, .04, .05])
    parser.add_argument("--vdp-result-dir", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "final_vdp_900_adaptive_true_kkt_e200_test400_seed20260731")
    parser.add_argument("--penicillin-result-dir", type=Path,
                        default=ROOT / "kkt_collocation" / "results" / "final_penicillin_true_kkt_e200_w001_test400_seed20260741")
    args = parser.parse_args()
    if any(not 0 < value < 1 for value in args.fractions):
        raise ValueError("threshold fractions must be in (0, 1)")
    vdp_peaks, vdp_scale = _vdp_peaks(args.vdp_result_dir)
    penicillin_peaks, penicillin_scale = _penicillin_peaks(args.penicillin_result_dir)
    report = {
        "protocol": "All 25 paired combinations of severe-rate and peak thresholds are scanned. The allowed severe-sample fraction is fixed at 5%.",
        "VDP": _scan(vdp_peaks, vdp_scale, args.fractions),
        "Penicillin": _scan(penicillin_peaks, penicillin_scale, args.fractions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
