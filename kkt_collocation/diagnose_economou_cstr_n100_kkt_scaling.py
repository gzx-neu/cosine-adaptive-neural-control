"""Diagnose units and conditioning of the CSTR N=100 finite-dimensional KKT residual.

This is a read-only post-processing diagnostic.  It does not train a model or
solve any new NLP.  In particular, it distinguishes the raw physical-unit
stationarity reported by the training driver from a dimensionless relative
stationarity: residual force divided componentwise by the RMS magnitude of
the objective, path-dual, box-dual, and augmented-penalty forces.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from generate_economou_cstr_reduced_kkt_data import ReducedEconomouCSTR  # noqa: F401
from run_economou_cstr_supervised_hds import PolicyValue
from run_economou_cstr_two_stage_vs_kkt_only import ROOT, load_labels, rollout_flat
from screen_economou_cstr_30x30 import EconomouScreenConfig


RESULT = ROOT / "kkt_collocation/results/economou_cstr_n100_label_holdout_s_sk_single"
LABELS = ROOT / "kkt_collocation/results/economou_cstr_reduced_kkt_n100_rk10_margin0_ca030_050_t410_420/records.jsonl"
OUTPUT = ROOT / "kkt_collocation/results/economou_cstr_n100_label_holdout_kkt_scaling_diagnostic_v2"
RHO = 10.0


def summary(values: torch.Tensor) -> dict[str, float]:
    value = values.detach().cpu().numpy().astype(float)
    return {"mean": float(value.mean()), "std": float(value.std()),
            "median": float(np.median(value)), "p95": float(np.quantile(value, .95)),
            "max": float(value.max())}


def kkt_diagnostic(initial, flat, path_duals, bound_duals, cstr):
    """Return raw and force-normalized stationarity, all per sample."""
    objective, g = rollout_flat(initial, flat, cstr)
    low = torch.tensor([cstr.ti_bounds_K[0], cstr.flow_bounds[0]], dtype=flat.dtype).repeat(cstr.zoh_steps)
    high = torch.tensor([cstr.ti_bounds_K[1], cstr.flow_bounds[1]], dtype=flat.dtype).repeat(cstr.zoh_steps)
    mu_lo, mu_hi = bound_duals[:, 0], bound_duals[:, 1]
    pieces = {
        "objective": objective,
        "path_dual": (path_duals * g).sum(1),
        "bound_dual": (mu_lo * (low - flat)).sum(1) + (mu_hi * (flat - high)).sum(1),
        "augmented_penalty": .5 * RHO * torch.relu(g).square().sum(1),
    }
    gradients = {
        name: torch.autograd.grad(value.sum(), flat, retain_graph=True)[0]
        for name, value in pieces.items()
    }
    lagrangian_gradient = sum(gradients.values())
    # This denominator is a scale of the individual physical KKT forces.  It
    # is nonzero at an exact KKT point, unlike the net stationarity residual.
    force_rms_by_coordinate = torch.sqrt(sum(value.square() for value in gradients.values())).clamp_min(1e-12)
    raw_rms = torch.sqrt(lagrangian_gradient.square().mean(1))
    relative_rms = torch.sqrt((lagrangian_gradient / force_rms_by_coordinate).square().mean(1))
    ti_rms = torch.sqrt(lagrangian_gradient[:, 0::2].square().mean(1))
    flow_rms = torch.sqrt(lagrangian_gradient[:, 1::2].square().mean(1))
    # Gradient with respect to the bounded, unit-interval control coordinate
    # v=(u-low)/(high-low).  It is supplied as a unit diagnostic, not claimed
    # to be a better absolute KKT metric.
    span = torch.tensor([70.0, 1.0], dtype=flat.dtype).repeat(cstr.zoh_steps)
    unit_coordinate_rms = torch.sqrt((lagrangian_gradient * span).square().mean(1))
    return {"raw_rms_stationarity": raw_rms, "relative_force_normalized_rms": relative_rms,
            "temperature_coordinate_raw_rms": ti_rms, "flow_coordinate_raw_rms": flow_rms,
            "unit_control_coordinate_rms": unit_coordinate_rms}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    label_summary = json.loads((LABELS.parent / "summary.json").read_text(encoding="utf-8"))
    config = dict(label_summary["config"])
    for field in ("ti_bounds_K", "flow_bounds", "ca_initial_range", "temperature_initial_range_K"):
        config[field] = tuple(config[field])
    cstr = EconomouScreenConfig(**config)
    states, controls, _objectives, path, bounds, _ = load_labels(LABELS)
    prior = json.loads((RESULT / "summary.json").read_text(encoding="utf-8"))
    indices = np.asarray(prior["split"]["test_label_indices"], dtype=int)
    x = torch.tensor(states[indices], dtype=torch.float64)
    d = torch.tensor(path[indices], dtype=torch.float64)
    b = torch.tensor(bounds[indices], dtype=torch.float64)
    output = {"protocol": "Read-only held-out-label diagnostic; no training and no new cold-start NLP solves.",
              "definition": "relative_force_normalized_rms = RMS(grad L / sqrt(sum_i grad L_i^2)), with i = objective, path-dual, box-dual, augmented penalty.",
              "warning": "The finite-dimensional multipliers are discretized-NLP quantities, not continuous-time multipliers.",
              "methods": {}}
    # Exact held-out teacher confirms the denominator diagnostic is not hiding
    # a faulty label reconstruction.
    teacher_flat = torch.tensor(controls[indices].reshape(len(indices), -1), dtype=torch.float64, requires_grad=True)
    output["teacher"] = {name: summary(value) for name, value in kkt_diagnostic(x, teacher_flat, d, b, cstr).items()}
    for method in ("S", "S+K"):
        checkpoint = torch.load(RESULT / f"{method}.pth", map_location="cpu", weights_only=False)
        model = PolicyValue(cstr).double().eval()
        model.load_state_dict(checkpoint["model"])
        state_mean = torch.as_tensor(checkpoint["state_mean"], dtype=torch.float64)
        state_std = torch.as_tensor(checkpoint["state_std"], dtype=torch.float64)
        normalized_x = (x[:, [0, 2]] - state_mean) / state_std
        with torch.no_grad():
            _predicted_j, control = model(normalized_x)
        flat = control.detach().reshape(len(x), -1).requires_grad_(True)
        output["methods"][method] = {name: summary(value) for name, value in kkt_diagnostic(x, flat, d, b, cstr).items()}
    (OUTPUT / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    rows = ["| Quantity | Teacher | S | S+K |", "|---|---:|---:|---:|"]
    for metric in ("raw_rms_stationarity", "relative_force_normalized_rms", "temperature_coordinate_raw_rms", "flow_coordinate_raw_rms", "unit_control_coordinate_rms"):
        rows.append("| " + metric + " (mean) | " + f"{output['teacher'][metric]['mean']:.6g}"
                    + " | " + f"{output['methods']['S'][metric]['mean']:.6g}" + " | "
                    + f"{output['methods']['S+K'][metric]['mean']:.6g}" + " |")
    (OUTPUT / "summary_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
