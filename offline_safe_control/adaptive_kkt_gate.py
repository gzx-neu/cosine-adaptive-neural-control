"""Pre-registered validation gate for optional offline KKT refinement.

The gate is evaluated once after supervised training, using raw nominal
open-loop controls on an independent validation set.  It is deliberately not
evaluated after safety correction: the latter would hide the constraint
generalization error that the gate is meant to diagnose.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AdaptiveKKTThresholds:
    """Dimensionless, pre-registered gate limits.

    ``numerical_violation_tolerance`` separates a real HDS-observed path
    violation from integration noise.  ``rate_normalized_violation`` defines
    which violations are materially large enough to count in the refinement
    rate; it prevents the model-selection gate from reacting to harmless,
    subsequently correctable numerical-scale overshoots.  The strict final
    HDS safety audit remains governed by numerical_violation_tolerance.
    """

    numerical_violation_tolerance: float = 1e-8
    allowed_violation_rate: float = 0.05
    rate_normalized_violation: float = 0.025
    allowed_normalized_peak_violation: float = 0.03
    engineering_constraint_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.numerical_violation_tolerance < 0:
            raise ValueError("numerical_violation_tolerance must be non-negative")
        if not 0.0 <= self.allowed_violation_rate <= 1.0:
            raise ValueError("allowed_violation_rate must be in [0, 1]")
        if self.rate_normalized_violation < 0:
            raise ValueError("rate_normalized_violation must be non-negative")
        if self.allowed_normalized_peak_violation < 0:
            raise ValueError("allowed_normalized_peak_violation must be non-negative")
        if self.engineering_constraint_scale <= 0:
            raise ValueError("engineering_constraint_scale must be positive")


@dataclass(frozen=True)
class AdaptiveKKTGateAudit:
    """Raw-policy validation statistics and the fixed binary decision."""

    sample_count: int
    maximum_violation: float
    mean_violation: float
    rate_violation_threshold: float
    violation_rate: float
    normalized_peak_violation: float
    trigger_by_rate: bool
    trigger_by_peak: bool
    kkt_refinement_required: bool


def audit_raw_hds_peaks(
    hds_peaks: Sequence[float], thresholds: AdaptiveKKTThresholds
) -> AdaptiveKKTGateAudit:
    """Audit raw HDS peaks where path feasibility is ``g(x)<=0``."""
    peaks = np.asarray(hds_peaks, dtype=float)
    if peaks.ndim != 1 or not len(peaks):
        raise ValueError("hds_peaks must be a nonempty one-dimensional sequence")
    if not np.isfinite(peaks).all():
        raise ValueError("hds_peaks contains a non-finite value")
    violations = np.maximum(peaks, 0.0)
    maximum = float(violations.max())
    rate_threshold = thresholds.rate_normalized_violation * thresholds.engineering_constraint_scale
    rate = float(np.mean(violations > rate_threshold))
    normalized = maximum / thresholds.engineering_constraint_scale
    by_rate = rate > thresholds.allowed_violation_rate
    by_peak = normalized > thresholds.allowed_normalized_peak_violation
    return AdaptiveKKTGateAudit(
        sample_count=int(len(peaks)),
        maximum_violation=maximum,
        mean_violation=float(violations.mean()),
        rate_violation_threshold=float(rate_threshold),
        violation_rate=rate,
        normalized_peak_violation=float(normalized),
        trigger_by_rate=bool(by_rate),
        trigger_by_peak=bool(by_peak),
        kkt_refinement_required=bool(by_rate or by_peak),
    )
