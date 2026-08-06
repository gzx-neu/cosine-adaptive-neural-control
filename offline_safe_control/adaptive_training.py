"""Shared policy/value interface and validation-gated fine-tuning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class AdaptiveThresholds:
    """Pre-registered limits used to decide whether fine-tuning is needed."""

    allowed_peak_violation: float
    allowed_violation_rate: float

    def __post_init__(self) -> None:
        if self.allowed_peak_violation < 0:
            raise ValueError("allowed_peak_violation must be non-negative")
        if not 0 <= self.allowed_violation_rate <= 1:
            raise ValueError("allowed_violation_rate must lie in [0, 1]")


@dataclass(frozen=True)
class ValidationAudit:
    peak_violation: float
    mean_violation: float
    violation_rate: float
    sample_count: int
    finetune_required: bool


def audit_violations(
    violations: Sequence[float], thresholds: AdaptiveThresholds
) -> ValidationAudit:
    """Audit non-negative path violations from an independent validation set."""
    values = np.maximum(np.asarray(violations, dtype=float), 0.0)
    if values.size == 0:
        raise ValueError("validation set must contain at least one sample")
    peak = float(values.max())
    rate = float(np.mean(values > thresholds.allowed_peak_violation))
    required = peak > thresholds.allowed_peak_violation or rate > thresholds.allowed_violation_rate
    return ValidationAudit(peak, float(values.mean()), rate, int(values.size), required)


class SharedPolicyValueNetwork(nn.Module):
    """Maps a measured pre-optimization initial state to (J_hat, u_hat)."""

    def __init__(self, state_dim: int, control_steps: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.control_head = nn.Linear(hidden_dim, control_steps)

    def forward(self, initial_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(initial_state)
        return self.value_head(features), self.control_head(features)


def supervised_policy_value_loss(
    predicted_value: torch.Tensor,
    predicted_controls: torch.Tensor,
    target_value: torch.Tensor,
    target_controls: torch.Tensor,
    value_weight: float = 0.1,
    control_weight: float = 1.0,
) -> torch.Tensor:
    """Baseline loss; all targets must already be consistently normalized."""
    mse = nn.functional.mse_loss
    return value_weight * mse(predicted_value, target_value) + control_weight * mse(
        predicted_controls, target_controls
    )
