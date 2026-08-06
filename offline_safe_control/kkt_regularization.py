"""Differentiable reduced-space augmented-Lagrangian KKT residuals."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KKTResidual:
    """Loss terms for a path-constrained reduced-space NLP."""

    stationarity: torch.Tensor
    primal_feasibility: torch.Tensor
    complementarity: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.stationarity + self.primal_feasibility + self.complementarity


def augmented_lagrangian_kkt_residual(
    objective: torch.Tensor,
    controls: torch.Tensor,
    path_constraints: torch.Tensor,
    path_multipliers: torch.Tensor,
    penalty: float,
) -> KKTResidual:
    """Return differentiable KKT residuals for ``g(u)<=0``.

    ``objective`` has shape ``(batch,)``; ``controls`` has shape
    ``(batch, n_controls)``; and both ``path_constraints`` and
    ``path_multipliers`` have shape ``(batch, n_path_nodes)``. The multiplier
    labels come from the direct-transcription NLP. Equality dynamics have
    already been eliminated by differentiable rollout, hence this is the
    reduced-space KKT stationarity residual with respect to the ZOH controls.
    """
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    if objective.ndim != 1 or controls.ndim != 2:
        raise ValueError("objective must be (batch,) and controls must be (batch, controls)")
    if path_constraints.shape != path_multipliers.shape:
        raise ValueError("path constraints and multipliers must have identical shapes")
    if path_constraints.shape[0] != objective.shape[0]:
        raise ValueError("batch dimensions must agree")

    multipliers = torch.clamp(path_multipliers, min=0.0)
    violations = torch.relu(path_constraints)
    augmented = objective + (multipliers * path_constraints).sum(dim=1)
    augmented = augmented + 0.5 * penalty * violations.square().sum(dim=1)
    stationarity_gradient = torch.autograd.grad(
        augmented.sum(), controls, create_graph=True, retain_graph=True
    )[0]
    return KKTResidual(
        stationarity=stationarity_gradient.square().mean(),
        primal_feasibility=violations.square().mean(),
        complementarity=(multipliers * path_constraints).square().mean(),
    )
