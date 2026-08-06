"""Operating-domain checks for pre-execution open-loop deployment.

The learned policy is only an amortized approximation on a declared operating
domain.  This module makes that scope executable: an out-of-domain initial
condition, or an initial condition that is already unsafe, is delegated to the
offline optimizer *before* a policy sequence is released for execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

Array = np.ndarray
Barrier = Callable[[Array], float]


@dataclass(frozen=True)
class BoxOperatingDomain:
    """Closed box domain for the measured components of an initial state.

    ``state_indices`` permits a benchmark to declare a domain only for the
    components that vary between batches.  All unspecified state components
    are deliberately ignored by the domain guard.
    """

    lower: Sequence[float]
    upper: Sequence[float]
    state_indices: Sequence[int] | None = None
    name: str = "operating domain"

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.ndim != 1 or upper.ndim != 1 or lower.shape != upper.shape:
            raise ValueError("lower and upper must be equally sized one-dimensional vectors")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
            raise ValueError("operating-domain bounds must be finite and ordered")
        if self.state_indices is not None:
            indices = tuple(int(index) for index in self.state_indices)
            if len(indices) != len(lower) or len(set(indices)) != len(indices) or min(indices, default=0) < 0:
                raise ValueError("state_indices must be unique non-negative indices matching the bounds")

    @property
    def dimension(self) -> int:
        return len(self.lower)

    def projected_state(self, initial_state: Sequence[float]) -> Array:
        state = np.asarray(initial_state, dtype=float)
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError("initial_state must be a finite one-dimensional vector")
        if self.state_indices is None:
            if state.size != self.dimension:
                raise ValueError("initial_state dimension does not match the operating-domain bounds")
            return state
        if state.size <= max(self.state_indices):
            raise ValueError("initial_state is shorter than a declared state index")
        return state[np.asarray(self.state_indices, dtype=int)]

    def contains(self, initial_state: Sequence[float], *, tolerance: float = 0.0) -> bool:
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        point = self.projected_state(initial_state)
        return bool(np.all(point >= np.asarray(self.lower) - tolerance) and
                    np.all(point <= np.asarray(self.upper) + tolerance))


@dataclass(frozen=True)
class InitialStateAssessment:
    """Deployment decision made before neural-policy inference."""

    in_operating_domain: bool
    initially_safe: bool
    action: str
    reason: str

    @property
    def use_policy(self) -> bool:
        return self.action == "use_policy"


def assess_initial_state(
    initial_state: Sequence[float],
    operating_domain: BoxOperatingDomain,
    barriers: Sequence[Barrier] = (),
    *,
    safety_tolerance: float = 1e-8,
) -> InitialStateAssessment:
    """Return the default deployment action for one measured initial state.

    The domain check intentionally precedes policy inference.  A point outside
    the declared operating domain is not a failed neural correction: it is a
    scoped deployment decision and is therefore sent directly to the chosen
    offline optimizer.  A point already outside a barrier safe set is likewise
    not eligible for the segment-start safety induction.
    """

    if safety_tolerance < 0:
        raise ValueError("safety_tolerance must be non-negative")
    state = np.asarray(initial_state, dtype=float)
    in_domain = operating_domain.contains(state)
    if not in_domain:
        return InitialStateAssessment(False, False, "fallback_offline_optimizer",
                                      f"initial state is outside the declared {operating_domain.name}")
    values = [float(barrier(state)) for barrier in barriers]
    if not np.isfinite(values).all():
        raise ValueError("barrier evaluation returned a non-finite value")
    initially_safe = all(value >= -safety_tolerance for value in values)
    if not initially_safe:
        return InitialStateAssessment(True, False, "fallback_recovery_optimizer",
                                      "initial state is unsafe; segment-start induction is inapplicable")
    return InitialStateAssessment(True, True, "use_policy", "inside operating domain and initially safe")
