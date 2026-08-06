"""Adaptive-step DOP853 HDS audit with event-located segment extrema.

This module is intentionally separate from the historical corrector.  It
retains the same scalar lambda ray and closest-to-one early-stop rule while
removing the historical ``max_step=duration/N`` restriction: DOP853 chooses
its own adaptive steps under the declared tolerances.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from offline_safe_control.hds_lambda_corrector import SegmentCorrection, SequenceCorrection


Array = np.ndarray
Ode = Callable[[float, Array, float], Array]
Constraint = Callable[[Array], float]
ConstraintDerivative = Callable[[Array, float], float]


@dataclass(frozen=True)
class AdaptiveEventHDSConfig:
    """Declared numerical settings for adaptive continuous-time HDS."""

    grid_size: int = 31
    allow_nonformal_grid: bool = False
    safety_margin: float = 1e-8
    rtol: float = 1e-10
    atol: float = 1e-12
    integrator: str = "DOP853"
    skip_known_unsafe_nominal_candidate: bool = True

    def __post_init__(self) -> None:
        if self.grid_size != 31 and not self.allow_nonformal_grid:
            raise ValueError("The frozen formal protocol requires a 31-point lambda base grid")
        if self.safety_margin < 0 or self.rtol <= 0 or self.atol <= 0:
            raise ValueError("safety margin and ODE tolerances must be positive")
        if self.integrator != "DOP853":
            raise ValueError("The frozen formal protocol requires DOP853")

    @property
    def acceptance_threshold(self) -> float:
        return -float(self.safety_margin)


class AdaptiveEventHDSCorrector:
    """Sequential scalar-ray HDS correction with adaptive event location."""

    def __init__(
        self,
        ode: Ode,
        constraint: Constraint,
        constraint_derivative: ConstraintDerivative,
        control_bounds: tuple[float, float],
        config: AdaptiveEventHDSConfig = AdaptiveEventHDSConfig(),
    ) -> None:
        self.ode = ode
        self.constraint = constraint
        self.constraint_derivative = constraint_derivative
        self.u_min, self.u_max = map(float, control_bounds)
        self.config = config
        if self.u_min > self.u_max:
            raise ValueError("control bounds must be ordered")

    def segment_peak(self, state: Array, control: float, duration: float) -> tuple[float, Array]:
        """Audit one ZOH segment at endpoints and all ``gdot=0`` events."""
        def stationary_event(_time: float, x: Array, *_args: float) -> float:
            return self.constraint_derivative(x, float(control))

        stationary_event.direction = 0
        stationary_event.terminal = False
        solution = solve_ivp(
            self.ode,
            (0.0, duration),
            np.asarray(state, dtype=float),
            args=(float(control),),
            method=self.config.integrator,
            rtol=self.config.rtol,
            atol=self.config.atol,
            dense_output=True,
            events=stationary_event,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        candidates = np.unique(np.r_[0.0, duration, solution.t_events[0]])
        values = np.asarray([self.constraint(solution.sol(time)) for time in candidates], dtype=float)
        return float(values.max()), solution.y[:, -1].copy()

    def audit(self, initial_state: Array, controls: Sequence[float], duration: float) -> float:
        state = np.asarray(initial_state, dtype=float).copy()
        peak = -np.inf
        for control in np.asarray(controls, dtype=float):
            local_peak, state = self.segment_peak(state, float(control), duration)
            peak = max(peak, local_peak)
        return float(peak)

    def _candidate_lambdas(self, nominal_control: float, *, exclude_nominal: bool) -> Array:
        """Build the bounded correction grid used by the archived experiments.

        ``grid_size`` is the number of points in the base linspace. The
        already-audited nominal value ``lambda=1`` is inserted explicitly so
        it is never lost when it is not a linspace point. Bound-infeasible
        values are then removed. The nominal audit is not counted as a
        correction trial.
        """
        if abs(nominal_control) < 1e-14:
            return np.asarray([1.0])
        maximum = max(
            abs(self.u_min / nominal_control),
            abs(self.u_max / nominal_control),
            1.0,
        )
        raw = np.unique(np.r_[np.linspace(0.0, maximum, self.config.grid_size), 1.0])
        feasible = raw[
            (raw * nominal_control >= self.u_min - 1e-14)
            & (raw * nominal_control <= self.u_max + 1e-14)
        ]
        ordered = feasible[np.argsort(np.abs(feasible - 1.0))]
        if exclude_nominal:
            ordered = ordered[np.abs(ordered - 1.0) > 1e-14]
        return ordered

    def correct(self, initial_state: Array, nominal_controls: Sequence[float], duration: float) -> SequenceCorrection:
        state = np.asarray(initial_state, dtype=float).copy()
        accepted_controls: list[float] = []
        records: list[SegmentCorrection] = []
        for index, nominal in enumerate(np.asarray(nominal_controls, dtype=float)):
            nominal_peak, nominal_terminal = self.segment_peak(state, float(nominal), duration)
            if nominal_peak <= self.config.acceptance_threshold:
                records.append(SegmentCorrection(
                    index, float(nominal), float(nominal), 1.0,
                    nominal_peak, nominal_peak, False, True, 0,
                ))
                accepted_controls.append(float(nominal))
                # Reuse the single adaptive propagation of the safe nominal
                # segment rather than integrating it again.
                state = nominal_terminal
                continue

            chosen: tuple[float, float, Array] | None = None
            evaluations = 0
            for scale in self._candidate_lambdas(
                float(nominal),
                exclude_nominal=self.config.skip_known_unsafe_nominal_candidate,
            ):
                evaluations += 1
                candidate = float(scale * nominal)
                peak, terminal = self.segment_peak(state, candidate, duration)
                if peak <= self.config.acceptance_threshold:
                    chosen = (float(scale), peak, terminal)
                    break
            if chosen is None:
                records.append(SegmentCorrection(
                    index, float(nominal), None, None, nominal_peak, None,
                    False, False, evaluations,
                ))
                return SequenceCorrection(None, tuple(records), False, True)
            scale, corrected_peak, state = chosen
            corrected = float(scale * nominal)
            records.append(SegmentCorrection(
                index, float(nominal), corrected, scale, nominal_peak,
                corrected_peak, True, True, evaluations,
            ))
            accepted_controls.append(corrected)
        return SequenceCorrection(np.asarray(accepted_controls), tuple(records), True, False)
