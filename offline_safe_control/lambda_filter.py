"""Continuous-time, segment-wise lambda safety filtering for ZOH policies.

The filter is deliberately an *offline* component.  Given a measured initial
state, a neural policy first generates the whole nominal ZOH sequence.  The
filter then simulates and corrects that sequence before it is sent to the
plant.  It never uses measurements collected while the sequence is executed.

The mathematical safety statement used in the paper is conditional: if a
candidate satisfies h(x(t)) >= 0 and h_dot(x(t),u) + alpha*h(x(t)) >= 0 over a
segment whose initial state is safe, the comparison lemma makes the segment
forward invariant.  Numerical integration below is an implementation check of
those conditions; it is not a replacement for the theorem or a claim of
validated interval arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp

Array = np.ndarray
Ode = Callable[[float, Array, float], Array]
Barrier = Callable[[Array], float]
BarrierDerivative = Callable[[Array, float], float]


@dataclass(frozen=True)
class LambdaFilterConfig:
    """Configuration of the scalar ray search u=lambda*u_nom."""

    alpha: float
    lambda_min: float = 0.0
    lambda_max: float = 2.0
    grid_size: int = 61
    safety_tolerance: float = 1e-8
    cbf_tolerance: float = 1e-8
    integration_rtol: float = 1e-10
    integration_atol: float = 1e-12
    max_step_fraction: float = 200.0
    cbf_check_points: int = 401

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not (0 <= self.lambda_min <= self.lambda_max):
            raise ValueError("lambda bounds must satisfy 0 <= min <= max")
        if self.grid_size < 2 or self.cbf_check_points < 3:
            raise ValueError("grid_size and cbf_check_points are too small")


@dataclass(frozen=True)
class SegmentCertificate:
    """Result of certifying one ZOH segment."""

    segment: int
    nominal_u: float
    corrected_u: Optional[float]
    lambda_value: Optional[float]
    max_constraint: float
    min_cbf_residual: float
    certified: bool
    reason: str


@dataclass(frozen=True)
class SequenceCertificate:
    """Result of filtering a complete pre-execution open-loop sequence."""

    corrected_controls: Optional[Array]
    segment_certificates: tuple[SegmentCertificate, ...]
    certified: bool
    requires_reoptimization: bool


class LambdaSafetyFilter:
    """Grid-search safety filter without a monotonicity assumption.

    ``barrier`` is h(x), so safety means h >= 0.  ``barrier_derivative`` must
    return h_dot under the supplied scalar control.  A candidate is accepted
    only when both the path constraint and the CBF differential condition pass.
    """

    def __init__(
        self,
        ode: Ode,
        barrier: Barrier,
        barrier_derivative: BarrierDerivative,
        control_bounds: tuple[float, float],
        config: LambdaFilterConfig,
    ) -> None:
        u_min, u_max = control_bounds
        if u_min > u_max:
            raise ValueError("control_bounds must be (min, max)")
        self.ode = ode
        self.barrier = barrier
        self.barrier_derivative = barrier_derivative
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.config = config

    def _integrate(self, x0: Array, u: float, duration: float):
        if duration <= 0:
            raise ValueError("duration must be positive")
        return solve_ivp(
            self.ode,
            (0.0, duration),
            np.asarray(x0, dtype=float),
            args=(float(u),),
            method="DOP853",
            rtol=self.config.integration_rtol,
            atol=self.config.integration_atol,
            max_step=duration / self.config.max_step_fraction,
            dense_output=True,
        )

    def _hds_max_violation(self, x0: Array, u: float, duration: float) -> float:
        """Track gamma=max g with event-located stationary points of g=-h.

        This stable implementation evaluates the running-maximum state gamma
        at the two segment endpoints and every event satisfying g_dot=0.  It
        is equivalent to the HDS maximum monitor when all activation and
        deactivation events are located, while avoiding zero-time switching
        chatter in a generic ODE solver.
        """
        def stationary_event(t: float, x: Array, *_ignored: float) -> float:
            # g=-h, so g_dot=0 iff h_dot=0.
            return self.barrier_derivative(x, u)

        stationary_event.direction = 0
        stationary_event.terminal = False
        solution = solve_ivp(
            self.ode,
            (0.0, duration),
            np.asarray(x0, dtype=float),
            args=(float(u),),
            method="DOP853",
            rtol=self.config.integration_rtol,
            atol=self.config.integration_atol,
            max_step=duration / self.config.max_step_fraction,
            events=stationary_event,
            dense_output=True,
        )
        times = np.concatenate(([0.0, duration], solution.t_events[0]))
        states = solution.sol(times)
        gamma = float(np.max([-self.barrier(states[:, i]) for i in range(states.shape[1])]))
        return gamma

    def _candidate_lambdas(self, nominal_u: float) -> Array:
        candidates = np.linspace(
            self.config.lambda_min, self.config.lambda_max, self.config.grid_size
        )
        if self.config.lambda_min <= 1.0 <= self.config.lambda_max:
            candidates = np.append(candidates, 1.0)
        candidates = np.unique(candidates)
        feasible = candidates[(candidates * nominal_u >= self.u_min) & (candidates * nominal_u <= self.u_max)]
        # Closest-to-one ordering implements minimum intervention after safety.
        return feasible[np.argsort(np.abs(feasible - 1.0))]

    def audit_sequence(
        self, initial_state: Array, controls: Sequence[float], duration: float
    ) -> float:
        """Return the HDS maximum of ``g=-h`` for a nominal ZOH sequence.

        This is a read-only pre-execution audit: it deliberately does not
        apply CBF filtering or reject a trajectory whose intermediate state is
        unsafe.  It is therefore suitable for the validation gate, where the
        purpose is to quantify the raw supervised-policy violation.
        """
        state = np.asarray(initial_state, dtype=float).copy()
        maximum = -np.inf
        for control in controls:
            maximum = max(maximum, self._hds_max_violation(state, float(control), duration))
            state = self._integrate(state, float(control), duration).y[:, -1]
        return float(maximum)

    def certify_segment(
        self, x0: Array, nominal_u: float, duration: float, segment: int
    ) -> tuple[SegmentCertificate, Optional[Array]]:
        """Return the least-modified certified candidate and its terminal state."""
        if self.barrier(np.asarray(x0, dtype=float)) < -self.config.safety_tolerance:
            return SegmentCertificate(
                segment, nominal_u, None, None, np.nan, np.nan, False,
                "unsafe segment initial state; re-optimization is required",
            ), None

        best_failure: Optional[SegmentCertificate] = None
        for lam in self._candidate_lambdas(float(nominal_u)):
            u = float(lam * nominal_u)
            sol = self._integrate(x0, u, duration)
            times = np.linspace(0.0, duration, self.config.cbf_check_points)
            states = sol.sol(times)
            h_values = np.array([self.barrier(states[:, i]) for i in range(states.shape[1])])
            cbf_residuals = np.array([
                self.barrier_derivative(states[:, i], u) + self.config.alpha * h_values[i]
                for i in range(states.shape[1])
            ])
            max_constraint = self._hds_max_violation(x0, u, duration)
            min_residual = float(np.min(cbf_residuals))
            certified = (
                max_constraint <= self.config.safety_tolerance
                and min_residual >= -self.config.cbf_tolerance
            )
            record = SegmentCertificate(
                segment, float(nominal_u), u, float(lam), max_constraint,
                min_residual, certified,
                "certified" if certified else "path or CBF condition failed",
            )
            if certified:
                return record, sol.y[:, -1].copy()
            if best_failure is None or max_constraint < best_failure.max_constraint:
                best_failure = record

        assert best_failure is not None
        return SegmentCertificate(
            segment, float(nominal_u), None, None, best_failure.max_constraint,
            best_failure.min_cbf_residual, False,
            "safe lambda set is empty; re-optimization is required",
        ), None

    def filter_sequence(
        self, initial_state: Array, nominal_controls: Sequence[float], duration: float
    ) -> SequenceCertificate:
        """Certify the full ZOH sequence before offline execution."""
        state = np.asarray(initial_state, dtype=float).copy()
        corrected: list[float] = []
        certificates: list[SegmentCertificate] = []
        for segment, nominal_u in enumerate(nominal_controls):
            certificate, terminal_state = self.certify_segment(state, float(nominal_u), duration, segment)
            certificates.append(certificate)
            if not certificate.certified:
                return SequenceCertificate(None, tuple(certificates), False, True)
            corrected.append(float(certificate.corrected_u))
            state = terminal_state  # type: ignore[assignment]
        return SequenceCertificate(np.asarray(corrected), tuple(certificates), True, False)
