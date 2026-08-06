"""HDS-verified, segment-wise scalar safety correction for ZOH controls.

This module is intentionally separate from :mod:`lambda_filter`.  The latter
also tests a sampled CBF differential inequality, which is a sufficient but
often conservative condition.  The corrector below implements the procedure
used for the VDP ablation: HDS-style event location evaluates the *actual*
continuous-time path constraint, and a non-monotonicity-free grid search finds
the closest feasible scale on the nominal-control ray.  It is a numerical
model-based verifier/corrector, not a formal CBF certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

Array = np.ndarray
Ode = Callable[[float, Array, float], Array]
Constraint = Callable[[Array], float]  # g(x) <= 0
ConstraintDerivative = Callable[[Array, float], float]


@dataclass(frozen=True)
class HDSLambdaConfig:
    """Numerical settings for an offline HDS--lambda correction pass."""

    grid_size: int = 101
    # A sequence is accepted only if the computed peak satisfies
    # ``g_max <= -safety_margin``.  This conservative buffer is distinct from
    # the ODE solver tolerances, which are numerical controls rather than a
    # certified global error bound.
    safety_margin: float = 1e-8
    rtol: float = 1e-10
    atol: float = 1e-12
    max_step_fraction: float = 200.0
    integrator: str = "DOP853"
    # Once ``segment_peak`` has established that the nominal input is unsafe,
    # re-evaluating the identical lambda=1 candidate cannot change the answer.
    # Retaining it is useful only for legacy timing comparisons.
    skip_known_unsafe_nominal_candidate: bool = True

    def __post_init__(self) -> None:
        if self.grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        if self.safety_margin < 0:
            raise ValueError("safety_margin must be non-negative")
        if self.rtol <= 0 or self.atol <= 0:
            raise ValueError("rtol and atol must be positive")
        if self.max_step_fraction <= 0:
            raise ValueError("max_step_fraction must be positive")

    @property
    def acceptance_threshold(self) -> float:
        """Conservative upper bound required of every computed HDS peak."""
        return -float(self.safety_margin)


@dataclass(frozen=True)
class SegmentCorrection:
    segment: int
    nominal_control: float
    corrected_control: float | None
    lambda_value: float | None
    nominal_peak_g: float
    corrected_peak_g: float | None
    corrected: bool
    accepted: bool
    candidate_evaluations: int = 0


@dataclass(frozen=True)
class SequenceCorrection:
    controls: Array | None
    segments: tuple[SegmentCorrection, ...]
    accepted: bool
    requires_reoptimization: bool


class HDSLambdaCorrector:
    """Correct a complete, pre-execution ZOH sequence along scalar rays."""

    def __init__(
        self,
        ode: Ode,
        constraint: Constraint,
        constraint_derivative: ConstraintDerivative,
        control_bounds: tuple[float, float],
        config: HDSLambdaConfig = HDSLambdaConfig(),
    ) -> None:
        self.ode = ode
        self.constraint = constraint
        self.constraint_derivative = constraint_derivative
        self.u_min, self.u_max = map(float, control_bounds)
        self.config = config
        if self.u_min > self.u_max:
            raise ValueError("control bounds must be ordered")

    def _integrate(self, state: Array, control: float, duration: float):
        return solve_ivp(
            self.ode,
            (0.0, duration),
            np.asarray(state, dtype=float),
            args=(float(control),),
            method=self.config.integrator,
            rtol=self.config.rtol,
            atol=self.config.atol,
            max_step=duration / self.config.max_step_fraction,
            dense_output=True,
        )

    def segment_peak(self, state: Array, control: float, duration: float) -> tuple[float, Array]:
        """Return max g and terminal state, including event-located extrema."""
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
            max_step=duration / self.config.max_step_fraction,
            dense_output=True,
            events=stationary_event,
        )
        candidates = np.concatenate(([0.0, duration], solution.t_events[0]))
        values = np.asarray([self.constraint(solution.sol(time)) for time in candidates])
        return float(values.max()), solution.y[:, -1].copy()

    def audit(self, initial_state: Array, controls: Sequence[float], duration: float) -> float:
        """HDS maximum over a sequence; this does not modify the sequence."""
        state = np.asarray(initial_state, dtype=float).copy()
        peak = -np.inf
        for control in controls:
            local_peak, state = self.segment_peak(state, float(control), duration)
            peak = max(peak, local_peak)
        return float(peak)

    def _audit_trace(self, initial_state: Array, controls: Sequence[float], duration: float):
        """Return segment starts, HDS peaks, and terminals for one full rollout."""
        state = np.asarray(initial_state, dtype=float).copy()
        starts: list[Array] = []
        peaks: list[float] = []
        terminals: list[Array] = []
        for control in np.asarray(controls, dtype=float):
            starts.append(state.copy())
            peak, state = self.segment_peak(state, float(control), duration)
            peaks.append(peak)
            terminals.append(state.copy())
        return starts, np.asarray(peaks, dtype=float), terminals

    def _candidate_lambdas(self, nominal_control: float, *, exclude_nominal: bool = False) -> Array:
        if abs(nominal_control) < 1e-14:
            return np.asarray([1.0])
        lower = max(0.0, self.u_min / nominal_control, self.u_max / nominal_control)
        upper = min(self.u_min / nominal_control, self.u_max / nominal_control)
        # The preceding min/max form is awkward for a negative nominal input;
        # construct the interval explicitly and retain only physically feasible
        # points, which is robust for either control sign.
        raw = np.linspace(0.0, max(abs(self.u_min / nominal_control), abs(self.u_max / nominal_control), 1.0), self.config.grid_size)
        raw = np.unique(np.append(raw, 1.0))
        feasible = raw[(raw * nominal_control >= self.u_min - 1e-14) & (raw * nominal_control <= self.u_max + 1e-14)]
        if feasible.size == 0:
            return np.asarray([], dtype=float)
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
                records.append(SegmentCorrection(index, float(nominal), float(nominal), 1.0, nominal_peak, nominal_peak, False, True, 0))
                accepted_controls.append(float(nominal))
                state = nominal_terminal
                continue

            chosen: tuple[float, float, Array] | None = None
            candidate_evaluations = 0
            for scale in self._candidate_lambdas(
                float(nominal),
                exclude_nominal=self.config.skip_known_unsafe_nominal_candidate,
            ):
                candidate_evaluations += 1
                candidate = float(scale * nominal)
                peak, terminal = self.segment_peak(state, candidate, duration)
                if peak <= self.config.acceptance_threshold:
                    chosen = (float(scale), peak, terminal)
                    break
            if chosen is None:
                records.append(SegmentCorrection(index, float(nominal), None, None, nominal_peak, None, False, False, candidate_evaluations))
                return SequenceCorrection(None, tuple(records), False, True)
            scale, corrected_peak, state = chosen
            control = float(scale * nominal)
            records.append(SegmentCorrection(index, float(nominal), control, scale, nominal_peak, corrected_peak, True, True, candidate_evaluations))
            accepted_controls.append(control)
        return SequenceCorrection(np.asarray(accepted_controls), tuple(records), True, False)

    def correct_detect_then_correct(
        self, initial_state: Array, nominal_controls: Sequence[float], duration: float, *, max_passes: int = 3,
    ) -> SequenceCorrection:
        """Offline detect--correct--audit variant with violation-set updates.

        A complete nominal HDS audit first identifies the violating segments.
        Only those segments enter a lambda search.  Since an earlier correction
        can alter later segment initial states, the full corrected sequence is
        HDS-audited again; any newly unsafe segments are treated in a new pass.
        This preserves numerical safety checking but may be slower than
        :meth:`correct`, which couples detection and correction in one sweep.
        """
        original = np.asarray(nominal_controls, dtype=float)
        controls = original.copy()
        candidate_counts = np.zeros(len(controls), dtype=int)
        first_peaks: np.ndarray | None = None
        for _pass in range(max_passes):
            starts, peaks, _terminals = self._audit_trace(initial_state, controls, duration)
            if first_peaks is None:
                first_peaks = peaks.copy()
            unsafe = np.flatnonzero(peaks > self.config.acceptance_threshold)
            if unsafe.size == 0:
                records = []
                for index, (original_u, applied_u, peak) in enumerate(zip(original, controls, peaks)):
                    changed = bool(abs(applied_u - original_u) > 1e-14)
                    scale = None if abs(original_u) < 1e-14 and changed else (float(applied_u / original_u) if abs(original_u) >= 1e-14 else 1.0)
                    records.append(SegmentCorrection(
                        index, float(original_u), float(applied_u), scale,
                        float(first_peaks[index]), float(peak), changed, True, int(candidate_counts[index]),
                    ))
                return SequenceCorrection(controls.copy(), tuple(records), True, False)

            unsafe_set = set(int(index) for index in unsafe)
            state = np.asarray(initial_state, dtype=float).copy()
            changed_this_pass = False
            for index, control in enumerate(controls.copy()):
                if index not in unsafe_set:
                    # This segment is not selected for correction in this pass.
                    # A terminal integration is sufficient to carry the altered
                    # state to the next selected segment; the final audit retains
                    # the continuous-time peak verification.
                    state = self._integrate(state, float(control), duration).y[:, -1].copy()
                    continue
                # A correction in an earlier selected segment can change the
                # state at this segment.  Re-evaluate this selected nominal
                # input before searching its lambda ray.
                current_peak, nominal_terminal = self.segment_peak(state, float(control), duration)
                if current_peak <= self.config.acceptance_threshold:
                    state = nominal_terminal
                    continue
                chosen: tuple[float, float, Array] | None = None
                for scale in self._candidate_lambdas(
                    float(control), exclude_nominal=self.config.skip_known_unsafe_nominal_candidate,
                ):
                    candidate_counts[index] += 1
                    peak, terminal = self.segment_peak(state, float(scale * control), duration)
                    if peak <= self.config.acceptance_threshold:
                        chosen = (float(scale), peak, terminal)
                        break
                if chosen is None:
                    records = tuple(SegmentCorrection(
                        item, float(original[item]), None, None, float(first_peaks[item]), None,
                        False, False, int(candidate_counts[item]),
                    ) for item in range(index + 1))
                    return SequenceCorrection(None, records, False, True)
                scale, _peak, state = chosen
                controls[index] = scale * control
                changed_this_pass = True
            if not changed_this_pass:
                break
        # One final trace establishes the fallback condition after exhausting
        # the bounded offline correction passes.
        _starts, peaks, _terminals = self._audit_trace(initial_state, controls, duration)
        if np.all(peaks <= self.config.acceptance_threshold):
            records = []
            for index, (original_u, applied_u, peak) in enumerate(zip(original, controls, peaks)):
                changed = bool(abs(applied_u - original_u) > 1e-14)
                scale = None if abs(original_u) < 1e-14 and changed else (float(applied_u / original_u) if abs(original_u) >= 1e-14 else 1.0)
                records.append(SegmentCorrection(
                    index, float(original_u), float(applied_u), scale,
                    float(first_peaks[index] if first_peaks is not None else peak), float(peak),
                    changed, True, int(candidate_counts[index]),
                ))
            return SequenceCorrection(controls.copy(), tuple(records), True, False)
        records = tuple(SegmentCorrection(
            index, float(original_u), None, None, float(first_peaks[index] if first_peaks is not None else peaks[index]), None,
            False, False, int(candidate_counts[index]),
        ) for index, original_u in enumerate(original))
        return SequenceCorrection(None, records, False, True)
