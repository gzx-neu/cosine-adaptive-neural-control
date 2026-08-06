# Reproducibility protocol

## Experimental unit and pairing

The inferential unit is the independently initialized training seed. Every
method uses the same 30 seeds (`20260771--20260800`) and the same fixed 400 test
initial conditions inside each benchmark. The 400 trajectories within one seed
are not treated as 400 independent training replicates.

## Training budget

- common supervised initialization: 200 epochs, Adam, learning rate `1e-3`;
- continuation: 10 epochs, learning rate `1e-5`;
- supervised control-only baseline: matched 210-epoch schedule;
- rollout-consistency weight: zero;
- anchor weight: one;
- augmented penalty parameter: ten;
- fixed 31-point lambda base grid, searched by increasing distance from one;
- the nominal value `lambda=1` is audited explicitly and is not counted as a
  correction trial; bound-infeasible base-grid values are discarded;
- no lambda bisection;
- no test-dependent candidate insertion or seed selection.

For CA-KKT,

```text
c_t   = max(0, -cos(g_B, g_K))
eta_t = c_t
```

and `eta_t` controls the fraction of the conflicting KKT component removed.
`g_B` is the control-supervision plus anchor gradient and `g_K` is the weighted
finite-dimensional KKT gradient.

## Continuous-time audit

The deployment audit uses adaptive `solve_ivp(method="DOP853")`. Segment
endpoints and event-located stationary points of each path constraint are
included in the segment maximum. A nominally safe segment reuses its propagated
end state. An unsafe segment tests the feasible values of the unchanged
31-point discrete lambda base grid in
nearest-to-one order and stops at the first candidate satisfying the audit.
The accepted candidate's terminal state and peak are also reused. Because every
applied segment has then already been audited from the correct preceding state,
the implementation accumulates the cached segment maxima and does not replay
the complete corrected sequence a second time.

All three benchmark populations use the conservative numerical acceptance rule
`g_max <= -1e-6`. For CSTR, both all-400 results and the 286-point subset whose cold-start
reference passed the independent continuous audit are reported. The all-400
test is not reduced or filtered.

## Objective comparison

All relative differences compare the HDS-applied network control with the
deterministic cold-start solution at the same test initial condition. Penicillin
uses the minimization convention `J=-final_x3`. These are finite-dimensional
matched-reference differences, not continuous-time global-optimality claims.

## Out-of-domain stress protocol

The OOD diagnostic intentionally bypasses the domain guard. In declared
deployment, every OOD state is dispatched to the deterministic solver. The
near shell lies outside the training domain and within its 10% expansion; the
far shell lies between the 10% and 20% expansions. Only initially feasible
states are retained. Each shell has 100 frozen network-audit points shared by
all 30 seeds.

For the matched comparison, 50 points per shell are selected without
replacement using fixed seeds declared in
`ca_kkt_ood_matched_reference_50x2_20260804_v1/selection_protocol.json`. The
selection does not use network outcomes. VDP and penicillin use the authors'
Jiang--Fu upper-bound solver with fixed neutral cold starts; CSTR uses the
N=100/RK10 reduced-space control-only NLP with its fixed declared starts. No
network, label, or neighbouring solution initializes a reference solve.

The deterministic controls receive the same adaptive-event HDS
audit/correction and must satisfy `g_max <= -1e-6`. The reported accepted-policy
gap is

```text
100 * (J_network+HDS - J_cold-start+HDS) / abs(J_cold-start+HDS).
```

All six frozen reference cohorts have 50/50 solver success and 50/50 final HDS
acceptance. Network fallback remains visible through the reported HDS
acceptance rate and is never counted as a neural-policy gap.

## Timing scope

Online deployment time contains one network inference plus HDS audit/lambda
correction. Offline training and report-only objective reevaluation are
excluded. VDP and penicillin deterministic times are matched Jiang--Fu MATLAB
cold starts. CSTR uses the matched cold-start reduced-space N=100/RK10 NLP.

## Frozen results expected by the paper

| Benchmark | Linear-cosine HDS gap, mean +/- seed SD | Mean corrected segments |
|---|---:|---:|
| VDP | `0.1657 +/- 0.0215 %` | `0.1119` |
| Penicillin | `0.3175 +/- 0.0528 %` | `1.9628` |
| CSTR all-400 | `0.2442 +/- 0.0228 %` | `11.5093` |
| CSTR qualified-286 | `0.2085 +/- 0.0260 %` | same deployed controls |

The final cached discrete-31 re-evaluation is produced by
`kkt_collocation/reevaluate_multiseed30_discrete31_cached.py` after the unified
training command has created checkpoints under `reproduced_results`. It reads
the included matched deterministic references directly and does not depend on
machine-specific historical result directories. The precise archived values
are stored under
`kkt_collocation/results/formal_multiseed30_discrete31_cached_margin1e6_20260806_v1`
and are checked by `scripts/verify_bundle.py`; checkpoints are regenerated,
not duplicated in the publication bundle.

## Hardware and software

The archived 30-seed runs used Windows 11, Python 3.12.9, NumPy 2.2.4,
SciPy 1.15.2, PyTorch 2.6.0+cu126, pandas 2.3.2, matplotlib 3.10.1 and CasADi
3.7.2. VDP and CSTR training used CUDA; penicillin used CPU. All methods within
one benchmark used the same device protocol.

Exact wall-clock times can change across machines. Numerical summaries should
be compared within the recorded tolerances rather than by requiring identical
timing or bitwise-identical CUDA reductions.
