# Code map

## Unified entry points

| File | Purpose |
|---|---|
| `scripts/reproduce.py` | environment check, frozen-artifact verification, 30-seed training, aggregation and figure regeneration |
| `scripts/verify_bundle.py` | checks required inputs, aggregate invariants, seed coverage, 400-point cohort coverage and SHA-256 inventory |
| `scripts/build_minimal_release.py` | creates a clean staged GitHub copy without deleting the working package |
| `MINIMAL_REPRODUCTION.md` | staged reproduction path for the minimum GitHub release |
| `configs/minimal_release_manifest.json` | machine-readable required/optional stage manifest |
| `configs/paper_30seed.json` | machine-readable frozen experimental protocol |

## Core training and evaluation

| File | Purpose |
|---|---|
| `kkt_collocation/run_unified_su_suj_sk_konly_ablation.py` | VDP and penicillin training plus matched-400 HDS evaluation |
| `kkt_collocation/run_vdp_k10_cuda_multiseed.py` | multi-seed launcher for VDP or penicillin; supports a CPU override |
| `kkt_collocation/train_unified_economou_cstr_n100_ablation.py` | CSTR N=100/RK10 four-method training |
| `kkt_collocation/evaluate_unified_economou_cstr_n100_hds.py` | CSTR matched-400 HDS evaluation and all-400/qualified-286 gaps |
| `kkt_collocation/run_cstr_k10_cuda_multiseed.py` | multi-seed CUDA launcher for CSTR |
| `kkt_collocation/reevaluate_cstr_margin1e6_30seeds.py` | conservative CSTR re-audit with `g_max <= -1e-6` |
| `kkt_collocation/reevaluate_multiseed30_discrete31_cached.py` | unified reproduced-checkpoint re-evaluation: 30 seeds, matched 400 points, a 31-point lambda base grid without bisection, and cached segment propagation |
| `kkt_collocation/evaluate_ca_kkt_ood_stress_30seeds.py` | 30-seed guard-bypassed near/far OOD network audit on fixed 100-point layers |
| `kkt_collocation/run_ood_matched_reference_50x2.py` | fixed 50+50 OOD selection, CSTR cold references, reference audit and matched-gap aggregation |

## KKT and HDS implementation

| File | Purpose |
|---|---|
| `offline_safe_control/kkt_regularization.py` | reduced-space discrete-KKT residual components |
| `offline_safe_control/adaptive_event_hds.py` | adaptive DOP853 segment audit, event-located extrema, early-stop lambda search, and accepted-segment terminal-state reuse |
| `offline_safe_control/hds_lambda_corrector.py` | general HDS/lambda correction utilities |
| `kkt_collocation/economou_cstr_hds_fast.py` | CSTR event-based segment audit, fixed lambda candidates, and cached applied-segment peaks/states without redundant final replay |
| `kkt_collocation/run_vdp_ablation.py` | VDP model, path constraint and derivative event |
| `kkt_collocation/run_penicillin_ablation.py` | penicillin model, path constraint and derivative event |
| `kkt_collocation/screen_economou_cstr_30x30.py` | Economou CSTR model and transcription configuration |

## Aggregation and statistics

| File | Purpose |
|---|---|
| `kkt_collocation/aggregate_vdp_k10_projection_comparison.py` | paired VDP/penicillin seed aggregation |
| `kkt_collocation/aggregate_vdp_penicillin_k10_multiroot.py` | combines split run roots without selecting seeds |
| `kkt_collocation/aggregate_cstr_k10_projection_comparison.py` | CSTR all-400 and qualified-reference aggregation |
| `kkt_collocation/aggregate_cstr_k10_multiroot.py` | combines split CSTR run roots |

## Paper figures

| File | Purpose |
|---|---|
| `paper_assets/objective_gap_30seeds/plot_objective_gap_30seeds.py` | four-method 30-seed objective-gap figure |
| `kkt_collocation/plot_gradient_conflict_and_timing.py` | gradient cosine, projection coefficient and timing evidence |
| `kkt_collocation/plot_linear_cosine_constraint_population_30seeds.py` | complete 12,000-trajectory constraint population |
| `kkt_collocation/plot_linear_cosine_control_corrections_30seeds.py` | complete 12,000-sequence control population with segment-local corrections |
| `kkt_collocation/plot_ood_stress_diagnostic.py` | guard-bypassed OOD diagnostic |
| `kkt_collocation/plot_submission_figures.py` | legacy validation-gated experiment figures |

## Deterministic labels and references

| File | Purpose |
|---|---|
| `kkt_collocation/generate_vdp_kkt_data.py` | VDP finite-dimensional KKT labels |
| `kkt_collocation/generate_penicillin_kkt_data.py` | penicillin KKT labels |
| `kkt_collocation/generate_economou_cstr_reduced_kkt_data.py` | CSTR reduced-space controls and finite-dimensional multipliers |
| `kkt_collocation/run_vdp_cvp20_jiang_labels.m` | Jiang--Fu VDP wrapper |
| `kkt_collocation/run_penicillin_cvp20_jiang_labels.m` | Jiang--Fu penicillin wrapper |
| `matlab/jiang_fu_source/01_Single_Constraint_Cases/Proposed_Method/run_Codex_JiangFu_ood100.m` | sharded fixed-cold-start Jiang--Fu OOD references |
| `matlab/jiang_fu_source/` | clean archived deterministic upper-bound-method source (no nested Git metadata) |

## Frozen OOD artifacts

| Directory | Purpose |
|---|---|
| `kkt_collocation/results/ca_kkt_ood_stress_30seeds_20260803_v1/` | 18,000 guard-bypassed network/HDS trajectories and seed-level inputs |
| `kkt_collocation/results/ca_kkt_ood_matched_reference_50x2_20260804_v1/` | selected points, raw cold-start records, HDS-audited references and final table |
| `paper_assets/tables/ood_stress_table.tex` | compact manuscript table generated from the matched OOD summary |

Files not listed above are retained because some legacy tables and diagnostics
import them. They are supporting or exploratory utilities rather than separate
paper claims; the frozen protocol is always the one in
`configs/paper_30seed.json`.
