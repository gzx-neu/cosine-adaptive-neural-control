# CA-KKT reproducibility package

This repository reproduces the numerical experiments for cosine-adaptive
discrete-KKT continuation (CA-KKT) on the constrained VDP, penicillin, and
Economou CSTR benchmarks.

The package contains two reproducibility tracks:

1. **Compact frozen verification** checks the archived headline aggregate,
   OOD evidence, seed coverage, and the exact training/reference inputs. It
   does not ship historical checkpoints or complete trajectory caches.
2. **Full recomputation** trains all four methods for 30 seeds, evaluates every
   model on the same 400 matched test points, performs adaptive continuous-time
   HDS auditing, and rebuilds the aggregate statistics.

The HDS output is continuous-time numerical audit evidence under the declared
model and numerical settings. It is not an absolute safety guarantee for a
physical plant.

## Benchmarks and methods

The fixed seeds are `20260771` through `20260800`.

| Benchmark | Training device | Discretization | Test points | Deterministic reference |
|---|---|---|---:|---|
| VDP | CUDA | 10 ZOH controls | 400 | matched Jiang--Fu cold start |
| Penicillin | CPU | 10 ZOH controls | 400 | matched Jiang--Fu cold start |
| Economou CSTR | CUDA | 100 ZOH controls, RK10 transcription | 400 | cold-start reduced-space control-vector NLP |

Each benchmark compares:

- `supervised`: control-only supervised training with matched total budget;
- `unprocessed`: supervised initialization followed by direct KKT continuation;
- `linear_cosine`: CA-KKT with `eta=max(0,-cos(g_B,g_K))`;
- `standard_pcgrad`: full PCGrad projection of a conflicting KKT gradient.

The continuation protocol is 200 supervised epochs plus 10 continuation
epochs. The supervised baseline uses the matched 210-epoch schedule.

## Quick start

For the recommended GitHub release path, follow
[MINIMAL_REPRODUCTION.md](MINIMAL_REPRODUCTION.md). It divides the work into
independent environment, per-benchmark training, aggregation, cached-HDS, and
optional-figure stages. A one-command end-to-end run is not required.

Create the exact Python environment:

```bash
conda env create -f environment.yml
conda activate ca-kkt
python scripts/reproduce.py check
```

Run the fast code-level regression tests:

```bash
python -m unittest discover -s tests -v
```

Verify the archived 30-seed headline summary and required frozen inputs:

```bash
python scripts/verify_bundle.py
python scripts/reproduce.py frozen
```

The `figures` command is retained for authors who also download the optional
full trajectory-cache archive. Those large caches and rendered manuscript
files are intentionally not part of the compact GitHub repository.

Re-audit the frozen OOD deterministic references and rebuild the matched
30-seed stress-test table:

```bash
python scripts/reproduce.py ood
```

Run one smoke seed before starting the full experiment:

```bash
python scripts/reproduce.py train --benchmarks vdp penicillin cstr --seeds 20260771
python scripts/reproduce.py aggregate --seeds 20260771
```

Run the complete 30-seed experiment:

```bash
python scripts/reproduce.py train --benchmarks vdp penicillin cstr
python scripts/reproduce.py aggregate
```

After the training command has populated `reproduced_results`, re-evaluate
those checkpoints without retraining or recomputing deterministic references,
using the final cached 31-point-base-grid HDS path:

```bash
python scripts/reproduce.py hds-cached31 --workers 8
```

For the separately labeled threshold-sensitivity check, keep the formal
default unchanged and write to a new output directory, for example:

```powershell
$env:CAKKT_HDS_THRESHOLD='-1e-8'
python -m kkt_collocation.reevaluate_multiseed30_discrete31_cached `
  --output kkt_collocation/results/nonformal_threshold1e8_discrete31_cached `
  --workers 8 --resume
```

This changes only the declared numerical acceptance threshold; it does not
retrain networks or regenerate deterministic references. The resulting files
must be labeled nonformal threshold-sensitivity results and must not overwrite
the archived `-1e-6` formal aggregate.

This audit uses no lambda bisection. It reuses the terminal state and peak from
every accepted nominal or corrected segment and therefore omits the redundant
full-sequence replay. Add `--resume` after an interrupted re-evaluation. The
published aggregate and per-seed audit outputs are included under
`kkt_collocation/results/formal_multiseed30_discrete31_cached_margin1e6_20260806_v1`
and can be checked with `python scripts/reproduce.py frozen`; checkpoint files
are intentionally not duplicated in the publication bundle.

Add `--resume` to the training command after an interrupted run; completed
method/seed directories are retained and the launcher records every child
command in its manifest.

The full run is computationally expensive. Deterministic reference generation
is not repeated by default because the matched cold-start references are
included and identified by checksums.

## Out-of-domain stress diagnostic

The frozen OOD experiment uses two non-overlapping expanded-box shells:
`near` is outside the training domain but within its 10% expansion, and `far`
is between the 10% and 20% expansions. Each layer contains 100 fixed states for
network auditing. A performance-independent fixed RNG selects 50 states from
each layer for matched deterministic reference solves. The same points are
shared by all 30 training seeds.

VDP and penicillin references use independent Jiang--Fu cold starts. CSTR uses
the N=100/RK10 reduced-space control-vector NLP. Every deterministic solution
is passed through the same adaptive-event HDS audit/correction before the gap
is computed. All six reference cohorts achieved 50/50 solver success and 50/50
final acceptance at `g_max <= -1e-6`.

The reported diagnostic is

```text
100 * (J_network+HDS - J_cold-start+HDS) / abs(J_cold-start+HDS).
```

All OOD states would be routed directly to the deterministic solver by the
declared domain guard. The bypassed neural-policy results are a stress
diagnostic, not an OOD safety or global-optimality guarantee.

## Directory layout

```text
configs/                 frozen protocol declaration
scripts/                 unified checks, training and aggregation entry points
kkt_collocation/         benchmark, training, evaluation and plotting code
offline_safe_control/    adaptive-event HDS and KKT utilities
matlab/jiang_fu_source/   clean Jiang--Fu deterministic-solver source and wrappers
paper_assets/             final figures, source tables and plotting entry points
论文写作/figures/          compatibility output location used by plotting scripts
```

The selected inputs and frozen outputs retain their original relative paths
under `kkt_collocation/data` and `kkt_collocation/results`. This is intentional:
the experiment drivers resolve paths relative to the repository root and do
not require machine-specific absolute paths.

See [CODE_MAP.md](CODE_MAP.md) for the role of each entry point and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact protocol, timing scope,
and expected outputs.

## GitHub release scope

Generated checkpoints, historical searches, complete trajectory caches,
rendered manuscript figures, and temporary reruns are ignored. The repository
does include the compact label, split, test-cohort, and matched-reference files
needed to rerun the three 30-seed experiments. All included files are below
GitHub's 100 MB per-file limit.

The optional full trajectory-cache archive can be deposited separately on
Zenodo or attached to a GitHub Release. Before publication, add the final
manuscript title, author list, DOI/Zenodo identifier, and citation metadata.
