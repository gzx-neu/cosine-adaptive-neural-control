# Minimal staged reproduction

The GitHub release should expose the headline experiment as independent
stages. No single command is required to run all three benchmarks or all 30
seeds. This keeps failures restartable and makes the computational cost
explicit.

The target result is the formal 30-seed comparison of S-u, unprocessed KKT,
linear-cosine CA-KKT, and standard PCGrad on 400 matched test points per seed.
The archived deterministic labels and matched references are reused; label or
reference generation is an optional provenance stage, not a prerequisite for
reproducing the reported neural-policy tables.

## Stage 0: environment and frozen-input check

```bash
conda env create -f environment.yml
conda activate ca-kkt
python scripts/reproduce.py check
python scripts/verify_bundle.py
python -m unittest discover -s tests -v
```

## Stage 1: train one benchmark at a time

Run a smoke seed first, then expand to all frozen seeds. These commands can be
run independently and resumed.

```bash
python scripts/reproduce.py train --benchmarks vdp --seeds 20260771
python scripts/reproduce.py train --benchmarks penicillin --seeds 20260771
python scripts/reproduce.py train --benchmarks cstr --seeds 20260771
```

For the formal run, replace the single seed with the 30 frozen seeds and run
each benchmark separately. VDP/CSTR use the declared CUDA protocol; penicillin
uses CPU.

## Stage 2: aggregate trained runs

```bash
python scripts/reproduce.py aggregate --seeds 20260771
```

For the complete experiment, use the 30 frozen seeds. Aggregation does not
choose seeds and does not regenerate references.

## Stage 3: cached continuous-time HDS re-evaluation

```bash
python scripts/reproduce.py hds-cached31 --workers 8 --resume
```

This stage uses adaptive DOP853 event audits, exactly 31 discrete lambda
candidates, closest-to-one early stopping, and accepted-segment propagation
reuse. The formal acceptance threshold is \(g_{\max}\le-10^{-8}\).

The separate sensitivity check is optional and must use a new output directory:

```powershell
$env:CAKKT_HDS_THRESHOLD='-1e-6'
python -m kkt_collocation.reevaluate_multiseed30_discrete31_cached `
  --output kkt_collocation/results/historical_threshold1e6_discrete31_cached `
  --workers 8 --resume
```

It is a historical threshold result and must not be mixed with the formal \(-10^{-8}\) aggregate.

## Stage 4: tables and figures (optional presentation stage)

```bash
python scripts/reproduce.py figures
```

This stage is not needed to reproduce the numerical tables. It only regenerates
the manuscript visualizations from the frozen source data.

## What is intentionally excluded from the minimum path

- full historical checkpoint archives and trajectory caches;
- exploratory lambda-grid, OOD, and legacy diagnostics;
- MATLAB label/reference regeneration;
- manuscript source files and rendered figures;
- threshold-sensitivity outputs.

Those materials can be distributed as a separate archive or release asset.
The formal headline result is checked by the archived aggregate files and the
stage-specific scripts above.
