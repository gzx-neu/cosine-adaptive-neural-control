# Formal cached discrete-31 HDS: 30-seed matched-400 results

All values use 31 closest-to-one candidates without bisection, adaptive DOP853 event audits, segment propagation reuse, and $g_{\max}\leq-10^{-8}$.

| Benchmark | Method | HDS gap (%; mean +/- seed SD) | Corrected segments | Nominal violation | Post-HDS violation | Accepted |
|---|---|---:|---:|---:|---:|---:|
| vdp | supervised | 0.39566 +/- 0.05456 | 0.5568 | 55.25% | 0.00% | 12000/12000 |
| vdp | unprocessed | 0.17154 +/- 0.02305 | 0.0883 | 8.79% | 0.00% | 12000/12000 |
| vdp | linear_cosine | 0.16559 +/- 0.02160 | 0.1119 | 11.10% | 0.00% | 12000/12000 |
| vdp | standard_pcgrad | 0.17729 +/- 0.03569 | 0.1812 | 18.00% | 0.00% | 12000/12000 |
| penicillin | supervised | 0.61741 +/- 0.09816 | 3.2049 | 99.20% | 0.00% | 12000/12000 |
| penicillin | unprocessed | 0.33122 +/- 0.07352 | 1.7730 | 92.66% | 0.00% | 12000/12000 |
| penicillin | linear_cosine | 0.31743 +/- 0.05277 | 1.9627 | 92.17% | 0.00% | 12000/12000 |
| penicillin | standard_pcgrad | 0.32749 +/- 0.06066 | 1.9637 | 91.93% | 0.00% | 12000/12000 |
| cstr | supervised | 0.24981 +/- 0.02431 | 10.9974 | 97.31% | 0.00% | 12000/12000 |
| cstr | unprocessed | 0.27359 +/- 0.01808 | 13.8737 | 99.57% | 0.00% | 12000/12000 |
| cstr | linear_cosine | 0.24399 +/- 0.02279 | 11.4988 | 97.89% | 0.00% | 12000/12000 |
| cstr | standard_pcgrad | 0.24294 +/- 0.02319 | 11.3437 | 97.65% | 0.00% | 12000/12000 |

CSTR qualified-286 gap:

| Method | Gap (%; mean +/- seed SD) |
|---|---:|
| supervised | 0.22337 +/- 0.02704 |
| unprocessed | 0.23286 +/- 0.02110 |
| linear_cosine | 0.20834 +/- 0.02603 |
| standard_pcgrad | 0.20823 +/- 0.02634 |

Batch timers were collected under concurrent CPU load and are throughput diagnostics, not serial online latency.
