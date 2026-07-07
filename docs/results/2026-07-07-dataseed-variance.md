# Data-seed variance (methodologist L1) + de-biased eval (L2)

Addresses two referee-pass threats to validity:
- **L1:** earlier ± std varied only the model init seed on ONE corpus
  realization, so deterministic-baseline rows showed a misleading `± 0.000`.
- **L2:** the harness scored only the first 64-event window, silently dropping
  long real traces whose crash lands past position 64.

Fix: `train_real` now scores the crash-containing window and reports
`attrib_dropped` (commit 6c449b3). Sweeps regenerate each corpus at 3 data
seeds × 5 model seeds; per-run logs in `docs/results/dataseed-logs/`.

## Distal v2 (gapped) — pooled over 3 data seeds × 5 model seeds (n=15)

| metric | single-seed (old) | pooled 3 data seeds (new) |
|---|---|---|
| transformer model Hit@1 (bidi, bias 4) | 0.877 ± 0.068 | **0.807 ± 0.152** |
| trig-window (oracle) Hit@1 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| recency Hit@1 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| same-obj write Hit@1 | 0.333 ± 0.000 | **0.355 ± 0.050** |
| attrib_dropped | 0 | 0 |

Per-data-seed transformer means: dseed 1 = 0.877 ± 0.068, dseed 2 = 0.805 ±
0.033, dseed 3 = 0.738 ± 0.232.

**Reading:**
- The methodologist was right: `± 0.000` was a single-draw artifact. The
  `same-obj write` baseline genuinely moves (0.355 ± 0.050) across corpus
  realizations; the deterministic oracle (1.000) and recency (0.000) are
  provably stable (guaranteed by the generator's `verify()` and by
  construction), so their `± 0.000` is real, not luck.
- With honest data-seed variance the transformer is **0.807 ± 0.152** — still
  below the oracle's 1.000 and statistically indistinguishable from the
  bi-LSTM's 0.856 ± 0.113. The headline conclusions are unchanged; only their
  error bars widened.

## Bidirectionality ablation on v2 (data seed 1, 5 model seeds)

| attention | Hit@1 |
|---|---|
| bidirectional (bias 4) | 0.877 ± 0.068 |
| causal (bias 4) | 0.708 ± 0.376 |

On v2 the causal arm has enormous variance (± 0.376) and overlaps bidi
completely — consistent with the v1 finding: **no benchmark shows
bidirectionality helping.** (On v1 the causal *mean* was higher; on v2 it is
lower but with a huge spread. Either way, no evidence bidirectionality helps —
the title-drop decision holds.)

## UAF (real ASan) + de-biased eval

Three ASan captures (data seeds 99/2/3), bias 0/4/8 + causal on d99, bias 4 on
all three, 5 model seeds each, with the crash-window eval (L2 fix).

**L2 result — the filter did not bite on this corpus.** `attrib_dropped = 0`,
82 scored, on every run: the templated UAF traces are short enough that the
crash always falls in the first window, so the de-biased numbers are identical
to the pre-fix ones (bias 0 = 0.488 ± 0.145, bias 4 = 0.824 ± 0.247, bias 8 =
0.893 ± 0.125). The selection filter was real in principle and bit hard on the
mjs pilot (whose crash sat in window 191), but it does not distort the
templated-corpus numbers. Honest empirical answer, not an assumption.

**Data-seed variance (bias 4).** Per data seed: d99 = 0.824 ± 0.247, d2 =
0.819 ± 0.230, d3 = 0.812 ± 0.138. **Pooled = 0.818 ± 0.210 (n=15).** The mean
is stable across corpus realizations; the wide ± is model-init variance at
bias 4. Deterministic rows are stable: recency 0.378, obj-recency (the UAF
oracle) 1.000, spectrum FL 0.000, detection AUC 0.997.

**Bidirectionality ablation on real UAF, pooled over 3 data seeds (5 model seeds each): causal wins, replicated.**

| attention | pooled Hit@1 (n=15) | per data seed (d99 / d2 / d3) |
|---|---|---|
| bidirectional (bias 4) | 0.818 ± 0.210 | 0.824 / 0.819 / 0.812 |
| causal (bias 4) | **0.989 ± 0.021** | 0.976 / 0.994 / 0.998 |

The backward-only model beats the bidirectional one on every one of the three
UAF realizations, by ~0.17 Hit@1, and its variance is tight (± 0.021 pooled)
and near the oracle's 1.000. This replicates the single-seed result and
upgrades it: bidirectionality does not merely fail to help on real UAF, it
costs ~0.17 Hit@1, consistently. Combined with v1 (causal higher mean) and v2
(causal overlaps), **no corpus shows bidirectionality helping, and the real
corpus shows it hurting across three realizations.** The title-drop decision
is backed by three benchmarks including real data.
