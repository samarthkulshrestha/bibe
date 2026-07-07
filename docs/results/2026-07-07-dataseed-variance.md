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

[pending: dataseed_uaf.sh — 3 ASan captures at data seeds 99/2/3, then bias
0/4/8 + causal per seed, with the crash-window eval and attrib_dropped count.]
