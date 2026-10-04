# Learned baseline: cause-supervised bi-LSTM (PyTorch) vs the transformer

Script: `baselines/lstm_attrib.py`, run on the same `.trace` files, the same
sorted-path 80/20 split, the same cause supervision (cross-entropy over
positions), the same metrics, 5 seeds (7, 42, 99, 1234, 2025). Model: bi-LSTM,
hidden 64, function + object embeddings, per-position cause head. 30 epochs,
Adam 1e-3.

Environment for the recorded numbers below: not captured at the time. The
script pins the five model seeds but not torch, and there is no
`requirements.txt`.

± is the population standard deviation over the 5 model seeds, n = 5
(`statistics.pstdev`). Both corpora are a single data-seed realization
(generator seed 1) and the LSTM has not been re-run across data seeds, so
these numbers are not matched against the pooled n=15 transformer figures in
`2026-07-07-dataseed-variance.md`.

See `baselines/README.md` for what was and was not tuned on this baseline.

## Finding

**A plain bi-LSTM beats the from-scratch transformer on the transformer's own
benchmark.** On distal v1 the transformer gets Hit@1 = 0.537 ± 0.118
(`docs/results/2026-07-03-distal-v1-oracle.md`); the LSTM gets 0.842 ± 0.117.
Any architecture-novelty claim is dead: the bespoke bidirectional-attention
model is not even the best *learned* model on the synthetic task designed for
it (and both remain below the hand-coded oracle rule's 1.000).

## Numbers

Distal v1 (adjacent):
```
lstm_hit1    0.842 ± 0.117
lstm_hit3    0.968 ± 0.031
lstm_mrr     0.905 ± 0.071
```

Distal v2 (gapped):
```
lstm_hit1    0.856 ± 0.113
lstm_hit3    0.964 ± 0.038
lstm_mrr     0.913 ± 0.068
```

### Re-run 2026-10-04 (distal v1)

`python3 baselines/lstm_attrib.py instrumentation/out/distal_v1`, 80 s wall,
torch 2.8.0 on macOS-26.6.2-arm64 (Apple silicon):

| metric | recorded above | re-run 2026-10-04 |
|---|---|---|
| `lstm_hit1` | 0.842 ± 0.117 | 0.847 ± 0.099 |
| `lstm_hit3` | 0.968 ± 0.031 | 0.968 ± 0.031 |
| `lstm_mrr`  | 0.905 ± 0.071 | 0.907 ± 0.063 |

The original numbers are left as recorded. Hit@3 reproducing exactly while
Hit@1 moves is the signature of a few borderline ranks flipping, not of a
different model. The seeds are pinned; the torch version is not, so this is
the drift to expect from an unpinned environment.

Caveats: the LSTM trains on whole traces (≤ 64 events, no windowing) with a
pure cause objective, while the Rust harness trains multi-objective
(anomaly + sparsity + attribution) on windows; the comparison favors neither
side obviously, but the protocol difference should be stated in the paper.
