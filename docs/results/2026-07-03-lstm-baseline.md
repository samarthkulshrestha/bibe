# Learned baseline: cause-supervised bi-LSTM (PyTorch) vs the transformer

Script: `baselines/lstm_attrib.py` — same `.trace` files, same sorted-path
80/20 split, same cause supervision (cross-entropy over positions), same
metrics, 5 seeds (7, 42, 99, 1234, 2025). Model: bi-LSTM, hidden 64, function
+ object embeddings, per-position cause head. 30 epochs, Adam 1e-3.

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

Caveats: the LSTM trains on whole traces (≤ 64 events, no windowing) with a
pure cause objective, while the Rust harness trains multi-objective
(anomaly + sparsity + attribution) on windows; the comparison favors neither
side obviously, but the protocol difference should be stated in the paper.
