# Distal v2 (gapped) benchmark: full rule ladder

Corpus: `cargo run --release --example synth_distal_gen -- 400 instrumentation/out/distal_v2 1 gapped`
(400 traces, 183 anomalous; generator invariants verified on every trace:
causal write is never trigger-adjacent, trigger carries the object id, first
same-object write after the trigger is the cause, all traces ≤ 64 events).
Eval: `train_real instrumentation/out/distal_v2` (5 model seeds, object_bias 4,
bidirectional, cause-supervised — heuristics unsupervised, disclosed).

## Findings

1. **The v1 circularity fix worked:** trig-adjacent collapses from 1.000 to
   0.026 — no adjacency rule survives the gapped construction.
2. **The oracle rule is still perfect, by construction:** trig-window ("first
   same-object write after the same-object trigger") scores 1.000 ± 0.000.
   Rule-labeled synthetic data always has a perfect oracle; this row is the
   honesty anchor of the benchmark.
3. **The model closes most of the gap when the marker is visible to its
   priors:** 0.877 ± 0.068 (vs 0.537 ± 0.118 on v1, where the trigger token
   carried object id 0 and was invisible to the same-object attention bias).
   Comparable to the bi-LSTM on the same corpus (0.856 ± 0.113,
   `docs/results/2026-07-03-lstm-baseline.md`). Still below the oracle.
4. Recency-family and spectrum-FL baselines are all ≈ 0 at Hit@1.

## Aggregate over 5 seeds (7, 42, 99, 1234, 2025), mean ± std

```
detection_auc          1.000 ± 0.000
localize_hit1          1.000 ± 0.000
localize_mrr           1.000 ± 0.000
model_hit1             0.877 ± 0.068
model_hit3             0.985 ± 0.031
model_mrr              0.929 ± 0.044
recency_hit1           0.000 ± 0.000
recency_hit3           0.000 ± 0.000
recency_mrr            0.113 ± 0.000
obj_recency_hit1       0.000 ± 0.000
obj_recency_hit3       0.538 ± 0.000
obj_recency_mrr        0.295 ± 0.000
obj_write_hit1         0.333 ± 0.000
obj_write_hit3         1.000 ± 0.000
obj_write_mrr          0.658 ± 0.000
trig_adjacent_hit1     0.026 ± 0.000
trig_adjacent_hit3     0.026 ± 0.000
trig_adjacent_mrr      0.137 ± 0.000
trig_window_hit1       1.000 ± 0.000
trig_window_hit3       1.000 ± 0.000
trig_window_mrr        1.000 ± 0.000
ochiai_hit1            0.000 ± 0.000
ochiai_hit3            0.000 ± 0.000
ochiai_mrr             0.113 ± 0.000
tarantula_hit1         0.000 ± 0.000
tarantula_hit3         0.000 ± 0.000
tarantula_mrr          0.113 ± 0.000
```
