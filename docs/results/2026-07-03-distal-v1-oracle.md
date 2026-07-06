# Distal v1 benchmark: oracle-rule baselines vs model

Corpus: `cargo run --release --example synth_distal_gen -- 400 instrumentation/out/distal_v1 1`
(400 traces, 193 anomalous; 320 train / 80 test, 38 anomalous test traces with cause ≠ crash).
Eval: `cargo run --release --example train_real -- instrumentation/out/distal_v1`
(config: d_model 64, 4 heads, 2 layers, window 64, object_bias 4, RawAttention supervision,
`attribution_lambda = 1.0` — the model is supervised on the ground-truth cause;
all heuristic baselines are unsupervised).

## Finding

The v1 distal generator emits the causal write as an atomic step
`[("trigger", 0), ("write", oid)]` with filler only between steps, so the
causal write is always trigger-adjacent. The planted oracle rule
("most-recent same-object write whose previous event is a trigger") is
**perfect**, and the model — trained *with cause supervision* on a benchmark
constructed to favor it — sits far below it with high seed variance:

**trig-adjacent Hit@1 = 1.000 ± 0.000 vs model Hit@1 = 0.537 ± 0.118.**

The previously reported model 0.689 (PROGRESS.md) was a single seed; across 5
seeds the mean is 0.537 and the swing is ~0.3. The claim that this benchmark
"survives the just-hand-code-it critique" is retracted: the strongest
hand-coded rule was simply never implemented, and the baselines that were
implemented are structurally blind to the trigger token (`object_id = 0`).

## Aggregate over 5 seeds (7, 42, 99, 1234, 2025), mean ± std

```
detection_auc          1.000 ± 0.000
localize_hit1          1.000 ± 0.000
localize_mrr           1.000 ± 0.000
model_hit1             0.537 ± 0.118
model_hit3             0.874 ± 0.096
model_mrr              0.706 ± 0.084
recency_hit1           0.000 ± 0.000
recency_hit3           0.026 ± 0.000
recency_mrr            0.101 ± 0.000
obj_recency_hit1       0.000 ± 0.000
obj_recency_hit3       0.500 ± 0.000
obj_recency_mrr        0.292 ± 0.000
obj_write_hit1         0.289 ± 0.000
obj_write_hit3         1.000 ± 0.000
obj_write_mrr          0.632 ± 0.000
trig_adjacent_hit1     1.000 ± 0.000
trig_adjacent_hit3     1.000 ± 0.000
trig_adjacent_mrr      1.000 ± 0.000
trig_window_hit1       0.000 ± 0.000   (v1 trigger has object_id 0 — same-object
trig_window_hit3       0.026 ± 0.000    search finds nothing; degrades to recency.
trig_window_mrr        0.101 ± 0.000    This rule is the oracle for the v2 corpus.)
```

## Single-seed reference run (seed 7, first run after adding the oracles)

```
attribution over 38 use-after-free (Hit@1 / Hit@3 / MRR):
  model              0.500   0.632   0.651
  recency baseline   0.000   0.026   0.101
  same-obj recency   0.000   0.500   0.292
  same-obj write     0.289   1.000   0.632
  trig-adjacent      1.000   1.000   1.000
  trig-window        0.000   0.026   0.101
```
