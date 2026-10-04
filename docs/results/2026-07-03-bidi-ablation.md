# Bidirectional vs causal attention ablation (distal v1)

First-ever test of the project's namesake mechanism. Same corpus
(`instrumentation/out/distal_v1`, seed 1), same config (d_model 64, 4 heads,
2 layers, window 64, object_bias 4, RawAttention supervision), 5 model seeds
(7, 42, 99, 1234, 2025). `causal` masks future keys (j > i) via
`BibeConfig.causal` (src/model.rs).

Commands:
```
./target/release/examples/train_real instrumentation/out/distal_v1 raw 4.0 bidi
./target/release/examples/train_real instrumentation/out/distal_v1 raw 4.0 causal
```

Bidirectional attention does not help attribution on this benchmark. The
backward-only (causal) model is ahead on the mean:

| attention     | Hit@1         | Hit@3         | MRR           |
|---------------|---------------|---------------|---------------|
| bidirectional | 0.537 ± 0.118 | 0.874 ± 0.096 | 0.706 ± 0.084 |
| causal        | 0.805 ± 0.235 | 0.953 ± 0.082 | 0.884 ± 0.148 |

± here (and in every note in this directory) is the population standard
deviation over the 5 model seeds, n = 5. It is a spread, not a confidence
interval.

Detection and localization are 1.000 ± 0.000 under both.

The stds overlap, so nothing here is significant in either direction: causal
is not shown to be better, and bidirectional is not shown to be worse. The
reason to stop looking is structural rather than statistical. Every cause in
this benchmark precedes its symptom, so backward-only attention already sees
every event the label depends on, and the forward half can only contribute
events the model has to learn to ignore. A benchmark built this way has no
bidirectionality benefit available to measure, at any n.

## Decision

Per plan Task 5/13, the paper drops "bidirectional" from the title and
contribution claims. The README's "key insight" framing (crash explained by
future events) is a motivating hypothesis that no current benchmark
exercises; it may only be testable on real bugs where the
"should-have-happened-later" event actually appears in traces.
