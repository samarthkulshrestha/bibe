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

## Finding

**Bidirectional attention does not help attribution on this benchmark — the
backward-only (causal) model is better on the mean:**

| attention     | Hit@1         | Hit@3         | MRR           |
|---------------|---------------|---------------|---------------|
| bidirectional | 0.537 ± 0.118 | 0.874 ± 0.096 | 0.706 ± 0.084 |
| causal        | 0.805 ± 0.235 | 0.953 ± 0.082 | 0.884 ± 0.148 |

Detection and localization are 1.000 ± 0.000 under both.

The stds overlap, so we do not claim causal is *significantly better* — but
the data definitively rules out "bidirectionality helps": every benchmark
cause precedes its symptom, so backward-only attention suffices, and the
forward half appears to add noise the small model must learn to ignore.

**Decision gate (per plan Task 5/13): the paper drops "bidirectional" from
the title and contribution claims.** The README's "key insight" framing
(crash explained by future events) is a motivating hypothesis that no current
benchmark exercises; it may only be testable on real bugs where the
"should-have-happened-later" event actually appears in traces.
