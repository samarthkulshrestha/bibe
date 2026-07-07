# Object-bias ablation matrix

`object_bias` is an additive attention bonus on same-object keys
(`src/model.rs`) — a hand-injected relational prior that encodes the domain
answer ("a use relates to same-object events") directly into attention.
This matrix quantifies how much of "the model's" performance is actually the
prior. 5 model seeds per cell (7, 42, 99, 1234, 2025); UAF corpus regenerated
with `corpus_gen 800 … 99` + ASan capture (398 anomalous; 82 anomalous test
traces). Runs: sequential arms in `docs/results/matrix-logs/`, per-seed
parallel arms in `docs/results/matrix-logs/fast/`.

## model Hit@1, mean ± std

| corpus            | bias 0        | bias 4        | bias 8        | best hand rule (Hit@1) |
|-------------------|---------------|---------------|---------------|------------------------|
| UAF (real, 800)   | 0.488 ± 0.145 | 0.824 ± 0.247 | 0.893 ± 0.125 | same-obj recency **1.000** (definitional) |
| distal v1 (adjacent) | 0.342 ± 0.160 | 0.537 ± 0.118 | 0.463 ± 0.209 | trig-adjacent **1.000** (oracle) |
| distal v2 (gapped)   | 0.441 ± 0.139 | 0.877 ± 0.068 | 0.913 ± 0.070 | trig-window **1.000** (oracle) |

UAF baseline ladder (deterministic, from the fast logs): recency 0.378,
same-obj recency 1.000, same-obj write 0.378, Ochiai 0.000, Tarantula 0.000.

## Findings

1. **The injected prior does heavy lifting everywhere:** +0.34 to +0.47
   Hit@1 from bias 0 → 8 on UAF and v2. Without it the model is at 0.34–0.49
   on every corpus. Any paper number using bias > 0 must be labeled as an
   oracle-prior upper bound, not a learned capability.
2. **The model never reaches the definitional/oracle rule on any corpus**,
   even with the strongest prior and cause supervision: best cell 0.913 vs
   the rule's 1.000.
3. **The prior can hurt when it points away from the marker:** on v1 the
   `trigger` carries object id 0 (invisible to the same-object prior), and
   bias 8 (0.463 ± 0.209) is no better than bias 4 (0.537 ± 0.118) — the
   prior drags attention toward same-object events while the label-defining
   marker is an object-less token. Consistent with "prior ≠ capability."
4. Detection AUC is 0.993–1.000 in every cell — detection was never the
   hard part.
