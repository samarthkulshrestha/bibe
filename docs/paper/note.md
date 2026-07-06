# When Does Learned Root-Cause Attribution Beat a One-Line Heuristic?

> Draft status: OUTLINE + partial numbers. Every number in this note must be
> copied, with its ± std, from a file in `docs/results/` — no number appears
> here that is not in a results file. Sections marked [pending: <results file>]
> are waiting on runs in flight.

**Claim (the only claim):** For bug classes with automatic oracles
(sanitizer-catchable), the causal event is definitionally the most-recent
related event, and a one-line heuristic achieves Hit@1 = 1.0 — learned
attribution adds nothing. Learning begins to earn its keep only when the
cause is distal and relational; we characterize that regime with a synthetic
capability probe and an honest account of its limits.

## 1. Introduction — the negative result is the finding

- Debugging tools locate symptoms; root-cause attribution is the hard part.
- The tempting ML story: train an attention model over execution traces,
  read causes off the attention map.
- What we found instead, in order: (a) on the one bug class with a free
  oracle (UAF via ASan), a one-line heuristic is perfect and the learned
  model is not; (b) on a synthetic benchmark built so recency heuristics
  fail, the generator's own oracle rule is perfect and the learned model is
  not; (c) a plain bi-LSTM beats our bespoke bidirectional-attention model
  on its own benchmark. We publish the boundary, not a victory.

## 2. Setup

- Traces: function-call events with timestamps, aux counters, and an object
  id derived from captured allocation addresses. Corpora: (i) real ASan-
  labeled executions of small templated C programs (UAF), (ii) synthetic
  distal-cause generators v1 (adjacent) / v2 (gapped) with injected labels.
- Model: from-scratch Rust transformer encoder — d_model 64, 4 heads,
  2 layers, window 64 (engineering color only; not a contribution).
- Supervision DISCLOSED: the model trains with `attribution_lambda = 1.0`
  on the ground-truth cause; all heuristic baselines are unsupervised. The
  bi-LSTM baseline is cause-supervised identically.
- Metrics: Hit@1/Hit@3/MRR over candidate events (crash excluded), mean ±
  std over 5 seeds (7, 42, 99, 1234, 2025).

## 3. Negative control: use-after-free

- The cause of a UAF is *definitionally* the most-recent same-object event
  before the crash — the recency-on-object heuristic scores Hit@1 = 1.0.
- Model without object bias: ≈ 0.585; ≈ 0.99 only with the same-object
  attention bias hand-injected (`object_bias`) — an oracle prior, reported
  as an upper bound, not a capability.
- [pending: bias-matrix run → docs/results/2026-07-03-object-bias-matrix.md
  for fresh mean ± std at bias 0/4/8]

## 4. Capability probe: synthetic distal causes (full rule ladder)

From `docs/results/2026-07-03-distal-v1-oracle.md` (v1, adjacent):

| rule / model            | Hit@1         | Hit@3         | MRR           |
|-------------------------|---------------|---------------|---------------|
| recency                 | 0.000 ± 0.000 | 0.026 ± 0.000 | 0.101 ± 0.000 |
| same-obj recency        | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.292 ± 0.000 |
| same-obj write          | 0.289 ± 0.000 | 1.000 ± 0.000 | 0.632 ± 0.000 |
| Ochiai FL               | 0.000 ± 0.000 | 0.026 ± 0.000 | 0.101 ± 0.000 |
| Tarantula FL            | 0.000 ± 0.000 | 0.026 ± 0.000 | 0.101 ± 0.000 |
| **trig-adjacent (oracle)** | **1.000 ± 0.000** | 1.000 ± 0.000 | 1.000 ± 0.000 |
| bi-LSTM (supervised)    | 0.842 ± 0.117 | 0.968 ± 0.031 | 0.905 ± 0.071 |
| transformer (supervised)| 0.537 ± 0.118 | 0.874 ± 0.096 | 0.706 ± 0.084 |

- Framing: rule-labeled synthetic data always has an oracle rule by
  construction. The v1 construction additionally hid its marker from the
  same-object baselines (`trigger` carried object id 0) — we report this as
  a cautionary tale about self-refereed benchmarks.
- v2 (gapped) removes the hidden token and breaks adjacency; the oracle
  becomes trig-window. [pending: docs/results/2026-07-03-distal-v2-ladder.md]
- The observation that survives: cause supervision alone recovers a planted
  relational pattern well above the recency family — but below both the
  oracle rule and a plain bi-LSTM.

## 5. Ablations

- Bidirectional vs causal attention: the project's namesake mechanism,
  tested here for the first time. All benchmark causes precede symptoms, so
  backward-only attention may suffice. Decision gate: if causal ties bidi
  within 1 std, the bidirectional framing is dropped from the title.
  [pending: docs/results/2026-07-03-bidi-ablation.md]
- Object bias 0/4/8 across corpora: reported as an injected relational
  prior / oracle upper bound.
  [pending: docs/results/2026-07-03-object-bias-matrix.md]

## 6. Real-bug pilot

[pending: Task 10 — one OSS-Fuzz crash, timeboxed; outcome recorded
whichever way it lands: NICHE-CONFIRMED / REAL-DISTAL-WIN / JOIN-FAILED]

## 7. Limitations (verbatim honesty)

- The templated→real gap is unstarted; cross-program generalization is
  unmeasured (all corpora share one program family or one generator).
- The interesting regime (distal causes) has no automatic oracle, so
  scaling it means hand labeling — a data-acquisition wall, not a modeling
  gap.
- n ≈ 38–80 anomalous test traces per corpus; synthetic evaluation.
- Evaluated at toy scale (d_model 64, 2 layers, window 64).
- Supervised model vs unsupervised heuristics: the comparison is disclosed
  and favors the model, which still loses to the oracle rules.

## 8. Related work

- LogBERT / DeepLog: bidirectional encoders over logs are standard; the
  bidirectional framing is not novel.
- Abnar & Zuidema (2020) attention rollout: tried, abandoned for raw
  last-layer head-averaged attention.
- Spectrum-based FL (Ochiai, Tarantula): implemented as baselines; blind on
  our corpora because every function appears in both passing and failing
  traces (event-level attribution, not statement-level coverage).
