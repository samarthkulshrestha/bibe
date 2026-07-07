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
  before the crash — the recency-on-object heuristic scores Hit@1 = 1.000 on
  our real ASan-labeled corpus (82 anomalous test traces).
- The supervised model never matches it: 0.488 ± 0.145 without the object
  bias, 0.824 ± 0.247 at bias 4, 0.893 ± 0.125 at bias 8 — and the bias is a
  hand-injected same-object prior, i.e. the heuristic itself wired into
  attention, reported as an oracle upper bound, not a capability.
- Spectrum FL scores 0.000 here (every function appears in both classes);
  plain recency 0.378. (docs/results/2026-07-03-object-bias-matrix.md)

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
- v2 (gapped) removes the hidden token and breaks adjacency
  (`docs/results/2026-07-03-distal-v2-ladder.md`): trig-adjacent collapses
  to 0.026 ± 0.000; the new oracle trig-window is 1.000 ± 0.000 (as any
  oracle must be on rule-labeled data); the transformer reaches
  0.877 ± 0.068 — much higher than on v1 because the trigger now carries
  the object id and is visible to the same-object attention prior — and
  comparable to the bi-LSTM (0.856 ± 0.113). Recency-family and spectrum
  baselines stay ≈ 0 at Hit@1.
- The observation that survives: cause supervision alone recovers a planted
  relational pattern far above the recency family and spectrum FL — but
  never above the oracle rule, and (on v1) below a plain bi-LSTM. On v2 the
  two learned models tie within noise, which is itself evidence that the
  bespoke architecture adds nothing over a generic sequence model.

## 5. Ablations

- Bidirectional vs causal attention (the namesake mechanism, tested for the
  first time): **bidirectionality does not help — the causal model is better
  on the mean** (bidi Hit@1 0.537 ± 0.118 vs causal 0.805 ± 0.235; stds
  overlap, so we claim only "not better", not "worse"). All benchmark causes
  precede symptoms, so backward-only attention suffices and the forward half
  appears to add noise. **The decision gate fired: "bidirectional" is
  dropped from the title and claims.** The future-events motivation remains
  an untested hypothesis no current benchmark exercises.
  (docs/results/2026-07-03-bidi-ablation.md)
- Object bias 0/4/8 across corpora (the injected relational prior),
  model Hit@1:

  | corpus     | bias 0        | bias 4        | bias 8        | best rule |
  |------------|---------------|---------------|---------------|-----------|
  | UAF real   | 0.488 ± 0.145 | 0.824 ± 0.247 | 0.893 ± 0.125 | 1.000     |
  | distal v1  | 0.342 ± 0.160 | 0.537 ± 0.118 | 0.463 ± 0.209 | 1.000     |
  | distal v2  | 0.441 ± 0.139 | 0.877 ± 0.068 | 0.913 ± 0.070 | 1.000     |

  The prior contributes +0.34–0.47 Hit@1 where it aligns with the label
  structure, *hurts* where it doesn't (v1, whose marker token carries no
  object id), and the model never reaches the definitional/oracle rule in
  any cell. (docs/results/2026-07-03-object-bias-matrix.md)

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
