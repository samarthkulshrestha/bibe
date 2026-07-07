# When Does Learned Root-Cause Attribution Beat a One-Line Heuristic?

> Draft status: OUTLINE + verified numbers. Every number in this note is
> copied, with its ± std, from a file in `docs/results/`. The ± are over 5
> model-init seeds on a **single corpus realization** (one data seed); data-
> seed variance is not yet reported (see Limitations L1).

**Claim (the only claim):** For bug classes with an automatic oracle
(sanitizer-catchable), the causal event is definitionally the most-recent
related event, so a one-line heuristic achieves Hit@1 = 1.0 and learned
attribution adds nothing. On synthetic *distal* causes — where recency fails
— a learned model recovers the planted relational signal far above recency,
**but never beats the generator's oracle rule and never beats a plain
bi-LSTM**; our bespoke attention architecture earns nothing over a generic
sequence model. We publish this boundary, not a victory.

**What we do NOT claim:** (a) that ML is useless for debugging in general;
(b) that this architecture is good — the ablations say it is not; (c) that
heuristics solve bugs without an automatic oracle — the interesting regime is
exactly the one with no oracle, which we could not reach on real code.

## 1. Introduction — the negative result is the finding

- Debugging tools locate symptoms; root-cause attribution is the hard part.
- The tempting ML story: train an attention model over execution traces,
  read causes off the attention map.
- What we found instead, in order: (a) on the one bug class with a free
  oracle (UAF via ASan), a one-line heuristic is perfect and the learned
  model is not; (b) on a synthetic benchmark built so recency heuristics
  fail, the generator's own oracle rule is perfect and the learned model is
  not; (c) a plain bi-LSTM matches or beats our bespoke attention model on
  that same benchmark; (d) on a real bug the labeling pipeline reaches, the
  heuristic still wins and the learned model could not even be run. We
  publish the boundary, not a victory.

### Summary of all results (Hit@1, mean ± std over 5 seeds)

| corpus | recency | best domain rule | bi-LSTM | our transformer | oracle rule |
|---|---|---|---|---|---|
| UAF (real ASan) | 0.378 | **1.000** (same-obj recency, definitional) | — | 0.488 (bias 0) / 0.893 (bias 8) | 1.000 |
| distal v1 (adjacent) | 0.000 | 0.289 (same-obj write) | 0.842 ± 0.117 | 0.537 ± 0.118 | **1.000** (trig-adjacent) |
| distal v2 (gapped) | 0.000 | 0.333 (same-obj write) | 0.856 ± 0.113 | 0.441 (bias 0) / 0.877 (bias 4) | **1.000** (trig-window) |
| mjs (real UAF, n=1) | 0 (rank 24) | **1.000** (most-recent free) | — | not runnable | — |

The learned transformer wins no cell: it loses to the oracle everywhere,
loses to the bi-LSTM on v1, and only ties it on v2 — and that tie is the
transformer *with* a hand-injected object prior (bias 4) against an LSTM
without one. The only thing learning beats is raw recency. (dashes: LSTM not
run on the real UAF corpus; single-free real trace has no rankable candidate
set.)

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

Primary benchmark is **v2 (gapped)**, from
`docs/results/2026-07-03-distal-v2-ladder.md`. v2 fixes the flaws of our
earlier v1 (see the sidebar): the `trigger` marker carries the real object id
(visible to same-object baselines — nothing hidden), and a verified ≥2-event
gap separates it from the causal write, so no adjacency rule works. Every
number below is model-supervised at object_bias 4:

| rule / model            | Hit@1         | Hit@3         | MRR           |
|-------------------------|---------------|---------------|---------------|
| recency                 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.113 ± 0.000 |
| same-obj recency        | 0.000 ± 0.000 | 0.538 ± 0.000 | 0.295 ± 0.000 |
| same-obj write          | 0.333 ± 0.000 | 1.000 ± 0.000 | 0.658 ± 0.000 |
| Ochiai FL               | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.113 ± 0.000 |
| Tarantula FL            | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.113 ± 0.000 |
| trig-adjacent (v1 oracle, now dead) | 0.026 ± 0.000 | 0.026 ± 0.000 | 0.137 ± 0.000 |
| **trig-window (oracle)** | **1.000 ± 0.000** | 1.000 ± 0.000 | 1.000 ± 0.000 |
| bi-LSTM (supervised)    | 0.856 ± 0.113 | 0.964 ± 0.038 | 0.913 ± 0.068 |
| transformer (supervised, bias 4) | 0.877 ± 0.068 | 0.985 ± 0.031 | 0.929 ± 0.044 |

- **Rule-labeled synthetic data always has a perfect oracle rule** by
  construction — here `trig-window` ("first same-object write after the
  same-object trigger") at 1.000. A strictly simpler rule also solves it —
  "the second same-object write, in order" — which we note to be explicit
  that no single hand rule is hard here; the benchmark is a capability probe,
  not proof the model beats hand-coded rules.
- The transformer (0.877 ± 0.068) and the bi-LSTM (0.856 ± 0.113) **tie
  within noise** — a ~1-trace difference on ~40 test traces. Since the
  transformer additionally carries a hand-injected object prior (bias 4) that
  the LSTM lacks, the tie is evidence the bespoke architecture adds nothing
  over a generic sequence model. Recency-family and spectrum-FL baselines
  stay ≈ 0 at Hit@1 (spectrum FL is a sanity floor, not a real competitor —
  it ranks functions by cross-trace coverage and every function appears in
  both classes here).
- What survives: cause supervision recovers the planted relational signal far
  above recency — but never above the oracle rule, and never above the LSTM.

> **Sidebar — v1 (adjacent), a retracted self-refereed benchmark.** Our first
> distal generator (`docs/results/2026-07-03-distal-v1-oracle.md`) emitted the
> `trigger` and causal write as one atomic step, so the cause was always
> trigger-adjacent, *and* the `trigger` carried object id 0 — invisible to the
> same-object baselines we compared against. We originally claimed the model
> (0.537 ± 0.118) beat "the best hand-coded rule" (same-obj write 0.289). It
> did not: the adjacency oracle `trig-adjacent` scores **1.000 ± 0.000** and
> was simply never implemented. On v1 the bi-LSTM (0.842 ± 0.117) also beats
> the transformer outright. We keep v1 only as a cautionary example of how a
> self-constructed benchmark manufactures a positive result.

## 5. Ablations

- Bidirectional vs causal attention (the namesake mechanism, tested for the
  first time): **no benchmark here shows bidirectionality helping; on the
  mean it is worse** (bidi Hit@1 0.537 ± 0.118 vs causal 0.805 ± 0.235; the
  stds overlap, so this is "not better", not a significant loss). All
  benchmark causes precede symptoms, so backward-only attention suffices.
  That is enough to justify the decision gate: **"bidirectional" is dropped
  from the title and claims.** The future-events motivation remains an
  untested hypothesis no current benchmark exercises.
  (docs/results/2026-07-03-bidi-ablation.md)
- Object bias 0/4/8 across corpora (the injected relational prior),
  model Hit@1:

  | corpus     | bias 0        | bias 4        | bias 8        | best rule |
  |------------|---------------|---------------|---------------|-----------|
  | UAF real   | 0.488 ± 0.145 | 0.824 ± 0.247 | 0.893 ± 0.125 | 1.000     |
  | distal v1  | 0.342 ± 0.160 | 0.537 ± 0.118 | 0.463 ± 0.209 | 1.000     |
  | distal v2  | 0.441 ± 0.139 | 0.877 ± 0.068 | 0.913 ± 0.070 | 1.000     |

  The prior contributes +0.34–0.47 Hit@1 where it aligns with the label
  structure and *stops helping* where it doesn't (v1 bias 8 0.463 ± 0.209 vs
  bias 4 0.537 ± 0.118 — overlapping, so "no longer helps", not "hurts",
  consistent with the marker token carrying no object id). The model never
  reaches the definitional/oracle rule in any cell. Much of "the model's"
  score is this injected prior, not learning.
  (docs/results/2026-07-03-object-bias-matrix.md)

## 6. Real-bug pilot (mjs use-after-free): the join works, attribution is not yet evaluable

One real crash from a program we did not write: a heap-use-after-free in the
mjs JavaScript engine's `mjs_next()` (array `splice()` during `for-in`;
issue #322, CWE-416). Run through the *existing* capture pipeline
(`docs/results/2026-07-07-oss-fuzz-pilot.md`). This is a single trace (n=1);
we report it as a feasibility probe, not a measured result.

- **What is confirmed:** the ASan→event-index join works on real,
  un-generated code — symptom `mjs_next` #12211, cause `gc_free_block` #12188,
  in a 12,213-event, 178-function trace. This retires the "the join won't
  survive real code" risk. The cause is genuinely distal (23 events, a full
  `gc_sweep` + interpreter churn, separate it from the crash).
- **What is not evaluable:** attribution has no meaningful metric here.
  `gc_free_block` occurs exactly once, so "most-recent deallocation" wins over
  a candidate set of size one — trivially perfect and uninformative. We do
  *not* claim a heuristic "win" on this trace. Positional recency ranks the
  cause 24th, which only re-shows that domain-agnostic recency is not the
  right rule for a distal cause.
- **The learned model could not be run at all:** one trace is not a trainable
  corpus, and the crash sits in the last of 191 windows (our harness scores
  the first window only — see L2). The regime where learning might matter —
  many candidate frees, ambiguous which is causal — does not arise in this
  single PoC. Getting a real learned result needs many labeled real traces,
  and that is the data-acquisition wall this note keeps hitting.

## 7. Limitations (verbatim honesty)

- **L1 — single-corpus variance.** The ± std throughout is over 5 model-init
  seeds on *one* corpus realization and one train/test split. The `± 0.000`
  on deterministic baselines means "one draw", not "stable"; those rows would
  move under a different data seed. Data-seed variance (regenerate each corpus
  at ≥3 seeds; `scripts/bench.sh` is set up for it) is not yet reported. No
  significance test backs any inequality; the two we lean on (LSTM ≥
  transformer on v1; model < oracle) are large enough to be safe, the rest
  are stated as ties.
- **L2 — first-window selection.** The harness scores attribution only on the
  first 64-event window of each trace, so on the real (variable-length) UAF
  corpus the "82 anomalous test traces" are the crash-fits-in-window subset,
  not the full anomalous population — a selection effect on the absolute
  numbers (the model-vs-baseline comparison stays fair, both see the same
  retained set). The mjs pilot is the extreme case: its crash fell outside
  the first window and was dropped entirely.
- The templated→real gap is unstarted; cross-program generalization is
  unmeasured (all corpora share one program family or one generator).
- The interesting regime (distal causes) has no automatic oracle, so
  scaling it means hand labeling — a data-acquisition wall, not a modeling
  gap.
- n ≈ 38–84 anomalous test traces per corpus; synthetic evaluation.
- Evaluated at toy scale (d_model 64, 2 layers, window 64).
- Supervised model vs unsupervised heuristics: disclosed and it *favors* the
  model (it is supervised on the cause and, at bias > 0, handed the answer as
  an attention prior), which still loses to the oracle rules and ties/loses to
  the LSTM. The bi-LSTM trains single-objective (cause cross-entropy) while
  the transformer trains multi-objective; on the synthetic corpora both see
  whole traces (≤ 64 events), so windowing is not a confound there.

## 8. Related work

- LogBERT / DeepLog: bidirectional encoders over logs are standard; the
  bidirectional framing is not novel.
- Abnar & Zuidema (2020) attention rollout: tried, abandoned for raw
  last-layer head-averaged attention.
- Spectrum-based FL (Ochiai, Tarantula): implemented as baselines; blind on
  our corpora because every function appears in both passing and failing
  traces (event-level attribution, not statement-level coverage).
