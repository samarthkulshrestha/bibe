# BiBE: Bidirectional Bug Exorcist

A transformer, written from scratch in Rust with no ML framework, that reads
program execution traces and tries to name the earlier event that caused a
crash.

**Read the results section before the feature list.** The headline finding is
negative: on the one bug class with an automatic ground-truth oracle, a
one-line heuristic beats the learned model, and the bidirectional attention
the project is named after does not help on any benchmark here. The name
predates that ablation.

## The problem

Traditional debugging tools show you *where* a crash happened, not always
*why*. A segfault at line 1000 may be caused by an allocation at line 200 and
a `free()` at line 800; a deadlock, by a lock order established much earlier.
Finding that link by hand means reading thousands of trace events.

BiBE tries to do it by learning: flag the anomalous event, then use attention
to point back at the event that explains it.

## How it works

BiBE reads execution traces captured by instrumentation or profiling tools
(`perf`, or the `-finstrument-functions` shim in `instrumentation/`): which
functions were called, when, at what call depth, with performance counters
alongside. A transformer over that sequence learns normal vs. buggy
execution, flags the suspicious event, and attention rollout ranks the
earlier events that explain it.

**Original hypothesis, now ablated.** BiBE attends *both* forward and backward in execution traces, on the theory that a crash can be explained by events that happen *after* it (like a deallocation that should have happened earlier). The ablation did not support this: a backward-only (causal) model matches or beats the bidirectional one on every current benchmark, because observed causes precede their symptoms (`docs/results/2026-07-03-bidi-ablation.md`). The forward-attention case remains an untested hypothesis that no current benchmark exercises.

## Current Status

The full system is implemented from scratch in Rust and trains end-to-end. What works today:

- **Numerical core**: dense tensors, broadcasting, matmul, numerically stable softmax/log-sum-exp.
- **Autograd**: reverse-mode automatic differentiation with finite-difference gradient checks on every operation.
- **Model**: multi-head attention (bidirectional or causal, selectable; causal is the stronger setting on every benchmark here), pre-LayerNorm transformer blocks, embeddings, sinusoidal positional encodings, a per-event anomaly head, and attention-rollout attribution.
- **Training**: Adam, warmup + cosine learning-rate schedule, gradient clipping, focal / contrastive / attention-sparsity / attribution-supervision losses, and parameter checkpointing.
- **Data**: a trace format with parser/serializer, vocabulary, sliding windows, batching, a synthetic trace generator, and a **real-trace capture pipeline** (instrument C programs, run them, and label bugs automatically with AddressSanitizer).
- **Evaluation**: AUC-ROC, Precision@K, Hit@K, MRR for detection, localization, and attribution.

### Results so far

**Finding 1 (negative): simple heuristics solve sanitizer-catchable attribution.**
On use-after-free, the one bug class with an automatic oracle (ASan), the
cause is *definitionally* the most-recent same-object event before the crash,
so the one-line heuristic "attribute to the most-recent same-object event"
scores Hit@1 = 1.0. The learned model scores ≈ 0.585, and reaches ≈ 0.99 only
when the same-object heuristic is hand-injected into attention as an additive
`object_bias`: an oracle prior wired in by hand, not a learned capability.
ML adds no value over a trivial rule on UAF; UAF serves as a negative control.

**Finding 2 (capability probe): cause-supervised attention partially recovers
a planted relational pattern.** On synthetic distal-cause traces
(`examples/synth_distal_gen.rs`), the model reaches Hit@1 = 0.537 ± 0.118
(5 seeds) while recency-family baselines score 0.0 to 0.29. But the generator's
own oracle rule ("the same-object write immediately preceded by a `trigger`",
`trig-adjacent` in `train_real.rs`) scores 1.000 ± 0.000, as any oracle rule
must on rule-labeled synthetic data. The honest reading: the model partially
learns a relational pattern from cause supervision alone, and never beats the
best hand-coded rule. Detection (AUC ≈ 1.0) and localization (Hit@1 ≈ 1.0)
are solved, but were never the hard part.

Evaluated config: d_model 64, 4 heads, 2 layers, window 64, 240 to 800 traces,
smaller than the design targets. The traces are real executions of small
*templated* programs; generalization to real applications is untested and is
the main open question. Full baseline ladders and per-seed variance live in
`docs/results/`.

## Prior work

Learned anomaly detection over execution and log streams is not new.
[DeepLog](https://doi.org/10.1145/3133956.3134015) (CCS 2017) runs an LSTM
over log templates; [LogBERT](https://arxiv.org/abs/2103.04475) uses a masked
transformer for the same. Spectrum-based fault localization
([Tarantula](https://doi.org/10.1145/1101908.1101949),
[Ochiai](https://doi.org/10.1109/TAIC.PART.2007.13)) and delta debugging
attack root-cause attribution without learning at all, and for the bug class
measured here they win. BiBE differs in operating on function-level execution
traces with per-event cause supervision and attention-rollout attribution,
and in reporting where that fails.

## Why from scratch

No PyTorch, no `candle`, no `burn`. Two dependencies total (`rand`,
`rand_distr`). Attention weights are the output being studied, so there is
something to be said for owning every line between the trace and the weight,
and for not having a framework's fused kernels in the way when a gradient
check disagrees. The honest other half: this was also a way to learn the
machinery by building it.

## Getting Started

### Prerequisites
- Rust toolchain (edition 2024 or later)
- Cargo package manager

### Build and Test
```bash
cargo build
cargo test          # 450 tests, including finite-difference gradient checks
```

### Run the experiments
```bash
# The two canonical benchmarks (UAF negative control + distal v2 capability
# probe), all baselines, 3 data seeds x 5 model seeds, logged to docs/results/
sh scripts/bench.sh

# Individual pieces:
cargo run --release --example train       # synthetic demo + metrics
cargo run --release --example ood_study   # leave-one-out generalization study
cargo run --release --example train_real -- <traces_dir> [raw|rollout|margin] [object_bias] [bidi|causal] [seeds_csv]
python3 baselines/lstm_attrib.py <traces_dir>   # learned LSTM baseline
```
(AddressSanitizer requires a `clang` toolchain.)

## Project Structure

- `src/tensor`, `src/autograd` - numerical core and automatic differentiation
- `src/nn`, `src/attention`, `src/transformer` - model layers and the encoder
- `src/data` - trace format, vocabulary, windowing, batching, synthetic generator
- `src/optim`, `src/train` - optimizer, schedule, losses, training loop, checkpoints
- `src/eval` - detection and attribution metrics
- `src/model.rs` - the assembled BiBE model
- `examples/` - runnable training, study, and capture-conversion programs
- `instrumentation/` - C instrumentation shim, sample programs, capture scripts
- `baselines/` - PyTorch bi-LSTM attribution baseline

## Reproducing the real-trace pilot

One real crash, [mjs issue #322](https://github.com/cesanta/mjs) (heap
use-after-free, CWE-416), captured through the pipeline and checked in at
`instrumentation/real/poc.trace`. Full capture steps in
`instrumentation/real/README.md`.

```
$ python3 instrumentation/real/pilot_analyze.py
positional recency rank: 24
most-recent 'free' substring rank: 1
vocab size: 178 | total events: 12213
```

The naive recency baseline puts the true cause 24th; a one-line "most recent
free-shaped call" rule puts it 1st. Same conclusion as the synthetic UAF
benchmark, now on code we did not write.

## License

MIT. See `LICENSE`.
