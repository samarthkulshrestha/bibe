# BiBE: Bidirectional Bug Exorcist

A machine learning system that analyzes program execution traces to find bugs and explain what caused them.

## What is BiBE?

When programs crash or behave incorrectly, understanding *why* often requires manually digging through thousands of lines of execution logs. BiBE automates this process by using a transformer neural network (similar to ChatGPT's architecture) to:

1. **Detect anomalies** in execution traces (crashes, deadlocks, performance issues)
2. **Identify root causes** by tracing back through the execution history to find which earlier events led to the problem

## The Problem

Traditional debugging tools show you *where* a crash happened, but not always *why*. For example:
- A segfault at line 1000 might be caused by a memory allocation at line 200 and a free() at line 800
- A deadlock might be caused by locks acquired in a specific order across multiple threads
- A performance regression might stem from cache-unfriendly memory access patterns earlier in execution

Finding these causal relationships manually is time-consuming and error-prone.

## How BiBE Works

BiBE reads execution traces captured by profiling tools (like `perf`) that record:
- Which functions were called
- When they were called
- Performance counters (cache misses, branch mispredictions, etc.)

It then uses a custom-built transformer model to:
1. Learn patterns of normal vs. buggy execution
2. Flag suspicious events in new traces
3. Use attention mechanisms to show which earlier events are causally related to the bug

**Original hypothesis (now ablated)**: BiBE attends *both* forward and backward in execution traces, on the theory that a crash can be explained by events that happen *after* it (like a deallocation that should have happened earlier). The ablation did not support this: a backward-only (causal) model matches or beats the bidirectional one on every current benchmark, because observed causes precede their symptoms (`docs/results/2026-07-03-bidi-ablation.md`). The forward-attention case remains an untested hypothesis that no current benchmark exercises.

## Current Status

The full system is implemented from scratch in Rust and trains end-to-end. What works today:

- **Numerical core**: dense tensors, broadcasting, matmul, numerically stable softmax/log-sum-exp.
- **Autograd**: reverse-mode automatic differentiation with finite-difference gradient checks on every operation.
- **Model**: multi-head bidirectional attention, pre-LayerNorm transformer blocks, embeddings, sinusoidal positional encodings, a per-event anomaly head, and attention-rollout attribution.
- **Training**: Adam, warmup + cosine learning-rate schedule, gradient clipping, focal / contrastive / attention-sparsity / attribution-supervision losses, and parameter checkpointing.
- **Data**: a trace format with parser/serializer, vocabulary, sliding windows, batching, a synthetic trace generator, and a **real-trace capture pipeline** (instrument C programs, run them, and label bugs automatically with AddressSanitizer).
- **Evaluation**: AUC-ROC, Precision@K, Hit@K, MRR for detection, localization, and attribution.

### Results so far (honest)

**Finding 1 (negative): simple heuristics solve sanitizer-catchable attribution.**
On use-after-free — the one bug class with an automatic oracle (ASan) — the
cause is *definitionally* the most-recent same-object event before the crash,
so the one-line heuristic "attribute to the most-recent same-object event"
scores Hit@1 = 1.0. The learned model scores ≈ 0.585, and reaches ≈ 0.99 only
when the same-object heuristic is hand-injected into attention as an additive
`object_bias` — an oracle prior wired in by hand, not a learned capability.
ML adds no value over a trivial rule on UAF; UAF serves as a negative control.

**Finding 2 (capability probe): cause-supervised attention partially recovers
a planted relational pattern.** On synthetic distal-cause traces
(`examples/synth_distal_gen.rs`), the model reaches Hit@1 = 0.537 ± 0.118
(5 seeds) while recency-family baselines score 0.0–0.29 — but the generator's
own oracle rule ("the same-object write immediately preceded by a `trigger`",
`trig-adjacent` in `train_real.rs`) scores 1.000 ± 0.000, as any oracle rule
must on rule-labeled synthetic data. The honest reading: the model partially
learns a relational pattern from cause supervision alone, and never beats the
best hand-coded rule. Detection (AUC ≈ 1.0) and localization (Hit@1 ≈ 1.0)
are solved, but were never the hard part.

Evaluated config: d_model 64, 4 heads, 2 layers, window 64, 240–800 traces —
smaller than the design targets elsewhere in this README. The traces are real
executions of small *templated* programs; generalization to real applications
is untested and is the main open question. Full baseline ladders and per-seed
variance live in `docs/results/`.

## Why Build From Scratch?

Rather than using existing ML frameworks (PyTorch, TensorFlow), BiBE is implemented from the ground up in Rust to:
- Ensure complete understanding of the numerical stability requirements
- Make attention weights fully interpretable and trustworthy
- Optimize specifically for execution trace analysis
- Learn the fundamentals deeply rather than treating ML as a black box

## Getting Started

### Prerequisites
- Rust toolchain (edition 2024 or later)
- Cargo package manager

### Build and Test
```bash
cargo build
cargo test          # ~430 tests, including finite-difference gradient checks
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
- `PLAN.md` - technical design document

## Goals

The ultimate goal is to create a tool that, given an execution trace from a crashed program, can:
1. Highlight the exact event where things went wrong
2. Show the chain of events that led to it
3. Provide interpretable explanations backed by attention weights

This would significantly speed up debugging complex systems issues, especially in large codebases where manual trace analysis is impractical.

## License

To be determined.
