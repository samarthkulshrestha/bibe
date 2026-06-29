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

**Key insight**: Unlike standard transformers that only look backward, BiBE looks *both* forward and backward in execution traces. This is crucial because sometimes a crash is explained by events that happen *after* a problematic allocation (like a deallocation that should have happened before).

## Current Status

The full system is implemented from scratch in Rust and trains end-to-end. What works today:

- **Numerical core**: dense tensors, broadcasting, matmul, numerically stable softmax/log-sum-exp.
- **Autograd**: reverse-mode automatic differentiation with finite-difference gradient checks on every operation.
- **Model**: multi-head bidirectional attention, pre-LayerNorm transformer blocks, embeddings, sinusoidal positional encodings, a per-event anomaly head, and attention-rollout attribution.
- **Training**: Adam, warmup + cosine learning-rate schedule, gradient clipping, focal / contrastive / attention-sparsity / attribution-supervision losses, and parameter checkpointing.
- **Data**: a trace format with parser/serializer, vocabulary, sliding windows, batching, a synthetic trace generator, and a **real-trace capture pipeline** (instrument C programs, run them, and label bugs automatically with AddressSanitizer).
- **Evaluation**: AUC-ROC, Precision@K, Hit@K, MRR for detection, localization, and attribution.

### Results so far (honest)

On a **real** corpus — C programs compiled with AddressSanitizer + function instrumentation, executed, and auto-labeled — where clean and buggy programs share the same set of `free`/`use` calls (so only execution *order* differs) and contain decoy frees:

- **Detection** (is there a use-after-free?): AUC-ROC ≈ 0.997
- **Localization** (which event is the symptom?): Hit@1 ≈ 0.96
- **Attribution** (which earlier `free` caused it, among decoys?): the true cause is always in the top 3 (Hit@3 = 1.0), and — once events carry an object identity linking each use to its allocation — the exact culprit is ranked first ~63% of the time (up from ~46% without it).

Detection and localization are strong; **pinpointing the exact root cause among decoys is improving but unsolved.** Giving the model an object identity that ties a use to its free — derived from the real allocation address captured at runtime — was the change that moved attribution, confirming the bottleneck is information, not the loss. (Linking the allocation too makes it a same-object competitor and *hurts*, so only the dealloc/use are linked.) The traces are real executions, but of small templated programs — generalization to real applications is future work.

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
# Train on synthetic traces and report detection/localization/attribution
cargo run --release --example train

# Leave-one-out generalization study + contrastive-weight sweep
cargo run --release --example ood_study

# Capture REAL traces: instrument C programs, run them, label with ASan
cargo run --example corpus_gen -- 240 instrumentation/out/corpus 99
sh instrumentation/capture_corpus.sh instrumentation/out/corpus instrumentation/out/traces

# Train and evaluate on the real captured traces
cargo run --release --example train_real -- instrumentation/out/traces
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
