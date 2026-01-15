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

This project is in early development. The current focus is building the foundational components from scratch:
- Linear algebra and matrix operations
- Automatic differentiation (autograd) system
- Numerically stable attention mechanisms
- Training infrastructure

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

### Build and Run
```bash
# Build the project
cargo build

# Run tests
cargo test

# Run the project
cargo run
```

## Project Structure

- `src/` - Source code
- `PLAN.md` - Detailed technical design document
- `CLAUDE.md` - Developer documentation for AI assistants

## Goals

The ultimate goal is to create a tool that, given an execution trace from a crashed program, can:
1. Highlight the exact event where things went wrong
2. Show the chain of events that led to it
3. Provide interpretable explanations backed by attention weights

This would significantly speed up debugging complex systems issues, especially in large codebases where manual trace analysis is impractical.

## License

To be determined.
