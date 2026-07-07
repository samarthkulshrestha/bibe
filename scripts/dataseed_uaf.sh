#!/bin/sh
# Data-seed variance + de-biased eval on the real UAF corpus.
# ASan captures are sequential (clang builds); model runs are parallelized
# per (dseed, bias/arm, mseed). Reports attrib_dropped (L2) from the fixed
# crash-window eval.
set -eu
cd "$(dirname "$0")/.."
L=docs/results/dataseed-logs
mkdir -p "$L"
T=./target/release/examples/train_real
cargo build --release --example train_real --example corpus_gen --example trace_convert 2>/dev/null

# Sequential ASan captures, one corpus per data seed.
for dseed in 99 2 3; do
  cargo run --release --example corpus_gen -- 800 instrumentation/out/corpus "$dseed" >/dev/null 2>&1
  sh instrumentation/capture_corpus.sh instrumentation/out/corpus "instrumentation/out/uaf_$dseed" \
    > "$L/uaf-capture-d$dseed.log" 2>&1
done

# Parallel model runs. dseed 99 (canonical): bias 0/4/8 + causal. dseeds 2,3: bias 4 only.
for mseed in 7 42 99 1234 2025; do
  for bias in 0 4 8; do
    $T instrumentation/out/uaf_99 raw $bias bidi "$mseed" > "$L/uaf-d99-b$bias-m$mseed.log" 2>&1 &
  done
  $T instrumentation/out/uaf_99 raw 4 causal "$mseed" > "$L/uaf-d99-causal-m$mseed.log" 2>&1 &
  $T instrumentation/out/uaf_2 raw 4 bidi "$mseed" > "$L/uaf-d2-b4-m$mseed.log" 2>&1 &
  $T instrumentation/out/uaf_3 raw 4 bidi "$mseed" > "$L/uaf-d3-b4-m$mseed.log" 2>&1 &
  wait   # cap concurrency at ~6 per wave
done
echo "uaf data-seed sweep done"
