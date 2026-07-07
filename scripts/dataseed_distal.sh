#!/bin/sh
# Data-seed variance on distal v2 (methodologist L1): regenerate the corpus at
# 3 generator seeds, run 5 model seeds each, one process per (dseed, mseed).
# Also runs the causal arm at dseed 1 for the v2 bidirectionality ablation.
set -eu
cd "$(dirname "$0")/.."
L=docs/results/dataseed-logs
mkdir -p "$L"
T=./target/release/examples/train_real
cargo build --release --example train_real --example synth_distal_gen 2>/dev/null

for dseed in 1 2 3; do
  ./target/release/examples/synth_distal_gen 400 "instrumentation/out/dv2_$dseed" "$dseed" gapped >/dev/null
done

# 15 model runs (3 dseed x 5 mseed), bidi bias 4, in parallel.
for dseed in 1 2 3; do
  for mseed in 7 42 99 1234 2025; do
    $T "instrumentation/out/dv2_$dseed" raw 4 bidi "$mseed" \
      > "$L/v2-d$dseed-m$mseed.log" 2>&1 &
  done
done
# causal arm at dseed 1 for the ablation
for mseed in 7 42 99 1234 2025; do
  $T instrumentation/out/dv2_1 raw 4 causal "$mseed" \
    > "$L/v2causal-d1-m$mseed.log" 2>&1 &
done
wait
echo "distal data-seed sweep done"
