#!/bin/sh
# Remaining object-bias-matrix arms, parallelized across independent work.
# Logs land in docs/results/matrix-logs/ — nothing to paste, the logs are the
# deliverable. Safe to re-run; every arm is deterministic (fixed seeds).
set -eu
cd "$(dirname "$0")/.."
L=docs/results/matrix-logs
mkdir -p "$L"
T=./target/release/examples/train_real
cargo build --release --example train_real --example corpus_gen 2>/dev/null

# Stage 1 (parallel): bias-8 on distal v2 || UAF corpus regen + ASan capture
$T instrumentation/out/distal_v2 raw 8 bidi > "$L/bias8-v2.log" 2>&1 &
P1=$!
(
  cargo run --release --example corpus_gen -- 800 instrumentation/out/corpus 99
  sh instrumentation/capture_corpus.sh instrumentation/out/corpus instrumentation/out/traces
) > "$L/uaf-gen.log" 2>&1
wait $P1

# Stage 2 (parallel): the three UAF arms
for bias in 0 4 8; do
  $T instrumentation/out/traces raw $bias bidi > "$L/bias$bias-uaf.log" 2>&1 &
done
wait
echo "ALL ARMS DONE — aggregates:"
grep -A40 "aggregate over" "$L"/bias*.log | grep -E "log|model_hit1"
