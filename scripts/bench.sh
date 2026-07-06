#!/bin/sh
# Canonical BiBE benchmark: two corpora only.
#   1. UAF (real ASan-labeled executions of templated C) — negative control.
#   2. Distal v2 (gapped synthetic) — capability probe, full rule ladder.
# Regenerates each corpus for several data seeds; train_real itself averages
# over 5 model seeds. All output logged under docs/results/bench-<date>/.
set -eu
cd "$(dirname "$0")/.."
OUT=${1:-docs/results/bench-$(date +%Y-%m-%d)}
mkdir -p "$OUT"

for dseed in 1 2 3; do
  # UAF negative control
  cargo run --release --example corpus_gen -- 800 instrumentation/out/corpus "$dseed"
  sh instrumentation/capture_corpus.sh instrumentation/out/corpus instrumentation/out/traces
  cargo run --release --example train_real -- instrumentation/out/traces \
    | tee "$OUT/uaf-dseed$dseed.log"

  # Distal v2 capability probe
  cargo run --release --example synth_distal_gen -- 400 instrumentation/out/distal_v2 "$dseed" gapped
  cargo run --release --example train_real -- instrumentation/out/distal_v2 \
    | tee "$OUT/distal-v2-dseed$dseed.log"
done
echo "logs in $OUT"
