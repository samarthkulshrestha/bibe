#!/bin/sh
# Compile, run, and convert a directory of C programs into BiBE traces.
# AddressSanitizer labels each run automatically (quiet -> normal, heap error
# -> anomalous with symptom/cause).
#
#   capture_corpus.sh <src_dir> <out_dir>
set -e

DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$DIR/.." && pwd)
SRC="$1"
OUT="$2"
[ -n "$SRC" ] && [ -n "$OUT" ] || { echo "usage: capture_corpus.sh <src_dir> <out_dir>"; exit 2; }
mkdir -p "$OUT"
CC=${CC:-clang}

$CC -O1 -c "$DIR/trace_shim.c" -o "$OUT/trace_shim.o"
( cd "$ROOT" && cargo build --quiet --release --example trace_convert )
CONV="$ROOT/target/release/examples/trace_convert"

# Build+run+convert each program. Independent per file, so fan out across
# cores: JOBS parallel workers (default = CPU count) instead of one clang at a
# time. This is the capture bottleneck — serial, it barely uses one core.
JOBS=${JOBS:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) )}
export CC OUT CONV

# Each worker gets one .c path in $1; CC/OUT/CONV come from the environment.
ls "$SRC"/*.c | xargs -P "$JOBS" -I {} sh -c '
    c="$1"; base=$(basename "$c" .c)
    "$CC" -fsanitize=address -finstrument-functions -g -O0 "$c" "$OUT/trace_shim.o" -o "$OUT/$base"
    BIBE_TRACE="$OUT/$base.log" "$OUT/$base" >/dev/null 2>"$OUT/$base.asan" || true
    "$CONV" "$OUT/$base.log" "$OUT/$base.asan" "$OUT/$base.trace" >/dev/null
' _ {}

n=$(ls "$SRC"/*.c | wc -l | tr -d ' ')
anom=$(grep -l "label=anomalous" "$OUT"/*.trace 2>/dev/null | wc -l | tr -d ' ')
echo "captured $n traces into $OUT ($anom anomalous, $JOBS-way parallel)"
