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

n=0
for c in "$SRC"/*.c; do
    base=$(basename "$c" .c)
    $CC -fsanitize=address -finstrument-functions -g -O0 "$c" "$OUT/trace_shim.o" -o "$OUT/$base"
    BIBE_TRACE="$OUT/$base.log" "$OUT/$base" >/dev/null 2>"$OUT/$base.asan" || true
    "$CONV" "$OUT/$base.log" "$OUT/$base.asan" "$OUT/$base.trace" >/dev/null
    n=$((n + 1))
done

anom=$(grep -l "label=anomalous" "$OUT"/*.trace 2>/dev/null | wc -l | tr -d ' ')
echo "captured $n traces into $OUT ($anom anomalous)"
