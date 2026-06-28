#!/bin/sh
# Build the sample programs with AddressSanitizer + function instrumentation,
# run them to capture a function-call log and ASan report, then convert each to
# a BiBE trace through the existing pipeline.
set -e

DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$DIR/.." && pwd)
OUT="$DIR/out"
mkdir -p "$OUT"
CC=${CC:-clang}

# Shim is compiled WITHOUT instrumentation so it does not trace itself.
$CC -O1 -c "$DIR/trace_shim.c" -o "$OUT/trace_shim.o"

for prog in clean uaf; do
    $CC -fsanitize=address -finstrument-functions -g -O0 -c "$DIR/$prog.c" -o "$OUT/$prog.o"
    $CC -fsanitize=address "$OUT/$prog.o" "$OUT/trace_shim.o" -o "$OUT/$prog"
    # ASan aborts the buggy program with a non-zero exit; capture and continue.
    BIBE_TRACE="$OUT/$prog.log" "$OUT/$prog" >/dev/null 2>"$OUT/$prog.asan" || true
    ( cd "$ROOT" && cargo run --quiet --example trace_convert -- \
        "$OUT/$prog.log" "$OUT/$prog.asan" "$OUT/$prog.trace" )
done

echo
echo "captured use-after-free trace:"
cat "$OUT/uaf.trace"
