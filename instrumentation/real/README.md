# Real-bug pilot (plan Task 10)

One real crash from a program we did not write, run through the *existing*
BiBE capture pipeline, to test whether the synthetic findings hold on real
data. Timeboxed; outcome recorded either way.

## Target

- **Program:** [mjs](https://github.com/cesanta/mjs) — Cesanta's embedded
  JavaScript engine, single-file `mjs.c` (~14k lines).
- **Bug:** heap-use-after-free in `mjs_next()` when an array is spliced during
  `for-in` iteration (GitHub issue #322, CWE-416). The iterator holds a raw
  pointer to a GC-managed property cell; `splice()` makes it unreachable,
  `gc_sweep()` frees it, the next `mjs_next()` reads freed memory.
- **Why chosen:** public PoC + full ASan output (free oracle, skips the hard
  labeling), single-file C (easy to instrument), builds on macOS/clang.

## Reproduce

```sh
cd instrumentation/real/mjs   # (git clone --depth 1 https://github.com/cesanta/mjs)
printf 'let a=[]; for(let i=0;i<10;i++) a.push(i); for(let k in a){a.splice(0,1);}\n' > poc.js
# ASan-only sanity check (crashes):
clang -g -O1 -fsanitize=address -fno-omit-frame-pointer -DMJS_MAIN -o mjs_asan mjs.c -ldl -lm
./mjs_asan poc.js
# Instrumented capture:
clang -O1 -c ../../trace_shim.c -o trace_shim.o
clang -fsanitize=address -finstrument-functions -g -O0 -DMJS_MAIN mjs.c trace_shim.o -ldl -lm -o mjs_traced
BIBE_TRACE=poc.log ./mjs_traced poc.js >/dev/null 2>poc.asan
cd ../../.. && ./target/release/examples/trace_convert \
  instrumentation/real/mjs/poc.log instrumentation/real/mjs/poc.asan instrumentation/real/mjs/poc.trace
python3 "$CLAUDE_JOB_DIR/tmp/pilot_analyze.py"   # or see docs/results
```

The mjs clone and build products are git-ignored; only `poc.trace` (the one
labeled real artifact) and this README are tracked.

## Outcome: NICHE-CONFIRMED

The ASan→event-index join **succeeded** (no JOIN-FAILED): symptom `mjs_next`
#12211, cause `gc_free_block` #12188, in a 12,213-event, 178-function trace.
The cause is genuinely distal — 23 events (a full `gc_sweep` plus interpreter
stack churn) separate it from the crash.

On this real trace:
- domain-agnostic **positional recency ranks the true cause 24th** (Hit@1 = 0)
  — the naive baseline fails on a real distal cause, exactly as on the
  synthetic benchmarks;
- a one-line **"most-recent deallocation-shaped call" heuristic gets Hit@1 =
  1.0**, because `gc_free_block` is the unique free before the crash.

So the heuristic solves it and ML is unnecessary — the same story as UAF on
the synthetic side, now on real code. The regime where learning *might* win
(several candidate frees, ambiguous which one is causal) does not even arise
in this single PoC.

**Honest limitation (the data wall, made concrete):** we could not train or
evaluate the learned model here. One real trace is not a trainable corpus,
and the crash sits in the last of 191 windows while `train_real` evaluates
only the first window. Getting a real *learned* result needs many labeled
real traces — and the interesting multi-free regime needs bugs sanitizers
can't auto-label. That is the budget-hawk's data-acquisition wall, confirmed
on contact with real software.
