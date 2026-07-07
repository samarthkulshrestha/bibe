# Real-bug pilot: mjs use-after-free (Task 10)

**Outcome: NICHE-CONFIRMED.** One real, genuinely-distal UAF in a real
program; the ASan→event join succeeded and a one-line heuristic solves the
attribution, so learning is unnecessary — the synthetic story, reproduced on
real code.

## Target

mjs (Cesanta embedded JS engine), heap-use-after-free in `mjs_next()` via
array `splice()` during `for-in` (GitHub issue #322, CWE-416). Public PoC:

```js
let a=[]; for(let i=0;i<10;i++) a.push(i); for(let k in a){a.splice(0,1);}
```

Captured with the existing pipeline (`-finstrument-functions` + ASan +
`trace_convert`). See `instrumentation/real/README.md` to reproduce.

## Trace

| property | value |
|---|---|
| events | 12,213 |
| distinct functions (vocab) | 178 |
| symptom | `mjs_next` #12211 |
| cause | `gc_free_block` #12188 |
| cause→symptom gap | 23 events (a full `gc_sweep` + interpreter stack ops) |
| `gc_free_block` occurrences | 1 (unique — labeling is not circular) |

## Attribution on the real trace

| rule | cause rank | Hit@1 |
|---|---|---|
| positional recency (domain-agnostic) | 24 / 12212 | 0.0 |
| most-recent "free"-family call (domain heuristic) | 1 | 1.0 |

The cause is 24th by raw recency — the naive baseline fails on a real distal
cause, matching the synthetic benchmarks. But `gc_free_block` is the only
dealloc-shaped call before the crash, so a trivial domain rule nails it.

## Reading

- The ASan→event join works on real, un-generated code (retires the
  JOIN-FAILED risk the plan flagged as the pilot's main threat).
- A distal cause in real software does **not** imply learning is needed: a
  one-line "most-recent deallocation" rule is perfect here. The learned model
  earns its keep only when *multiple* candidate frees make that rule
  ambiguous — a regime this single PoC does not exercise.
- **Data wall confirmed:** we could not run the learned model. One trace is
  not a trainable corpus, and the crash is in the last of 191 windows
  (`train_real` evaluates the first window only). A real learned result needs
  many labeled real traces; the interesting multi-free regime needs bugs with
  no automatic oracle. Concrete confirmation of the budget-hawk's critique.

## Bottom line for the paper

Section 6 reports this as: the pipeline reaches real bugs, the heuristic still
wins on the case we could label, and the path to a learned win runs straight
into the data-acquisition wall — which is the note's thesis, now with a real
data point rather than only synthetic ones.
