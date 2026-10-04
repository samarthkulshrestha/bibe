# Learned baselines (PyTorch)

Apples-to-apples learned competitors for the Rust model, on the same `.trace`
files, split, supervision, and metrics. Research iteration happens here, not
in the from-scratch Rust core (which is frozen as an artifact).

```bash
python3 baselines/lstm_attrib.py instrumentation/out/distal_v2
```

## Tuning

Short version: the LSTM baseline was not tuned. It is one configuration, run
once per seed. Anyone arguing the baseline was handicapped relative to the
Rust model will find the evidence for that here.

What is swept: the model init seed, over 7 / 42 / 99 / 1234 / 2025. That is
the only sweep.

What is fixed and never varied, all hard-coded at the top of
`lstm_attrib.py`: hidden size 64, 1 bidirectional LSTM layer, 30 epochs, Adam
at lr 1e-3, no LR schedule, no weight decay, no dropout, no gradient clipping.
Traces are fed whole (`WINDOW = 64` is a guard, not a crop: `load()` refuses
a corpus whose longest trace exceeds it). Batch size is 1: training loops one
trace at a time and steps the optimizer on each.

What does not exist: no validation split (the data is a sorted-path 80/20
train/test split and the test half is what gets reported, so there is no held-
out set to select on), no early stopping (always exactly 30 epochs), no
hyperparameter search of any kind. No search was run and none is recorded.

Undisclosed clamp, now disclosed: `LstmAttrib` allocates an object embedding
table of size `n_objects=16`, and `tensors()` maps every object id through
`min(o, 15)`. Any trace with more than 16 distinct object ids has its
high-numbered objects collapsed into one bucket. Whether any corpus actually
exceeds 16 distinct ids is not checked anywhere: the clamp is a silent cap,
not an asserted invariant. Unknown function names likewise fall back
to vocab index 0, which is `<PAD>`.

Environment: `torch` is imported with no version pin, and the repo has no
requirements file or lockfile. The torch version used for the published
numbers is not recorded. A 2026-10-04 re-run on torch 2.8.0 / macOS-arm64
moves distal v1 Hit@1 from 0.842 ± 0.117 to 0.847 ± 0.099; see
`docs/results/2026-07-03-lstm-baseline.md`.

How much tuning effort the Rust model received by comparison is not recorded.
