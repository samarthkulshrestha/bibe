# Learned baselines (PyTorch)

Apples-to-apples learned competitors for the Rust model, on the same `.trace`
files, split, supervision, and metrics. Research iteration happens here, not
in the from-scratch Rust core (which is frozen as an artifact).

```bash
python3 baselines/lstm_attrib.py instrumentation/out/distal_v2
```
