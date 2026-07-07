#!/bin/sh
# Remaining matrix arms, one process per (arm, seed) — saturates an M-series
# CPU instead of running seeds sequentially. Results identical (fixed seeds);
# only wall-clock changes. Requires instrumentation/out/traces to exist
# (the UAF capture — already done if uaf-gen.log says "captured 800 traces").
set -eu
cd "$(dirname "$0")/.."
L=docs/results/matrix-logs/fast
mkdir -p "$L"
T=./target/release/examples/train_real
cargo build --release --example train_real 2>/dev/null

for seed in 7 42 99 1234 2025; do
  $T instrumentation/out/distal_v2 raw 8 bidi "$seed" > "$L/bias8-v2-s$seed.log" 2>&1 &
  $T instrumentation/out/traces   raw 0 bidi "$seed" > "$L/bias0-uaf-s$seed.log" 2>&1 &
  $T instrumentation/out/traces   raw 4 bidi "$seed" > "$L/bias4-uaf-s$seed.log" 2>&1 &
  $T instrumentation/out/traces   raw 8 bidi "$seed" > "$L/bias8-uaf-s$seed.log" 2>&1 &
done
wait

python3 - "$L" << 'EOF'
import glob, re, statistics, sys, collections
vals = collections.defaultdict(list)
for f in glob.glob(sys.argv[1] + "/*.log"):
    arm = re.sub(r"-s\d+\.log$", "", f.split("/")[-1])
    for line in open(f):
        m = re.match(r"\s+(\w+)\s+([\d.]+) ±", line)
        if m:
            vals[(arm, m.group(1))].append(float(m.group(2)))
print("ALL ARMS DONE — mean ± std over seeds:")
for (arm, metric), xs in sorted(vals.items()):
    if "model" in metric or metric == "detection_auc":
        print(f"  {arm:<12} {metric:<16} {statistics.mean(xs):.3f} ± {statistics.pstdev(xs):.3f}  (n={len(xs)})")
EOF
