#!/usr/bin/env bash
# Phase 2: baseline calibration. Run on the GPU host.
# Establishes unlimited-budget accuracy + trace-length distribution per
# benchmark, which is used to set the budget_75pct/50pct/25pct token counts
# in configs/grid.yaml before Phase 3 (full sweep) can run.
set -euo pipefail
cd "$(dirname "$0")/.."

for bench in math gpqa code; do
  echo "=== Calibrating on $bench ==="
  python harness/run_benchmark.py --benchmark "$bench" --mode calibrate \
    --weight_precision fp16 --kv_cache_precision fp16
done

echo ""
echo "Now manually copy the printed budget_75pct/50pct/25pct values into"
echo "configs/grid.yaml (per-benchmark, since trace lengths differ a lot"
echo "between math/gpqa/code), then run scripts/run_sweep.sh."
