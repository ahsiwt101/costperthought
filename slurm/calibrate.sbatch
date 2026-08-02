#!/usr/bin/env bash
#SBATCH --job-name=cpt-calib
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1     # 1 datacenter GPU (>=80GB); FP8 KV-cache needs a recent GPU arch
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm/logs/calib-%j.out
#SBATCH --error=slurm/logs/calib-%j.out
#
# Phase 4: baseline calibration. Unlimited-budget FP16/FP16 run over all 3
# benchmarks; prints per-benchmark think-token median/p25/p75 used to fill
# the budget_75/50/25pct token counts in configs/grid.yaml.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"

for bench in math gpqa code; do
  echo "=== Calibrating on $bench ==="
  python harness/run_benchmark.py --benchmark "$bench" --mode calibrate \
    --weight_precision fp16 --kv_cache_precision fp16
done

echo "[calib] copy the printed budget_75/50/25pct values into configs/grid.yaml,"
echo "[calib] then submit slurm/sweep.sbatch."
