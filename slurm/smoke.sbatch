#!/usr/bin/env bash
#SBATCH --job-name=cpt-smoke
#SBATCH --partition=gpu
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1     # 1 datacenter GPU (>=80GB); FP8 KV-cache needs a recent GPU arch
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm/logs/smoke-%j.out
#SBATCH --error=slurm/logs/smoke-%j.out
#
# Phase 3 go/no-go gate: runs ONE tiny cell end-to-end (5 math problems,
# FP16 weights + FP8 KV-cache) to prove engine build + FP8 acceptance on GPU
# + budget forcing + grading + cost math all work before scaling.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/env.sh"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python harness/run_benchmark.py \
  --benchmark math \
  --weight_precision fp16 \
  --kv_cache_precision fp8 \
  --budget unlimited \
  --limit 5

echo "[smoke] tail of summary.jsonl:"
tail -n 1 results/summary.jsonl
