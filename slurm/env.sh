#!/usr/bin/env bash
# Shared environment for every CostPerThought Slurm job. Sourced (not executed)
# by smoke/calibrate/sweep sbatch scripts.
#
# The three values marked [RECON] are finalized from slurm/recon.sh output in
# Phase 0. Until then they hold best-guess defaults.
set -euo pipefail

# --- Project root (repo lives at ~/costperthought on the cluster) -------------
export CPT_ROOT="${CPT_ROOT:-$HOME/costperthought}"

# --- Scratch: HF cache + model weights ---------------------------------------
# RECON (25 Jul): the cluster has no /scratch or /hpctmp; $HOME is a large network FS
# with no per-user quota, shared across login+compute nodes -> weights staged
# here are visible on the compute nodes. So scratch lives under $HOME.
export CPT_SCRATCH="${CPT_SCRATCH:-$HOME/cpt-scratch}"
export HF_HOME="$CPT_SCRATCH/hf"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # weights pre-staged; compute nodes may be offline
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_CACHE_ROOT="$CPT_SCRATCH/vllm"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"

# --- Toolchain: load CUDA + activate the python env --------------------------
# [RECON] replace with the module names recon.sh finds, or the micromamba path.
if command -v module >/dev/null 2>&1; then
  module load cuda 2>/dev/null || true
fi
# Activate the venv created in Phase 1 (slurm/setup.sbatch -> $HOME/cpt-venv).
if [ -f "$HOME/cpt-venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/cpt-venv/bin/activate"
else
  echo "[env] WARNING: $HOME/cpt-venv not found - run slurm/setup.sbatch first." >&2
fi

cd "$CPT_ROOT"
mkdir -p slurm/logs results data
echo "[env] host=$(hostname) python=$(command -v python) HF_HOME=$HF_HOME"
python -c "import vllm, torch; print(f'[env] vllm={vllm.__version__} torch={torch.__version__} cuda={torch.cuda.is_available()}')" || true
