#!/usr/bin/env bash
# Phase 0 cluster recon. Run ONCE on the login node to learn the cluster-specific
# facts the Slurm job headers depend on. Read-only; changes nothing.
#   ssh <user>@<cluster-login-host> 'bash -s' < slurm/recon.sh
set +e

echo "==================== HOST ===================="
hostname; whoami; echo

echo "==================== MODULES (cuda/python/conda) ===================="
( module avail ) 2>&1 | grep -iE 'cuda|python|conda|anaconda|miniconda|mamba' | head -40
echo "(if empty: no Lmod modules - will use system python / user-installed micromamba)"
echo

echo "==================== TOOLCHAIN ON LOGIN NODE ===================="
for t in python3 python conda mamba micromamba uv pip nvidia-smi git curl rsync; do
  printf '%-12s ' "$t"; command -v "$t" || echo "(not found)"
done
echo

echo "==================== STORAGE / QUOTA ===================="
quota -s 2>/dev/null || echo "(no quota cmd)"
echo "--- df on candidate locations ---"
df -h ~ /scratch* /hpctmp* /data* 2>/dev/null | sort -u
echo "--- writable scratch dirs ---"
for d in /scratch /scratch/$USER /hpctmp /hpctmp/$USER /data /data/$USER; do
  [ -d "$d" ] && printf '%-24s writable=%s\n' "$d" "$([ -w "$d" ] && echo yes || echo no)"
done
echo

echo "==================== GPU REQUEST SYNTAX (gres/features) ===================="
scontrol show node <gpu-node-1> <gpu-node-2> 2>/dev/null | grep -iE 'NodeName|Gres|ActiveFeat|AvailableFeat|CfgTRES'
echo

echo "==================== IDLE GPUs RIGHT NOW ===================="
sinfo -N -p <gpu-partition> -o '%N %G %t %f' 2>/dev/null | grep -iE 'gpu' | head -30
echo

echo "==================== DONE - paste this whole block back ===================="
