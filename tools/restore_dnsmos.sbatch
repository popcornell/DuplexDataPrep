#!/bin/bash
# GPU half of the duplex pipeline: Sidon restoration, then DNSMOS on both the
# raw and the restored audio of the same shard (so the restoration's effect is
# measured channel by channel, under one protocol).
#
#   sbatch --array=0-39 --export=ALL,CORPUS=openyap,N_SHARDS=40 tools/restore_dnsmos.sbatch
#
# Pilot / alternate paths: export DUPLEXDATAPREP_OPENYAP_CONFIG=<yaml> (see README).
# Idempotent: restore skips channels already written, DNSMOS resumes from its
# JSON, so a requeued or resubmitted task only does the missing work.
#SBATCH --job-name=duplexdataprep_gpu
#SBATCH --account=bbjs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48g
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/bbjs/cornell2/MFA/logs/%x_%A_%a.out
#SBATCH --error=/scratch/bbjs/cornell2/MFA/logs/%x_%A_%a.out

set -uo pipefail

DDP_ROOT=/scratch/bbjs/cornell2/MFA/DuplexDataPrep
SIDON_PY=/scratch/bbjs/cornell2/miniconda3/envs/sidon/bin/python
DNSMOS_PY=/scratch/bbjs/cornell2/temp_dahee_eval/urgent_baselines/env/bin/python

CORPUS="${CORPUS:?set CORPUS}"
N_SHARDS="${N_SHARDS:?set N_SHARDS}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
OUT_DIR="${OUT_DIR:-/scratch/bbjs/cornell2/MFA/logs/${CORPUS}}"
TAG=$(printf "shard%03d" "$SHARD")
mkdir -p "$OUT_DIR"

export PYTHONPATH="$DDP_ROOT"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "host=$(hostname) corpus=$CORPUS shard=$SHARD/$N_SHARDS start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

t0=$(date +%s)
"$SIDON_PY" -u -m duplexdataprep restore "$CORPUS" --shard "$SHARD" --n-shards "$N_SHARDS" \
    --report "$OUT_DIR/restore_${TAG}.json"
RC=$?
echo "restore rc=$RC wall=$(( $(date +%s) - t0 ))s"
[ "$RC" -ne 0 ] && exit $RC

# DNSMOS offline: the espnet wrapper otherwise tries the HF hub.
export HF_HOME=/scratch/bbjs/cornell2/temp_dahee_eval/urgent_baselines/urgent_metrics/hf_cache
export HF_HUB_OFFLINE=1
for AUDIO in raw restored; do
  t0=$(date +%s)
  "$DNSMOS_PY" -u -m duplexdataprep dnsmos "$CORPUS" --audio "$AUDIO" \
      --shard "$SHARD" --n-shards "$N_SHARDS" \
      --out "$OUT_DIR/dnsmos_${AUDIO}_${TAG}.json"
  RC=$?
  echo "dnsmos[$AUDIO] rc=$RC wall=$(( $(date +%s) - t0 ))s"
  [ "$RC" -ne 0 ] && exit $RC
done
echo "end=$(date -Is)"
exit 0
