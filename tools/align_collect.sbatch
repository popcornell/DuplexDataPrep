#!/bin/bash
# CPU half: MFA over one prepared shard, then fold the TextGrids back into
# supervisions. Run after `msprep prep` (seconds, login node is fine) and after
# the restore shard the corpus points at has finished.
#
#   sbatch --array=0-39 --export=ALL,CORPUS=openyap tools/align_collect.sbatch
#
# 8 cores / 15 GB: MFA's wall clock is flat from 4 to 16 cores on this data and
# peak RSS was 6.4 GB at nj=16 (see MFA/scripts/align_array.sbatch).
#SBATCH --job-name=msprep_align
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15g
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/bbjs/cornell2/MFA/logs/%x_%A_%a.out
#SBATCH --error=/scratch/bbjs/cornell2/MFA/logs/%x_%A_%a.out

set -uo pipefail

ROOT=/scratch/bbjs/cornell2/MFA
CORPUS="${CORPUS:?set CORPUS}"
TAG=$(printf "shard%03d" "${SLURM_ARRAY_TASK_ID:-0}")
CORPUS_DIR="${CORPUS_DIR:-$ROOT/corpus/$CORPUS}/$TAG"
ALIGN_DIR="${ALIGN_DIR:-$ROOT/out/$CORPUS}/$TAG"
OUT_DIR="${OUT_DIR:-$ROOT/logs/$CORPUS}"
ACOUSTIC=$ROOT/models/pretrained_models/acoustic/english_us_arpa.zip
DICT=$ROOT/models/pretrained_models/dictionary/english_us_arpa.dict

source /scratch/bbjs/cornell2/miniconda3/etc/profile.d/conda.sh
conda activate mfa
export PYTHONPATH=$ROOT/msprep

TMP_BASE="${SLURM_TMPDIR:-/tmp}/mfa_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMP_BASE" "$ALIGN_DIR" "$OUT_DIR"
trap 'rm -rf "$TMP_BASE"' EXIT
export MFA_ROOT_DIR="$TMP_BASE" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NJ="${SLURM_CPUS_PER_TASK:-8}"

echo "host=$(hostname) corpus=$CORPUS $TAG nj=$NJ files=$(ls "$CORPUS_DIR"/*.TextGrid | wc -l) start=$(date -Is)"
t0=$(date +%s)
mfa align --config_path "$ROOT/scripts/align_config.yaml" \
  --temporary_directory "$TMP_BASE" --num_jobs "$NJ" --use_mp --clean --overwrite \
  --no_use_postgres --output_format long_textgrid --include_original_text \
  "$CORPUS_DIR" "$DICT" "$ACOUSTIC" "$ALIGN_DIR"
RC=$?
echo "mfa rc=$RC wall=$(( $(date +%s) - t0 ))s aligned=$(ls "$ALIGN_DIR"/*.TextGrid 2>/dev/null | wc -l)"
[ "$RC" -ne 0 ] && exit $RC

python -u -m msprep collect "$CORPUS" --aligned-dir "$ALIGN_DIR" --phones \
  --output "$OUT_DIR/supervisions_mfa_${TAG}.jsonl.gz" \
  --report "$OUT_DIR/collect_${TAG}.json"
RC=$?
echo "collect rc=$RC end=$(date -Is)"
exit $RC
