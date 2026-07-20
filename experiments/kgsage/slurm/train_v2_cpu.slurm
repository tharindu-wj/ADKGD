#!/bin/bash
# KGSAGE-2 (dual-critic) training on a CPU node -- for when the GPU partition is
# blocked. The one GPU-hungry phase (RGCN warm-up -> E') is SKIPPED by reusing
# the frozen E' already stored in the v1 checkpoint (--init_context_from); E' is
# a frozen input in v2, so this is equivalent to (better than) a fresh warm-up.
#
#   CPU_PARTITION=<name> DATASET=fb15k237 sbatch experiments/kgsage/slurm/train_v2_cpu.slurm
#   CPU_PARTITION=<name> DATASET=wn18rr   sbatch experiments/kgsage/slurm/train_v2_cpu.slurm
#
# !! Set CPU_PARTITION to a real non-GPU partition -- run `sinfo -s` to list them
#    (Flinders DeepThought commonly has a 'general'/'hpc' CPU partition).
# The v1 checkpoint (kgsage_<tag>_s0.pt) must already be on disk for --init_context_from.

#SBATCH --job-name=kgsage2_cpu
#SBATCH --output=%x-%j.out.txt
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --account=cse
# NOTE: --partition is set from $CPU_PARTITION below via `scontrol`-free re-exec
#       is not possible; instead pass it at submit time:
#         sbatch --partition="$CPU_PARTITION" experiments/kgsage/slurm/train_v2_cpu.slurm
#       (SBATCH directives can't read env vars, so the partition MUST be on the
#        sbatch command line.)

set -eo pipefail

PROJECT_DIR="$HOME/ADKGD"
CONDA_ENV="$HOME/envs/adkgd"

DATASET="${DATASET:-fb15k237}"
case "$DATASET" in
  fb15k237) DATA_DIR="data/FB15K-237"; TAG="fb15k237" ;;
  wn18rr)   DATA_DIR="data/WN18RR";    TAG="wn18rr"   ;;
  *) echo "ERROR: DATASET must be fb15k237 or wn18rr (got '$DATASET')" >&2; exit 1 ;;
esac

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-20}"
DMATCH_EPOCHS="${DMATCH_EPOCHS:-3}"
BATCH="${BATCH:-256}"
CAND_K="${CAND_K:-256}"
ALPHA_TARGET="${ALPHA_TARGET:-0.13}"
ALPHA_MAX="${ALPHA_MAX:-10}"
DREAL_MISMATCH="${DREAL_MISMATCH:-1.0}"
DREAL_PRETRAIN="${DREAL_PRETRAIN:-3}"
ALPHA_WARMUP="${ALPHA_WARMUP:-5}"
CKPT_PATH="${CKPT_PATH:-experiments/kgsage/outputs/checkpoints/kgsage2_${TAG}_s${SEED}.pt}"
INIT_E="${INIT_E:-experiments/kgsage/outputs/checkpoints/kgsage_${TAG}_s${SEED}.pt}"

echo "Job $SLURM_JOB_ID on $(hostname) [$(sinfo -h -n $(hostname) -o %P 2>/dev/null)] $(date)"
echo "Dataset : $DATASET   CPU cores: $SLURM_CPUS_PER_TASK"
echo "Reuse E': $INIT_E   (RGCN warm-up SKIPPED)"
echo "Game    : epochs=$EPOCHS dmatch=$DMATCH_EPOCHS dreal_pre=$DREAL_PRETRAIN alpha_warmup=$ALPHA_WARMUP"
echo "Output  : $CKPT_PATH"
echo "----------------------------------------------------------------------"

module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || echo "WARNING: conda activate failed; using absolute python."
PY="$CONDA_ENV/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY missing." >&2; exit 1; }

[[ -f "$INIT_E" ]] || { echo "ERROR: E'-source checkpoint $INIT_E not found. scp the v1 s0 checkpoint up first." >&2; exit 1; }

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/experiments:$PYTHONPATH"
mkdir -p "$(dirname "$CKPT_PATH")"

$PY -m kgsage.gan.train_v2 \
    --data                  "$DATA_DIR" \
    --out                   "$CKPT_PATH" \
    --init_context_from     "$INIT_E" \
    --dmatch_epochs         "$DMATCH_EPOCHS" \
    --dreal_pretrain_epochs "$DREAL_PRETRAIN" \
    --alpha_warmup_epochs   "$ALPHA_WARMUP" \
    --epochs                "$EPOCHS" \
    --batch_size            "$BATCH" \
    --cand_k                "$CAND_K" \
    --alpha_target          "$ALPHA_TARGET" \
    --alpha_max             "$ALPHA_MAX" \
    --dreal_mismatch        "$DREAL_MISMATCH" \
    --seed                  "$SEED" \
    --device                cpu

echo "----------------------------------------------------------------------"
echo "Done $(date). Checkpoint: $CKPT_PATH"
