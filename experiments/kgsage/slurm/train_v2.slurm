#!/bin/bash
# KGSAGE-2 (dual-critic, LP-free) GAN training. One job per dataset.
#
#   DATASET=fb15k237 sbatch experiments/kgsage/slurm/train_v2.slurm
#   DATASET=wn18rr   sbatch experiments/kgsage/slurm/train_v2.slurm
#
# NO link-predictor artifacts needed (the LP is evaluation-only in v2).
# Env knobs (defaults follow the CPU-validated dynamics runs):
#   SEED, EPOCHS (30), DMATCH_EPOCHS (3), WARMUP_EPOCHS (10), BATCH (256),
#   CAND_K (256), ALPHA_TARGET (0.13 -- must stay ABOVE the structural
#   no-uncorroborated-candidate floor, ~0.12-0.13 on FB15K-237),
#   ALPHA_MAX (10), DREAL_MISMATCH (1.0 -- anchor-conditional plausibility;
#   the run without it collapsed to a universal-alien global ranking),
#   CKPT_PATH.
# After the run, evaluate LOCALLY (CPU): support_eval (bare + mask),
# knockout J@10, mask intervention rate -- pass gates before ADKGD cells.

#SBATCH --job-name=kgsage2_train
#SBATCH --output=%x-%j.out.txt
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --account=cse

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
EPOCHS="${EPOCHS:-30}"
DMATCH_EPOCHS="${DMATCH_EPOCHS:-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
BATCH="${BATCH:-256}"
CAND_K="${CAND_K:-256}"
ALPHA_TARGET="${ALPHA_TARGET:-0.13}"
ALPHA_MAX="${ALPHA_MAX:-10}"
DREAL_MISMATCH="${DREAL_MISMATCH:-1.0}"
CKPT_PATH="${CKPT_PATH:-experiments/kgsage/outputs/checkpoints/kgsage2_${TAG}_s${SEED}.pt}"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Dataset : $DATASET ($DATA_DIR)  Seed: $SEED"
echo "Game    : epochs=$EPOCHS dmatch=$DMATCH_EPOCHS warmup=$WARMUP_EPOCHS batch=$BATCH K=$CAND_K"
echo "Knobs   : alpha_target=$ALPHA_TARGET alpha_max=$ALPHA_MAX dreal_mismatch=$DREAL_MISMATCH"
echo "Output  : $CKPT_PATH"
echo "----------------------------------------------------------------------"

nvidia-smi || echo "WARNING: nvidia-smi failed"
module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || echo "WARNING: conda activate failed; using absolute python."
PY="$CONDA_ENV/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY missing." >&2; exit 1; }

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/experiments:$PYTHONPATH"

$PY -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print('GPU:', torch.cuda.get_device_name(0))"
$PY -c "from torch_geometric.nn import RGCNConv; print('PyG OK')"
mkdir -p "$(dirname "$CKPT_PATH")"

$PY -m kgsage.gan.train_v2 \
    --data            "$DATA_DIR" \
    --out             "$CKPT_PATH" \
    --warmup_epochs   "$WARMUP_EPOCHS" \
    --dmatch_epochs   "$DMATCH_EPOCHS" \
    --epochs          "$EPOCHS" \
    --batch_size      "$BATCH" \
    --cand_k          "$CAND_K" \
    --alpha_target    "$ALPHA_TARGET" \
    --alpha_max       "$ALPHA_MAX" \
    --dreal_mismatch  "$DREAL_MISMATCH" \
    --seed            "$SEED" \
    --device          cuda

echo "----------------------------------------------------------------------"
echo "Done $(date). Checkpoint: $CKPT_PATH"
echo "Bring back the checkpoint AND this .out.txt (the corr-pick/alpha/dm-online"
echo "trajectories are the training-health record)."
