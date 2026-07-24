#!/usr/bin/env bash
# KGSAGE (dual-discriminator, LP-free) GAN training on AWS -- plain bash, no
# scheduler. Same recipe as the DeepThought experiments/kgsage/slurm/train.slurm:
# the locked constants (curriculum, alpha controller, architecture) live in
# kgsage/gan/train.py; only operational knobs are env vars here.
#
#   DATASET=fb15k237 bash deploy/aws/run.sh
#   DATASET=wn18rr   bash deploy/aws/run.sh
#   DATASET=fb15k237 SEED=1 EPOCHS=8 SNAPSHOT_EVERY=1 bash deploy/aws/run.sh
#
# Runs stay SHORT on purpose: anchor-awareness peaks a few pressure epochs after
# the alpha ramp then erodes -- snapshot every epoch, pick the epoch afterwards
# by knockout J@10 (cli/knockout_eval.py).
set -eo pipefail

CONDA_ENV="${CONDA_ENV:-$HOME/envs/adkgd}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-fb15k237}"
case "$DATASET" in
  fb15k237) DATA_DIR="data/FB15K-237"; TAG="fb15k237" ;;
  wn18rr)   DATA_DIR="data/WN18RR";    TAG="wn18rr"   ;;
  *) echo "ERROR: DATASET must be fb15k237 or wn18rr (got '$DATASET')" >&2; exit 1 ;;
esac

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-8}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-1}"
CKPT_PATH="${CKPT_PATH:-experiments/kgsage/outputs/checkpoints/run_${TAG}_s${SEED}.pt}"
LOG="${CKPT_PATH%.pt}.out.txt"

mkdir -p "$(dirname "$CKPT_PATH")"
exec > >(tee "$LOG") 2>&1        # mirror the DeepThought .out.txt training-health record

echo "Host $(hostname) starting $(date)"
echo "Dataset: $DATASET ($DATA_DIR)  Seed: $SEED  Epochs: $EPOCHS  Snapshots: every $SNAPSHOT_EVERY"
echo "Output : $CKPT_PATH   Log: $LOG"
echo "----------------------------------------------------------------------"

nvidia-smi || echo "WARNING: nvidia-smi failed"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || echo "WARNING: conda activate failed; using absolute python."
PY="$CONDA_ENV/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY missing (run deploy/aws/bootstrap.sh first)." >&2; exit 1; }

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/experiments:$PYTHONPATH"

$PY -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print('GPU:', torch.cuda.get_device_name(0))"
$PY -c "from torch_geometric.nn import RGCNConv; print('PyG OK')"

$PY -m kgsage.gan.train \
    --data           "$DATA_DIR" \
    --out            "$CKPT_PATH" \
    --epochs         "$EPOCHS" \
    --snapshot_every "$SNAPSHOT_EVERY" \
    --seed           "$SEED" \
    --device         cuda

echo "----------------------------------------------------------------------"
echo "Done $(date). Checkpoint: $CKPT_PATH  (per-epoch snapshots: ${CKPT_PATH%.pt}.epNN.pt)"
echo "Select the best epoch by knockout J@10:"
echo "  $PY experiments/kgsage/cli/knockout_eval.py --ckpt <each .epNN.pt> --data $DATA_DIR"
