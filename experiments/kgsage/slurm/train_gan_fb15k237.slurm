#!/bin/bash
# Train the KGSAGE GAN (KGSAGEGenerator + KGSAGEDiscriminator) on FB15K-237 —
# one V100 allocation. The GAN learns its own embeddings and generates
# single-slot-corruption negatives for ADKGD's --neg_source gan path.
# Output: experiments/kgsage/outputs/checkpoints/kgsage_fb15k237.pt
#
#   sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm
#
# Prerequisites (run ONCE on the login node):
#   1. Conda env with torch+CUDA wheel (see RUNNING_ON_DEEPTHOUGHT.md).
#   2. FB15K-237 TSVs at data/FB15K-237/{train,valid,test}.txt.
#
# Env overrides (no script edits): EPOCHS, BATCH_SIZE, SEED, CKPT_PATH, DATASET_DIR.
#   EPOCHS=100 BATCH_SIZE=1024 sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm

#SBATCH --job-name=kgsage_train_gan_fb15k237
#SBATCH --output=%x-%j.out.txt
# (no --error directive: stderr merges into stdout — one file to tail.)
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --account=cse

set -eo pipefail

# ---- EDIT THESE PATHS to match your HPC layout ----
PROJECT_DIR="$HOME/ADKGD"
CONDA_ENV="$HOME/envs/adkgd"

# ---- Run knobs (override per submission via env vars) ----
DATASET_DIR="${DATASET_DIR:-data/FB15K-237}"
CKPT_PATH="${CKPT_PATH:-experiments/kgsage/outputs/checkpoints/kgsage_fb15k237.pt}"
EPOCHS="${EPOCHS:-50}"
# 512 is a safe V100 default (fits 32GB HBM2). Bump to 1024 for more throughput.
BATCH_SIZE="${BATCH_SIZE:-512}"
SEED="${SEED:-0}"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Dataset    : $DATASET_DIR"
echo "Checkpoint : $CKPT_PATH"
echo "Epochs     : $EPOCHS    Batch size: $BATCH_SIZE    Seed: $SEED"
echo "----------------------------------------------------------------------"

nvidia-smi || echo "WARNING: nvidia-smi failed (no GPU visible?)"

module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || echo "WARNING: conda activate failed; using absolute python path below."

PY="$CONDA_ENV/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "ERROR: $PY does not exist or is not executable." >&2
    echo "Create the env on the login node first (see RUNNING_ON_DEEPTHOUGHT.md step 1)." >&2
    exit 1
fi
echo "Using python: $PY"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

# Fail fast if torch can't see the GPU.
$PY -c "import torch; assert torch.cuda.is_available(), 'torch cannot see a GPU: install the CUDA wheel'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| cuda', torch.version.cuda)"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/experiments:$PYTHONPATH"
mkdir -p "$(dirname "$CKPT_PATH")"

echo "[$(date)] Training KGSAGE GAN -> $CKPT_PATH"
$PY -m kgsage.cli.train_gan \
    --data       "$DATASET_DIR" \
    --out        "$CKPT_PATH" \
    --epochs     "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --seed       "$SEED" \
    --device     cuda

echo "----------------------------------------------------------------------"
echo "Done $(date). Checkpoint: $CKPT_PATH"
echo "Next: sbatch experiments/slurm/run_baseline_with_kgsage_fb15k237.slurm"
