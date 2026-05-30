#!/bin/bash
# Train the simple GAN on FB15K-237 — one V100 allocation.
# Writes the checkpoint that ADKGD's --neg_source=gan path consumes via
# experiments/gan/adkgd_bridge.py (in-process, no intermediate TSV).
#
#   sbatch experiments/slurm/train_gan_fb15k.slurm
#
# Prerequisites (run ONCE on the login node):
#   1. Conda env with torch+CUDA wheel (see RUNNING_ON_DEEPTHOUGHT.md).
#   2. FB15K-237 TSVs at data/FB15K/{train,valid,test}.txt.
#
# Override hyperparameters via env vars (no script edits required):
#   EPOCHS=600 BATCH_SIZE=1024 sbatch experiments/slurm/train_gan_fb15k.slurm
#   DATASET_DIR=data/other_kg CKPT_PATH=experiments/gan/outputs/checkpoints/other.pt \
#       sbatch experiments/slurm/train_gan_fb15k.slurm

#SBATCH --job-name=gan_train_fb15k
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

# ---- Training knobs (override per submission via env vars) ----
DATASET_DIR="${DATASET_DIR:-data/FB15K}"
CKPT_PATH="${CKPT_PATH:-experiments/gan/outputs/checkpoints/fb15k.pt}"
EPOCHS="${EPOCHS:-100}"
# 512 is a safe V100 default — fits comfortably in 32GB HBM2 and gives much
# better GPU utilisation than train.py's CPU-friendly default of 256. Bump
# to 1024 if you want to squeeze more throughput; back off to 256 if the
# model size grows and OOMs.
BATCH_SIZE="${BATCH_SIZE:-512}"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Dataset    : $DATASET_DIR"
echo "Checkpoint : $CKPT_PATH"
echo "Epochs     : $EPOCHS"
echo "Batch size : $BATCH_SIZE"
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
# Force unbuffered Python output so progress shows up in the live log
# instead of waiting for stdout to accumulate a 4KB block.
export PYTHONUNBUFFERED=1

# Fail fast if torch can't see the GPU.
$PY -c "import torch; assert torch.cuda.is_available(), 'torch cannot see a GPU: install the CUDA wheel'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| cuda', torch.version.cuda)"

cd "$PROJECT_DIR"

mkdir -p "$(dirname "$CKPT_PATH")"

echo "[$(date)] Training GAN -> $CKPT_PATH"
$PY experiments/gan/train.py \
    --data       "$DATASET_DIR" \
    --out        "$CKPT_PATH" \
    --epochs     "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --device     cuda

echo "----------------------------------------------------------------------"
echo "Done $(date). Checkpoint: $CKPT_PATH"
echo "Next: sbatch experiments/slurm/run_gan_fb15k.slurm"
echo "  (defaults read this checkpoint via --neg_source gan --gan_path $CKPT_PATH)"
