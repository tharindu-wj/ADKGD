#!/bin/bash
# ADKGD on WN18RR with the KGSAGE generator (B1a) supplying training negatives
# via --neg_source gan, instead of ADKGD's uniform random corruption.
#
# The KGSAGE checkpoint is trained separately (train_gan_wn18rr.slurm). At ADKGD
# training time, dataset.py routes through experiments/kgsage_bridge/bridge.py ->
# kgsage.inference.generate_negatives, which produces ONE single-slot corruption
# per positive (GAN-chosen head/tail slot, context-conditioned generator; on a
# self-loop/collision it keeps the original triple and counts used_original).
#
# Before submitting: train the KGSAGE GAN first ->
#   sbatch experiments/kgsage/slurm/train_gan_wn18rr.slurm
#
#   sbatch experiments/slurm/run_baseline_with_kgsage_wn18rr.slurm
#
# Comparison on WN18RR:
#   Config 1 (random baseline) : sbatch experiments/slurm/run_baseline_wn18rr.slurm
#   Config 2 (KGSAGE negatives): this script
# Compare the RESULTS Precision@K / Recall@K blocks between the two .out.txt files.
#
# NOTE (WN18RR is sparse, mean degree ~4.3): the context-aware mechanism has much
# less neighbourhood to work with than on FB15K-237, so treat this as a
# standard-benchmark / sparse-contrast comparison.

#SBATCH --job-name=adkgd_baseline_with_kgsage_wn18rr
#SBATCH --output=%x-%j.out.txt
# (no --error directive: stderr merges into stdout — one file to tail.)
#SBATCH --time=01:30:00
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

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
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
$PY -c "import numpy, sklearn, matplotlib; print('deps ok | numpy', numpy.__version__, '| sklearn', sklearn.__version__, '| matplotlib', matplotlib.__version__)"

cd "$PROJECT_DIR"

# ADKGD writes logs + checkpoints under ./checkpoints/<dataset>/ and does NOT
# create the directory itself — pre-create it.
mkdir -p checkpoints/WN18RR

# Default to the WN18RR KGSAGE checkpoint (override with GAN_CKPT=...).
GAN_CKPT="${GAN_CKPT:-experiments/kgsage/outputs/checkpoints/kgsage_wn18rr.pt}"
if [[ ! -f "$GAN_CKPT" ]]; then
    echo "ERROR: $GAN_CKPT not found at $PROJECT_DIR/$GAN_CKPT" >&2
    echo "Train it first: sbatch experiments/kgsage/slurm/train_gan_wn18rr.slurm" >&2
    exit 1
fi
echo "Using KGSAGE checkpoint: $GAN_CKPT ($(du -h "$GAN_CKPT" | cut -f1))"

# --neg_source gan: dataset.py routes through kgsage_bridge.bridge ->
# generate_negatives (single-slot, context-conditioned).
$PY experiments/run_experiment.py \
    --dataset WN18RR \
    --anomaly_ratio 0.05 \
    --seed 0 \
    --max_epoch 1 \
    --neg_source gan \
    --gan_path "$GAN_CKPT"

echo "Done $(date)"
echo "Wiring check: grep the log for a '[GAN] ' line with 'used_original' + 'slot_distribution'."
