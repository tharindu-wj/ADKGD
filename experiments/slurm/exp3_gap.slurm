#!/bin/bash
# Experiment 3 (gap): ADKGD trained on RULE-BASED negatives, tested on KGSAGE
# anomalies. Can the baseline-trained detector catch the harder KGSAGE anomalies?
# The "gap" between baseline training and KGSAGE-quality anomalies.
#
#   sbatch experiments/slurm/exp3_gap.slurm
#   DATASET=WN18RR sbatch experiments/slurm/exp3_gap.slurm
#
# Matrix: Exp1 random/random  Exp2 gan/random  Exp3 random/gan <-  Exp4 gan/gan
# Requires a trained KGSAGE checkpoint (train_gan_<dataset>.slurm).

#SBATCH --job-name=exp3_gap
#SBATCH --output=%x-%j.out.txt
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

# ---- Experiment knobs (override via env vars) ----
DATASET="${DATASET:-FB15K-237}"
ANOMALY_RATIO="${ANOMALY_RATIO:-0.05}"
SEED="${SEED:-0}"
MAX_EPOCH="${MAX_EPOCH:-1}"
NEG_SOURCE="random"      # TRAIN negatives = rule-based
TEST_SOURCE="gan"        # TEST (injected) anomalies = KGSAGE

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Exp 3 (gap) | dataset=$DATASET | train=$NEG_SOURCE test=$TEST_SOURCE | anomaly_ratio=$ANOMALY_RATIO seed=$SEED"
nvidia-smi || echo "WARNING: nvidia-smi failed (no GPU visible?)"

module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || echo "WARNING: conda activate failed; using absolute python path below."

PY="$CONDA_ENV/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "ERROR: $PY does not exist or is not executable." >&2
    exit 1
fi
echo "Using python: $PY"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

$PY -c "import torch; assert torch.cuda.is_available(), 'torch cannot see a GPU: install the CUDA wheel'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"
$PY -c "import numpy, sklearn, matplotlib; print('deps ok')"

cd "$PROJECT_DIR"
mkdir -p "checkpoints/$DATASET"

# KGSAGE checkpoint (default derived from DATASET). GAN inference is PyG-free.
CKPT_TAG=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]' | tr -d '-')
GAN_CKPT="${GAN_CKPT:-experiments/kgsage/outputs/checkpoints/kgsage_${CKPT_TAG}.pt}"
if [[ ! -f "$GAN_CKPT" ]]; then
    echo "ERROR: GAN checkpoint $GAN_CKPT not found." >&2
    echo "Train it: sbatch experiments/kgsage/slurm/train_gan_${CKPT_TAG}.slurm" >&2
    exit 1
fi
echo "GAN checkpoint: $GAN_CKPT"

$PY experiments/run_experiment.py \
    --dataset "$DATASET" \
    --anomaly_ratio "$ANOMALY_RATIO" \
    --seed "$SEED" \
    --max_epoch "$MAX_EPOCH" \
    --neg_source "$NEG_SOURCE" \
    --test_anomaly_source "$TEST_SOURCE" \
    --gan_path "$GAN_CKPT"

echo "Done $(date)"
