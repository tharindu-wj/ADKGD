#!/bin/bash
# Experiment 1 (baseline): ADKGD with RULE-BASED train negatives + RULE-BASED test
# anomalies. Plain ADKGD, no GAN.
#
#   sbatch experiments/slurm/exp1_baseline.slurm
#   DATASET=WN18RR sbatch experiments/slurm/exp1_baseline.slurm    # any dataset
#
# Experiment matrix (train negatives x test anomalies):
#   Exp 1  random / random   <- this          Exp 3  random / gan   (exp3_gap)
#   Exp 2  gan    / random   (exp2_crosscheck) Exp 4  gan    / gan   (exp4_proposed)
# Compare the RESULTS Precision@K / Recall@K blocks across the four .out.txt files.

#SBATCH --job-name=exp1_baseline
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
NEG_SOURCE="random"      # TRAIN negatives
TEST_SOURCE="random"     # TEST (injected) anomalies

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Exp 1 (baseline) | dataset=$DATASET | train=$NEG_SOURCE test=$TEST_SOURCE | anomaly_ratio=$ANOMALY_RATIO seed=$SEED"
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

$PY experiments/run_experiment.py \
    --dataset "$DATASET" \
    --anomaly_ratio "$ANOMALY_RATIO" \
    --seed "$SEED" \
    --max_epoch "$MAX_EPOCH" \
    --neg_source "$NEG_SOURCE" \
    --test_anomaly_source "$TEST_SOURCE"

echo "Done $(date)"
