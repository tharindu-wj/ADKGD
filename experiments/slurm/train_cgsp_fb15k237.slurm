#!/bin/bash
# Train the Full CGSP model on FB15K-237 — one V100 allocation.
# Replaces the legacy Gumbel-Softmax GAN (which was train_gan_fb15k237.slurm).
#
# Two-stage flow (both phases inside this single SLURM job):
#   PHASE 1 (concept module):   preprocess.py builds entity_types.tsv,
#                               entity_types_metadata.json, and the
#                               concept_pools/<DATASET>.pkl cache.
#                               Runs in ~1 minute on CPU; idempotent
#                               (skipped if cache hash matches train.txt).
#   PHASE 2 (adversarial):      train.py runs the REINFORCE loop with
#                               D warmup then alternating G/D updates.
#                               This is the long-running GPU work.
#
#   sbatch experiments/slurm/train_cgsp_fb15k237.slurm
#
# Prerequisites (run ONCE on the login node):
#   1. Conda env with torch+CUDA wheel (see RUNNING_ON_DEEPTHOUGHT.md).
#   2. FB15K-237 TSVs at data/FB15K-237/{train,valid,test}.txt.
#
# Override hyperparameters via env vars (no script edits required):
#   TOTAL_EPOCHS=200 sbatch experiments/slurm/train_cgsp_fb15k237.slurm
#   BATCH_SIZE=256 EMBEDDING_DIM=200 sbatch experiments/slurm/train_cgsp_fb15k237.slurm

#SBATCH --job-name=cgsp_train_fb15k237
#SBATCH --output=%x-%j.out.txt
# (no --error directive: stderr merges into stdout — one file to tail.)
#SBATCH --time=03:00:00
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
DATASET="${DATASET:-FB15K-237}"
FAMILY="${FAMILY:-freebase}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EMBEDDING_DIM="${EMBEDDING_DIM:-100}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
N_S="${N_S:-64}"
G_LR="${G_LR:-1e-4}"
D_LR="${D_LR:-1e-3}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
SEED="${SEED:-0}"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Dataset       : $DATASET"
echo "Family        : $FAMILY"
echo "Warmup epochs : $WARMUP_EPOCHS"
echo "Total epochs  : $TOTAL_EPOCHS"
echo "Batch size    : $BATCH_SIZE"
echo "Embedding dim : $EMBEDDING_DIM"
echo "Seed          : $SEED"
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

# ─── PHASE 1: Concept Module ──────────────────────────────────────────
# Builds entity_types.tsv + concept_pools.pkl. Cheap and idempotent so we
# always run it - guarantees the cache matches the current train.txt.
echo "[$(date)] PHASE 1: building concept pools"
$PY -m experiments.gan.concept.preprocess \
    --dataset "$DATASET" \
    --family  "$FAMILY"

CACHE_PATH="experiments/gan/outputs/concept_pools/${DATASET}.pkl"
if [[ ! -f "$CACHE_PATH" ]]; then
    echo "ERROR: concept_pools cache not produced at $CACHE_PATH" >&2
    exit 1
fi
echo "  cache OK: $CACHE_PATH"

# ─── PHASE 2: Adversarial Module (REINFORCE training) ─────────────────
CKPT_PATH="experiments/gan/outputs/checkpoints/${DATASET}_cgsp.pt"
echo "[$(date)] PHASE 2: training CGSP -> $CKPT_PATH"

$PY -m experiments.gan.adversarial.train \
    --dataset             "$DATASET" \
    --warmup_epochs       "$WARMUP_EPOCHS" \
    --total_epochs        "$TOTAL_EPOCHS" \
    --batch_size          "$BATCH_SIZE" \
    --embedding_dim       "$EMBEDDING_DIM" \
    --hidden_dim          "$HIDDEN_DIM" \
    --n_s                 "$N_S" \
    --g_lr                "$G_LR" \
    --d_lr                "$D_LR" \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --seed                "$SEED"

echo "----------------------------------------------------------------------"
echo "Done $(date). Checkpoint: $CKPT_PATH"
echo "Log:        experiments/gan/outputs/logs/${DATASET}_cgsp_training.json"
echo ""
echo "Next: (once Phase 3 corruption module lands)"
echo "  sbatch experiments/slurm/run_baseline_with_cgsp_fb15k237.slurm"
