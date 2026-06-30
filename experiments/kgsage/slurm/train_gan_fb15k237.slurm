#!/bin/bash
# Train the pair-aware KGSAGE GAN (KGSAGEGenerator + KGSAGEDiscriminator) on
# FB15K-237 — one V100 allocation.
#
# The role-swap contradiction generator: it loads the Phase 1 encoder checkpoint
# (post-RGCN entity embeddings + DistMult relation embeddings), conditions the
# generator on them, and trains adversarially against the real-2-cycle positives.
# Output: experiments/kgsage/outputs/checkpoints/kgsage_fb15k237.pt + the Test
# 2.2 / 2.3 generation probes in the log.
#
#   sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm
#
# Prerequisites (run ONCE on the login node):
#   1. Conda env with torch+CUDA wheel (see RUNNING_ON_DEEPTHOUGHT.md).
#   2. PyTorch Geometric installed (the encoder-embedding loader imports RGCNConv):
#        pip install torch_geometric
#   3. FB15K-237 TSVs at data/FB15K-237/{train,valid,test}.txt.
#   4. PHASE 1 ENCODER ALREADY TRAINED — its checkpoint must exist:
#        experiments/kgsage/outputs/fb15k237_encoder.pt
#      (produced by train_encoder_fb15k237.slurm; Tests 1.1/1.2 should pass.)
#
# Env overrides (no script edits): EPOCHS, SEED, ENCODER_CKPT, CKPT_PATH, DATASET_DIR.
#   EPOCHS=100 sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm

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
ENCODER_CKPT="${ENCODER_CKPT:-experiments/kgsage/outputs/fb15k237_encoder.pt}"
CKPT_PATH="${CKPT_PATH:-experiments/kgsage/outputs/checkpoints/kgsage_fb15k237.pt}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-0}"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Dataset       : $DATASET_DIR"
echo "Encoder ckpt  : $ENCODER_CKPT"
echo "Output ckpt   : $CKPT_PATH"
echo "Epochs        : $EPOCHS    Seed: $SEED"
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
# Fail fast if PyG isn't installed — load_encoder_embeddings imports RGCNConv.
$PY -c "from torch_geometric.nn import RGCNConv; import torch_geometric; print('PyG OK | torch_geometric', torch_geometric.__version__)"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/experiments:$PYTHONPATH"

# Fail fast if the Phase 1 encoder checkpoint is missing.
if [[ ! -f "$ENCODER_CKPT" ]]; then
    echo "ERROR: encoder checkpoint not found: $ENCODER_CKPT" >&2
    echo "       Train Phase 1 first:" >&2
    echo "         sbatch experiments/kgsage/slurm/train_encoder_fb15k237.slurm" >&2
    exit 1
fi

mkdir -p "$(dirname "$CKPT_PATH")"

echo "[$(date)] Training pair-aware KGSAGE GAN -> $CKPT_PATH"
$PY -m kgsage.cli.train_gan \
    --data         "$DATASET_DIR" \
    --encoder_ckpt "$ENCODER_CKPT" \
    --out          "$CKPT_PATH" \
    --epochs       "$EPOCHS" \
    --seed         "$SEED" \
    --device       cuda

echo "----------------------------------------------------------------------"
echo "Done $(date). Checkpoint: $CKPT_PATH"
echo "Inspect the Test 2.2 (entropy > 1.5 nats) and Test 2.3 (validity >= 95%) lines above."
echo "Feed ADKGD: sbatch experiments/slurm/run_baseline_with_kgsage_fb15k237.slurm"
