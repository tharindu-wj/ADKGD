#!/bin/bash
# Train the KGSAGE Encoder (RGCN + DistMult) on FB15K-237.
#
# This is PHASE 1 of the thesis pipeline. The checkpoint produced here
# (entity + relation embeddings) feeds Phase 2 (KGSAGE Generator +
# Discriminator), which will get its own SLURM script later.
#
#   sbatch experiments/slurm/train_kgsage_fb15k237.slurm
#
# Prerequisites (run ONCE on the login node):
#   1. Conda env with torch+CUDA wheel (see RUNNING_ON_DEEPTHOUGHT.md).
#   2. PyTorch Geometric installed in that env:
#        pip install torch_geometric
#   3. FB15K-237 TSVs at data/FB15K-237/{train,valid,test}.txt
#   4. Phase 1 Test 1.3 (density audit) has passed:
#        python -m experiments.kgsage.audit_density --data data/FB15K-237
#
# Override hyperparameters via env vars (no script edits required):
#   EPOCHS=300 sbatch experiments/slurm/train_kgsage_fb15k237.slurm
#   DIM=256 BATCH_SIZE=1024 sbatch experiments/slurm/train_kgsage_fb15k237.slurm
#   DATASET_DIR=data/WN18RR \
#       CKPT_PATH=experiments/kgsage/outputs/wn18rr_encoder.pt \
#       sbatch experiments/slurm/train_kgsage_fb15k237.slurm

#SBATCH --job-name=kgsage_train_encoder_fb15k237
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
DATASET_DIR="${DATASET_DIR:-data/FB15K-237}"
CKPT_PATH="${CKPT_PATH:-experiments/kgsage/outputs/fb15k237_encoder.pt}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
LR="${LR:-1e-3}"
DIM="${DIM:-200}"
N_LAYERS="${N_LAYERS:-2}"
NUM_BASES="${NUM_BASES:-30}"
MARGIN="${MARGIN:-1.0}"
SEED="${SEED:-0}"
EVAL_EVERY="${EVAL_EVERY:-10}"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Dataset      : $DATASET_DIR"
echo "Checkpoint   : $CKPT_PATH"
echo "Epochs       : $EPOCHS"
echo "Batch size   : $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Embedding dim: $DIM"
echo "RGCN layers  : $N_LAYERS"
echo "RGCN bases   : $NUM_BASES"
echo "Margin       : $MARGIN"
echo "Seed         : $SEED"
echo "Eval every   : $EVAL_EVERY epochs"
echo "----------------------------------------------------------------------"

nvidia-smi || echo "WARNING: nvidia-smi failed (no GPU visible?)"

module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
# best-effort activate (sets CONDA_PREFIX etc.); we don't rely on it switching PATH.
conda activate "$CONDA_ENV" || echo "WARNING: conda activate failed; using absolute python path below."

# Use the env's python by absolute path -- robust to `conda activate` silently
# failing to switch PATH inside non-interactive slurm shells.
PY="$CONDA_ENV/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "ERROR: $PY does not exist or is not executable." >&2
    echo "Create the env on the login node first (see RUNNING_ON_DEEPTHOUGHT.md step 1)." >&2
    exit 1
fi
echo "Using python: $PY"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# Force unbuffered Python output so training progress streams live to the log.
export PYTHONUNBUFFERED=1

# Fail fast if torch can't see the GPU.
$PY -c "import torch; assert torch.cuda.is_available(), 'torch cannot see a GPU: install the CUDA wheel'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| cuda', torch.version.cuda)"

# Fail fast if PyG isn't installed (KGSAGE-specific dependency).
# We import RGCNConv specifically because that's what the encoder needs.
$PY -c "from torch_geometric.nn import RGCNConv; import torch_geometric; print('PyG OK | torch_geometric', torch_geometric.__version__)"

cd "$PROJECT_DIR"

# Belt-and-braces: make sure the dataset is on disk.
if [[ ! -f "$DATASET_DIR/train.txt" ]]; then
    echo "ERROR: $DATASET_DIR/train.txt not found at $PROJECT_DIR/$DATASET_DIR" >&2
    echo "Check DATASET_DIR or pull the dataset onto the HPC." >&2
    exit 1
fi

mkdir -p "$(dirname "$CKPT_PATH")"

echo "[$(date)] Training KGSAGE Encoder + DistMult Decoder -> $CKPT_PATH"
$PY -m experiments.kgsage.train_encoder \
    --data       "$DATASET_DIR" \
    --out        "$CKPT_PATH" \
    --epochs     "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr         "$LR" \
    --margin     "$MARGIN" \
    --dim        "$DIM" \
    --n_layers   "$N_LAYERS" \
    --num_bases  "$NUM_BASES" \
    --seed       "$SEED" \
    --eval_every "$EVAL_EVERY" \
    --device     cuda

echo "----------------------------------------------------------------------"
echo "Done $(date). Checkpoint: $CKPT_PATH"
echo ""
echo "Next steps (run on the LOGIN NODE, not as another SLURM job — they're cheap):"
echo ""
echo "  Test 1.1 — Link prediction MRR (Pass: MRR ≥ 0.30)"
echo "    cd \$PROJECT_DIR"
echo "    \$PY -m experiments.kgsage.evaluate_lp \\"
echo "        --data $DATASET_DIR \\"
echo "        --ckpt $CKPT_PATH"
echo ""
echo "  Test 1.2 — Anti-symmetric signal in relation embeddings (Pass: p < 0.05)"
echo "    \$PY -m experiments.kgsage.test_antisym_signal \\"
echo "        --ckpt $CKPT_PATH \\"
echo "        --audit experiments/kgsage/outputs/density_audit.json"
echo ""
echo "  (Both tests run in seconds on a single GPU. If you want to be thorough"
echo "   about it, scp the checkpoint to your laptop and run there.)"
echo ""
echo "Phase 1 complete once all three Phase 1 tests pass. Phase 2 (KGSAGE"
echo "Generator + Discriminator) will get its own SLURM script later — it"
echo "loads this checkpoint as initialisation."
