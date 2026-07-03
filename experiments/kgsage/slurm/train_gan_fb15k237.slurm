#!/bin/bash
# Train the KGSAGE conditional GAN (B1a) on FB15K-237 — one V100 allocation.
#
# B1a = an RGCN context encoder (torch_geometric FastRGCNConv) is trained JOINTLY
# with the generator + discriminator. The generator is CONDITIONED on the
# encoder's context table E', and trained toward TYPE-VALID but CONTEXT-DISTANT
# targets (the "converging-context contradiction" — kgsage.gan.targets). The
# final E' is cached in the checkpoint so ADKGD inference needs no PyG.
# Output: experiments/kgsage/outputs/checkpoints/kgsage_fb15k237.pt
#
#   sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm
#
# Prerequisites (run ONCE on the login node):
#   1. Conda env with torch+CUDA wheel (see RUNNING_ON_DEEPTHOUGHT.md).
#   2. PyTorch Geometric installed in that env (B1a's RGCN encoder needs it):
#        pip install torch_geometric
#      If importing torch_geometric crashes on markupsafe, repair it once:
#        pip install --force-reinstall markupsafe
#   3. FB15K-237 TSVs at data/FB15K-237/{train,valid,test}.txt.
#
# Env overrides (no script edits): EPOCHS, BATCH_SIZE, SEED, CKPT_PATH, DATASET_DIR,
#   DIM, Z_DIM, NUM_BASES, ENCODER_LAYERS, K_CANDIDATES, RECON_WEIGHT, CONTEXT_REFRESH,
#   TRAIN_SPLIT.
#   EPOCHS=100 BATCH_SIZE=1024 sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm
#
# TRAIN_SPLIT (default train): 'train' trains the generator on the TRAIN split only
#   (valid/test held out for clean evaluation); 'all' trains on train+valid+test with
#   the full context graph (final production generator, no held-out split).
#   TRAIN_SPLIT=all sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm
#
# CONTEXT_REFRESH (default 8): how often the full-graph RGCN is recomputed with
# encoder gradients. Per-batch (=1) is exact but too slow at FB15K-237 scale;
# 8 recomputes E' every 8 batches (still frequent joint training, ~8x faster).

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
# Recompute E' with encoder gradients every N batches (8 fits the 4h budget).
CONTEXT_REFRESH="${CONTEXT_REFRESH:-8}"
# Optional model/encoder overrides — empty means "use train.py's default".
DIM="${DIM:-}"
Z_DIM="${Z_DIM:-}"
NUM_BASES="${NUM_BASES:-}"
ENCODER_LAYERS="${ENCODER_LAYERS:-}"
K_CANDIDATES="${K_CANDIDATES:-}"
RECON_WEIGHT="${RECON_WEIGHT:-}"
# Data split to train on: empty = train.py default ('train'); set 'all' for train+valid+test.
TRAIN_SPLIT="${TRAIN_SPLIT:-}"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Dataset    : $DATASET_DIR"
echo "Checkpoint : $CKPT_PATH"
echo "Epochs     : $EPOCHS    Batch size: $BATCH_SIZE    Seed: $SEED    Context refresh: $CONTEXT_REFRESH"
echo "Overrides  : DIM=$DIM Z_DIM=$Z_DIM NUM_BASES=$NUM_BASES ENCODER_LAYERS=$ENCODER_LAYERS"
echo "             K_CANDIDATES=$K_CANDIDATES RECON_WEIGHT=$RECON_WEIGHT TRAIN_SPLIT=$TRAIN_SPLIT  (empty = default)"
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

# Fail fast if PyG isn't installed (B1a needs it for the RGCN context encoder).
# RGCNConv is exactly what kgsage.gan.encoder imports.
$PY -c "from torch_geometric.nn import RGCNConv; import torch_geometric; print('PyG OK | torch_geometric', torch_geometric.__version__)"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/experiments:$PYTHONPATH"
mkdir -p "$(dirname "$CKPT_PATH")"

# Build the command; optional overrides only appended when their env var is set.
CMD=( "$PY" -m kgsage.cli.train_gan
      --data            "$DATASET_DIR"
      --out             "$CKPT_PATH"
      --epochs          "$EPOCHS"
      --batch_size      "$BATCH_SIZE"
      --seed            "$SEED"
      --context_refresh "$CONTEXT_REFRESH"
      --device          cuda )

[[ -n "$DIM"            ]] && CMD+=( --dim            "$DIM" )
[[ -n "$Z_DIM"          ]] && CMD+=( --z_dim          "$Z_DIM" )
[[ -n "$NUM_BASES"      ]] && CMD+=( --num_bases      "$NUM_BASES" )
[[ -n "$ENCODER_LAYERS" ]] && CMD+=( --encoder_layers "$ENCODER_LAYERS" )
[[ -n "$K_CANDIDATES"   ]] && CMD+=( --k_candidates   "$K_CANDIDATES" )
[[ -n "$RECON_WEIGHT"   ]] && CMD+=( --recon_weight   "$RECON_WEIGHT" )
[[ -n "$TRAIN_SPLIT"    ]] && CMD+=( --train_split    "$TRAIN_SPLIT" )

echo "[$(date)] Training KGSAGE GAN (B1a) -> $CKPT_PATH"
echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "----------------------------------------------------------------------"
echo "Done $(date). Checkpoint: $CKPT_PATH"
echo "Next: NEG_SOURCE=gan TEST_SOURCE=random GAN_CKPT=$CKPT_PATH sbatch experiments/slurm/exp_cell.slurm"
