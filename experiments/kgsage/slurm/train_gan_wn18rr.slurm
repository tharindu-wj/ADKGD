#!/bin/bash
# Train the KGSAGE conditional GAN (B1a) on WN18RR — one V100 allocation.
#
# Same pipeline as train_gan_fb15k237.slurm (RGCN context encoder trained JOINTLY
# with the generator + discriminator; context-distant contradiction targets). The
# code is dataset-agnostic; only the paths/defaults change here.
# Output: experiments/kgsage/outputs/checkpoints/kgsage_wn18rr.pt
#
#   sbatch experiments/kgsage/slurm/train_gan_wn18rr.slurm
#
# WN18RR characteristics (vs FB15K-237): 40,943 entities, only 11 relations,
# 86,835 train triples, and it is SPARSE (mean degree ~4.3, median 3). Two
# consequences:
#   * The RGCN is CHEAP (22 effective relations after inverses; num_bases auto-caps
#     to 22), so CONTEXT_REFRESH can be low (per-batch joint training is affordable).
#   * The graph is sparse, so the "converging-context" signal is weaker than on
#     dense FB15K-237 — WN18RR is best treated as a standard-benchmark / sparse
#     contrast dataset, not a place the context mechanism is expected to shine.
#
# Prerequisites (run ONCE on the login node):
#   1. Conda env with torch+CUDA wheel (see RUNNING_ON_DEEPTHOUGHT.md).
#   2. PyTorch Geometric installed (B1a's RGCN encoder needs it):
#        pip install torch_geometric
#      If importing torch_geometric crashes on markupsafe, repair it once:
#        pip install --force-reinstall markupsafe
#   3. WN18RR TSVs at data/WN18RR/{train,valid,test}.txt.
#
# Env overrides (no script edits): EPOCHS, BATCH_SIZE, SEED, CKPT_PATH, DATASET_DIR,
#   DIM, Z_DIM, NUM_BASES, ENCODER_LAYERS, K_CANDIDATES, RECON_WEIGHT, CONTEXT_REFRESH,
#   TRAIN_SPLIT.
#   EPOCHS=150 sbatch experiments/kgsage/slurm/train_gan_wn18rr.slurm
#
# TRAIN_SPLIT (default train): 'train' trains the generator on the TRAIN split only
#   (valid/test held out for clean evaluation); 'all' trains on train+valid+test with
#   the full context graph (final production generator, no held-out split).
#   TRAIN_SPLIT=all sbatch experiments/kgsage/slurm/train_gan_wn18rr.slurm

#SBATCH --job-name=kgsage_train_gan_wn18rr
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
DATASET_DIR="${DATASET_DIR:-data/WN18RR}"
CKPT_PATH="${CKPT_PATH:-experiments/kgsage/outputs/checkpoints/kgsage_wn18rr.pt}"
# WN18RR is small (86,835 train triples) and trains fast, so more epochs are cheap.
EPOCHS="${EPOCHS:-100}"
# 512 is a safe V100 default. Note WN18RR has 40,943 entities, so the generator's
# per-slot logits are larger than FB15K-237's — 512 still fits 32GB comfortably.
BATCH_SIZE="${BATCH_SIZE:-512}"
SEED="${SEED:-0}"
# The RGCN is cheap here (22 relations), so recompute E' often. 2 = near-exact
# joint training; drop to 1 for exact, raise if you ever want more speed.
CONTEXT_REFRESH="${CONTEXT_REFRESH:-2}"
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

echo "[$(date)] Training KGSAGE GAN (B1a) on WN18RR -> $CKPT_PATH"
echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "----------------------------------------------------------------------"
echo "Done $(date). Checkpoint: $CKPT_PATH"
echo "Next: sbatch experiments/slurm/exp_cell.slurm (NEG_SOURCE=gan TEST_SOURCE=random DATASET=WN18RR GAN_CKPT=$CKPT_PATH)"
