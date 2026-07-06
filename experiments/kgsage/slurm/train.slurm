#!/bin/bash
# Train the KGSAGE generator (frozen ComplEx backbone + contextual
# residual discriminator; kgsage.gan.train) — one V100 allocation.
#
# Generic over datasets: set DATASET=fb15k237 (default) or DATASET=wn18rr.
# LP checkpoint + id-map dir are derived from DATASET; fetch them ONCE on the
# login node first:
#     python -m kgsage.cli.fetch_lp --dataset all      (PYTHONPATH=experiments)
# (runs the filtered-MRR loader gate; both datasets must print GATE PASS.)
#
#   DATASET=fb15k237 SEED=0 sbatch experiments/kgsage/slurm/train.slurm
#   DATASET=fb15k237 SEED=1 sbatch experiments/kgsage/slurm/train.slurm
#   DATASET=wn18rr   SEED=0 sbatch experiments/kgsage/slurm/train.slurm
#   DATASET=wn18rr   SEED=1 sbatch experiments/kgsage/slurm/train.slurm
#
# Env overrides: SEED, WARMUP_EPOCHS, WARMSTART_EPOCHS, EPOCHS, BATCH_SIZE,
#   CKPT_PATH, TAU, BAND_K, FENCE_SIGMA, LAMBDA_H, TRAIN_SPLIT.
# Output checkpoint: experiments/kgsage/outputs/checkpoints/kgsage_<tag>_s<seed>.pt

#SBATCH --job-name=kgsage_train
#SBATCH --output=%x-%j.out.txt
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

# ---- Dataset selection (derives data dir + LP artifacts) ----
DATASET="${DATASET:-fb15k237}"
case "$DATASET" in
  fb15k237) DATA_DIR="data/FB15K-237"; LP_STEM="fb15k-237"; TAG="fb15k237" ;;
  wn18rr)   DATA_DIR="data/WN18RR";    LP_STEM="wnrr";      TAG="wn18rr"   ;;
  *) echo "ERROR: DATASET must be fb15k237 or wn18rr (got '$DATASET')" >&2; exit 1 ;;
esac
LP_CKPT="experiments/kgsage/outputs/lp/${LP_STEM}-complex.pt"
LP_IDS="experiments/kgsage/outputs/lp/${LP_STEM}"

# ---- Run knobs ----
SEED="${SEED:-0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
WARMSTART_EPOCHS="${WARMSTART_EPOCHS:-2}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CKPT_PATH="${CKPT_PATH:-experiments/kgsage/outputs/checkpoints/kgsage_${TAG}_s${SEED}.pt}"
TAU="${TAU:-0.5}"
BAND_K="${BAND_K:-10}"
FENCE_SIGMA="${FENCE_SIGMA:-0.5}"
LAMBDA_H="${LAMBDA_H:-0.01}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Dataset : $DATASET ($DATA_DIR)   Seed: $SEED"
echo "LP      : $LP_CKPT (ids: $LP_IDS)"
echo "Phases  : warmup=$WARMUP_EPOCHS warmstart=$WARMSTART_EPOCHS adversarial=$EPOCHS  batch=$BATCH_SIZE"
echo "Output  : $CKPT_PATH"
echo "----------------------------------------------------------------------"

nvidia-smi || echo "WARNING: nvidia-smi failed (no GPU visible?)"

module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || echo "WARNING: conda activate failed; using absolute python path below."

PY="$CONDA_ENV/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY missing — create the env first (RUNNING_ON_DEEPTHOUGHT.md)." >&2; exit 1; }

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/experiments:$PYTHONPATH"

# Fail-fast preflight: GPU, PyG (warmup RGCN), and the LP artifacts.
$PY -c "import torch; assert torch.cuda.is_available(), 'torch cannot see a GPU'; print('GPU:', torch.cuda.get_device_name(0))"
$PY -c "from torch_geometric.nn import RGCNConv; print('PyG OK')"
[[ -f "$LP_CKPT" && -f "$LP_IDS/train.txt" ]] || {
    echo "ERROR: LP artifacts missing. Run once on the login node:" >&2
    echo "  PYTHONPATH=experiments $PY -m kgsage.cli.fetch_lp --dataset $DATASET" >&2
    exit 1; }

mkdir -p "$(dirname "$CKPT_PATH")"

echo "[$(date)] Training generator -> $CKPT_PATH"
$PY -m kgsage.gan.train \
    --data             "$DATA_DIR" \
    --out              "$CKPT_PATH" \
    --lp_ckpt          "$LP_CKPT" \
    --lp_ids           "$LP_IDS" \
    --train_split      "$TRAIN_SPLIT" \
    --warmup_epochs    "$WARMUP_EPOCHS" \
    --warmstart_epochs "$WARMSTART_EPOCHS" \
    --epochs           "$EPOCHS" \
    --batch_size       "$BATCH_SIZE" \
    --tau              "$TAU" \
    --band_k           "$BAND_K" \
    --fence_sigma      "$FENCE_SIGMA" \
    --lambda_h         "$LAMBDA_H" \
    --seed             "$SEED" \
    --device           cuda

echo "----------------------------------------------------------------------"
echo "Done $(date). Checkpoint: $CKPT_PATH"
echo "Inspect it:  PYTHONPATH=experiments $PY -m kgsage.cli.inspect_gan_lp \\"
echo "  --ckpt $CKPT_PATH --data $DATA_DIR --lp_ckpt $LP_CKPT --lp_ids $LP_IDS --n 40"
