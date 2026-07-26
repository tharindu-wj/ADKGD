#!/bin/bash
# KGSAGE (dual-discriminator, LP-free) GAN training. One job per dataset.
#
#   DATASET=fb15k237 sbatch experiments/kgsage/slurm/train.slurm
#   DATASET=wn18rr   sbatch experiments/kgsage/slurm/train.slurm
#   DATASET=yago45   sbatch experiments/kgsage/slurm/train.slurm   # needs data/YAGO4.5 (see commands.md §0)
#
# The locked recipe (curriculum, alpha controller, architecture) lives as
# constants at the top of kgsage/gan/train.py -- edit there to re-tune. Only
# operational knobs are env vars here: SEED, EPOCHS (8), SNAPSHOT_EVERY (1),
# CKPT_PATH. Runs stay SHORT on purpose: anchor-awareness peaks a few pressure
# epochs after the alpha ramp then erodes, so snapshot every epoch and select
# the epoch by knockout J@10 (cli/knockout_eval.py) after the run.

#SBATCH --job-name=kgsage_train
#SBATCH --output=%x-%j.out.txt
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --account=cse

set -eo pipefail

PROJECT_DIR="$HOME/ADKGD"
CONDA_ENV="$HOME/envs/adkgd"

DATASET="${DATASET:-fb15k237}"
case "$DATASET" in
  fb15k237) DATA_DIR="data/FB15K-237"; TAG="fb15k237" ;;
  wn18rr)   DATA_DIR="data/WN18RR";    TAG="wn18rr"   ;;
  yago45)   DATA_DIR="data/YAGO4.5";   TAG="yago45"   ;;
  *) echo "ERROR: DATASET must be fb15k237, wn18rr or yago45 (got '$DATASET')" >&2; exit 1 ;;
esac

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-8}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-1}"
CKPT_PATH="${CKPT_PATH:-experiments/kgsage/outputs/checkpoints/run_${TAG}_s${SEED}.pt}"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Dataset : $DATASET ($DATA_DIR)  Seed: $SEED  Epochs: $EPOCHS  Snapshots: every $SNAPSHOT_EVERY"
echo "Recipe  : locked constants in kgsage/gan/train.py"
echo "Output  : $CKPT_PATH"
echo "----------------------------------------------------------------------"

nvidia-smi || echo "WARNING: nvidia-smi failed"
module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || echo "WARNING: conda activate failed; using absolute python."
PY="$CONDA_ENV/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY missing." >&2; exit 1; }

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/experiments:$PYTHONPATH"

$PY -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print('GPU:', torch.cuda.get_device_name(0))"
$PY -c "from torch_geometric.nn import RGCNConv; print('PyG OK')"
mkdir -p "$(dirname "$CKPT_PATH")"

$PY -m kgsage.gan.train \
    --data           "$DATA_DIR" \
    --out            "$CKPT_PATH" \
    --epochs         "$EPOCHS" \
    --snapshot_every "$SNAPSHOT_EVERY" \
    --seed           "$SEED" \
    --device         cuda

echo "----------------------------------------------------------------------"
echo "Done $(date). Checkpoint: $CKPT_PATH"
echo "Bring back the checkpoint AND this .out.txt (the corr-pick/alpha/dm-online"
echo "trajectories are the training-health record). Then select the epoch:"
echo "  python experiments/kgsage/cli/knockout_eval.py --ckpt <each .epNN.pt> --data $DATA_DIR"
