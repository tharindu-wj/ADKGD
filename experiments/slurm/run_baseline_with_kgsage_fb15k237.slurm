#!/bin/bash
# Phase 4 / Config 2: ADKGD on FB15K-237 with the PAIR-AWARE KGSAGE generator
# supplying training negatives (role-swap contradiction partners) instead of
# uniform random corruption.
#
# Sibling of run_baseline_with_gan_fb15k237.slurm. The ONLY differences are the
# default checkpoint (a kgsage_pairgan checkpoint) and the --job-name, so the
# .out.txt files don't collide with the random baseline (Config 1) or the
# simple-GAN run. The bridge (experiments/kgsage_bridge/bridge.py) auto-detects
# the pair-aware checkpoint format and emits role-swap (t, r', h) negatives.
#
# Before submitting: train the Phase 2 GAN first ->
#   sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm
#
#   sbatch experiments/slurm/run_baseline_with_kgsage_fb15k237.slurm
#
# Phase 4 comparison:
#   Config 1 (baseline) : sbatch experiments/slurm/run_baseline_fb15k237.slurm
#   Config 2 (KGSAGE)   : this script
# Compare the RESULTS Precision@K / Recall@K blocks between the two .out.txt files.

#SBATCH --job-name=adkgd_baseline_with_kgsage_fb15k237
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

# Default to the Phase 2 pair-aware checkpoint (override with GAN_CKPT=...).
GAN_CKPT="${GAN_CKPT:-experiments/kgsage/outputs/checkpoints/kgsage_fb15k237.pt}"
if [[ ! -f "$GAN_CKPT" ]]; then
    echo "ERROR: $GAN_CKPT not found at $PROJECT_DIR/$GAN_CKPT" >&2
    echo "Train it first: sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm" >&2
    exit 1
fi
echo "Using KGSAGE checkpoint: $GAN_CKPT ($(du -h "$GAN_CKPT" | cut -f1))"

# --neg_source gan: dataset.py routes through kgsage_bridge.bridge, which
# auto-detects the pair-aware checkpoint and produces role-swap contradictions.
$PY experiments/run_experiment.py \
    --dataset FB15K-237 \
    --anomaly_ratio 0.05 \
    --seed 0 \
    --max_epoch 1 \
    --neg_source gan \
    --gan_path "$GAN_CKPT"

echo "Done $(date)"
echo "Phase 4 Test 4.1 wiring check: grep the log for a '[GAN] kgsage role-swap | processed=' line."
