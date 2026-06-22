#!/bin/bash
# ADKGD baseline on WN18RR — ONE experiment (anomaly_ratio=0.05, seed=0).
# All experiment logic lives in experiments/run_experiment.py; this script is
# just the HPC launcher (resource request + env activation + GPU check + invocation).
#
#   sbatch experiments/slurm/run_baseline_wn18rr.slurm

#SBATCH --job-name=adkgd_wn18rr
#SBATCH --output=%x-%j.out.txt
# (no --error directive on purpose: SLURM then merges stderr into stdout, so
#  Python's logging output ends up in the same file as print/echo -- one file
#  to tail. Drop this comment and add `#SBATCH --error=%x-%j.err.txt` if you
#  ever need them split again.)
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla_v100:1
# Explicit account (DeepThought requires this for the gpu partition).
#SBATCH --account=cse

set -eo pipefail

# ---- EDIT THESE PATHS to match your HPC layout ----
PROJECT_DIR="$HOME/ADKGD"
CONDA_ENV="$HOME/envs/adkgd"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
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
# Force unbuffered Python output so progress streams live to the log file.
export PYTHONUNBUFFERED=1

# Fail fast if torch can't see the GPU (CPU-only wheel installed by mistake, etc.)
$PY -c "import torch; assert torch.cuda.is_available(), 'torch cannot see a GPU: install the CUDA wheel'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| cuda', torch.version.cuda)"

# Fail fast if the other required deps are missing.
$PY -c "import numpy, sklearn, matplotlib; print('deps ok | numpy', numpy.__version__, '| sklearn', sklearn.__version__, '| matplotlib', matplotlib.__version__)"

cd "$PROJECT_DIR"

# Change experiment parameters here, never inside this file's SLURM header.
# Use $PY (absolute env python) so subprocesses spawned by run_experiment.py via
# sys.executable also use the env -- guaranteeing sklearn/matplotlib are visible.
$PY experiments/run_experiment.py \
    --dataset WN18RR \
    --anomaly_ratio 0.05 \
    --seed 0 \
    --max_epoch 1

echo "Done $(date)"
