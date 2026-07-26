#!/bin/bash
# KGSAGE generator training (dual-discriminator architecture) on a CPU node --
# for when the GPU partition is blocked. If a cached-E' donor checkpoint
# exists, the GPU-hungry Phase-1 RGCN warm-up is SKIPPED by reusing it
# (--init_context_from) and the job goes straight to Phase 2; otherwise the
# warm-up runs on CPU (fine for small KGs -- YAGO/WN warm up in minutes on a
# CPU node).
#
#   CPU_PARTITION=<name> DATASET=fb15k237 sbatch experiments/kgsage/slurm/train_cpu.slurm
#   CPU_PARTITION=<name> DATASET=wn18rr   sbatch experiments/kgsage/slurm/train_cpu.slurm
#   CPU_PARTITION=<name> DATASET=yago45   sbatch experiments/kgsage/slurm/train_cpu.slurm
#
# !! Set CPU_PARTITION to a real non-GPU partition -- run `sinfo -s` to list them
#    (Flinders DeepThought commonly has a 'general'/'hpc' CPU partition).
# For fb15k237/wn18rr an E'-donor (kgsage_<tag>_s0.pt) skips the warm-up if present;
# for a fresh dataset like yago45 there is no donor, so the warm-up runs on CPU.

# Historical job-name label, kept as-is: --output below is %x-%j.out.txt, so
# renaming it would change the log filenames already referenced in run notes.
#SBATCH --job-name=kgsage2_cpu
#SBATCH --output=%x-%j.out.txt
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --account=cse
# NOTE: --partition is set from $CPU_PARTITION below via `scontrol`-free re-exec
#       is not possible; instead pass it at submit time:
#         sbatch --partition="$CPU_PARTITION" experiments/kgsage/slurm/train_cpu.slurm
#       (SBATCH directives can't read env vars, so the partition MUST be on the
#        sbatch command line.)

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

# The locked recipe lives as constants in kgsage/gan/train.py; only operational
# knobs are env vars here. Runs stay SHORT (anchor-specificity peaks then
# erodes), snapshot every epoch, and select the epoch by knockout J@10 after
# the run.
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-8}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-1}"
CKPT_PATH="${CKPT_PATH:-experiments/kgsage/outputs/checkpoints/run_${TAG}_s${SEED}.pt}"
INIT_E="${INIT_E:-experiments/kgsage/outputs/checkpoints/kgsage_${TAG}_s${SEED}.pt}"

# Use the E'-donor to skip the Phase-1 warm-up if it exists; else warm up on CPU.
INIT_ARGS=()
if [[ -f "$INIT_E" ]]; then
  WARMUP_NOTE="Reuse context table E': $INIT_E   (Phase-1 RGCN warm-up SKIPPED)"
  INIT_ARGS=(--init_context_from "$INIT_E")
else
  WARMUP_NOTE="No E'-donor at $INIT_E -- running a fresh Phase-1 RGCN warm-up on CPU (OK for small KGs)."
fi

echo "Job $SLURM_JOB_ID on $(hostname) [$(sinfo -h -n $(hostname) -o %P 2>/dev/null)] $(date)"
echo "Dataset : $DATASET ($DATA_DIR)   CPU cores: $SLURM_CPUS_PER_TASK"
echo "$WARMUP_NOTE"
echo "Phase 2 : epochs=$EPOCHS snapshots=every $SNAPSHOT_EVERY  (recipe = constants in train.py)"
echo "Output  : $CKPT_PATH"
echo "----------------------------------------------------------------------"

module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || echo "WARNING: conda activate failed; using absolute python."
PY="$CONDA_ENV/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY missing." >&2; exit 1; }

# On HPC Linux use the allocated cores (multi-threaded). The OMP_NUM_THREADS=1
# workaround is a LOCAL-Windows segfault fix only -- do NOT set it here.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/experiments:$PYTHONPATH"
mkdir -p "$(dirname "$CKPT_PATH")"

$PY -m kgsage.gan.train \
    --data               "$DATA_DIR" \
    --out                "$CKPT_PATH" \
    "${INIT_ARGS[@]}" \
    --epochs             "$EPOCHS" \
    --snapshot_every     "$SNAPSHOT_EVERY" \
    --seed               "$SEED" \
    --device             cpu

echo "----------------------------------------------------------------------"
echo "Done $(date). Checkpoint: $CKPT_PATH"
