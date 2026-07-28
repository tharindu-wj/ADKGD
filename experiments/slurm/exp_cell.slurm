#!/bin/bash
# ONE experiment-matrix cell: (NEG_SOURCE x TEST_SOURCE) on DATASET at SEED.
# The single matrix-cell launcher (former exp1-4 named cells map to
# NEG_SOURCE/TEST_SOURCE pairs below; those scripts were removed).
#
#   NEG_SOURCE  in {random, gan}   -- training negatives
#   TEST_SOURCE in {random, gan}   -- injected eval anomalies
#
# "negative" and "anomaly" are the detector's names for what KGSAGE emits as a
# CORRUPTION -- same object, different side of the seam. These env var names,
# their random|gan values and GAN_CKPT are a frozen contract with
# run_experiment.py's --neg_source/--test_anomaly_source/--gan_path flags.
#
# The 2x2 matrix (rule-based = random, KGSAGE = gan):
#   NEG_SOURCE=random TEST_SOURCE=random SEED=0 sbatch experiments/slurm/exp_cell.slurm   # baseline
#   NEG_SOURCE=random TEST_SOURCE=gan    SEED=0 GAN_CKPT=... sbatch ...                    # the gap
#   NEG_SOURCE=gan    TEST_SOURCE=gan    SEED=0 GAN_CKPT=... sbatch ...                    # proposed
#   NEG_SOURCE=gan    TEST_SOURCE=random SEED=0 GAN_CKPT=... sbatch ...                    # cross-check
#
# Env knobs: DATASET (FB15K-237|WN18RR), SEED, MAX_EPOCH, ANOMALY_RATIO,
#   GAN_CKPT (required when a source is 'gan'; use the locked generator_<ds>.pt).
# Results: RESULTS table + AUC/AUPRC in this job's .out.txt, plus a per-run
#   JSON under checkpoints/<dataset>/ for experiments/aggregate_results.py.

#SBATCH --job-name=adkgd_cell
#SBATCH --output=%x-%j.out.txt
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --account=cse

set -eo pipefail

PROJECT_DIR="$HOME/ADKGD"
CONDA_ENV="$HOME/envs/adkgd"

DATASET="${DATASET:-FB15K-237}"
SEED="${SEED:-0}"
MAX_EPOCH="${MAX_EPOCH:-1}"
ANOMALY_RATIO="${ANOMALY_RATIO:-0.05}"
NEG_SOURCE="${NEG_SOURCE:-random}"
TEST_SOURCE="${TEST_SOURCE:-random}"
GAN_CKPT="${GAN_CKPT:-}"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Cell: dataset=$DATASET  neg=$NEG_SOURCE  test=$TEST_SOURCE  seed=$SEED  max_epoch=$MAX_EPOCH  ratio=$ANOMALY_RATIO"
echo "----------------------------------------------------------------------"

module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || echo "WARNING: conda activate failed; using absolute python path below."
PY="$CONDA_ENV/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY missing." >&2; exit 1; }

export PYTHONUNBUFFERED=1
cd "$PROJECT_DIR"

CMD=( "$PY" experiments/run_experiment.py
      --dataset             "$DATASET"
      --seed                "$SEED"
      --max_epoch           "$MAX_EPOCH"
      --anomaly_ratio       "$ANOMALY_RATIO"
      --neg_source          "$NEG_SOURCE"
      --test_anomaly_source "$TEST_SOURCE" )

if [[ "$NEG_SOURCE" == "gan" || "$TEST_SOURCE" == "gan" ]]; then
    # KGSAGE is a SEPARATE repository. This repo never trains a generator; it
    # consumes a checkpoint copied into artifacts/kgsage/ (see its README).
    [[ -n "$GAN_CKPT" && -f "$GAN_CKPT" ]] || {
        echo "ERROR: GAN_CKPT must point at a trained KGSAGE generator checkpoint for gan cells." >&2
        echo "  Got: '${GAN_CKPT:-<unset>}'" >&2
        echo "  Train one in the KGSAGE repo, then copy it here:" >&2
        echo "    cd \$KGSAGE_HOME && sbatch slurm/train.slurm      # DATASET=$DATASET SEED=$SEED" >&2
        echo "    cp \$KGSAGE_HOME/outputs/checkpoints/run_${DATASET}_s${SEED}.pt $PROJECT_DIR/artifacts/kgsage/" >&2
        echo "  Then resubmit with GAN_CKPT=artifacts/kgsage/run_${DATASET}_s${SEED}.pt" >&2
        exit 1; }
    CMD+=( --gan_path "$GAN_CKPT" )
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "----------------------------------------------------------------------"
echo "Done $(date). Aggregate all cells so far:"
echo "  $PY experiments/aggregate_results.py --dataset $DATASET"
