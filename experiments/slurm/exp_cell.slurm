#!/bin/bash
# ONE experiment-matrix cell: (NEG_SOURCE x TEST_SOURCE) on DATASET at SEED.
# Generic replacement for the exp1-4 scripts, adding the lp_band source.
#
#   NEG_SOURCE  in {random, lp_band, gan}   -- training negatives
#   TEST_SOURCE in {random, lp_band, gan}   -- injected eval anomalies
#
# Examples (the PoC readout grid; see OPTION_B_PLAN.md pre-registered criteria):
#   NEG_SOURCE=random  TEST_SOURCE=random  SEED=0 sbatch experiments/slurm/exp_cell.slurm
#   NEG_SOURCE=random  TEST_SOURCE=lp_band SEED=0 sbatch ...   # "the gap"
#   NEG_SOURCE=lp_band TEST_SOURCE=lp_band SEED=0 sbatch ...   # "the recovery"
#   NEG_SOURCE=lp_band TEST_SOURCE=random  SEED=0 sbatch ...   # "no regression"
#   NEG_SOURCE=gan     TEST_SOURCE=lp_band SEED=0 GAN_CKPT=... sbatch ...  # A-ii off-diagonal
#
# Env knobs: DATASET (FB15K-237|WN18RR), SEED, MAX_EPOCH, ANOMALY_RATIO,
#   GAN_CKPT (required when a source is 'gan'). LP paths are derived from
#   DATASET by run_experiment.py itself.
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
    [[ -n "$GAN_CKPT" && -f "$GAN_CKPT" ]] || {
        echo "ERROR: GAN_CKPT must point at a trained A-ii checkpoint for gan cells." >&2
        echo "  Train one first: DATASET=... sbatch experiments/kgsage/slurm/train_aii.slurm" >&2
        exit 1; }
    CMD+=( --gan_path "$GAN_CKPT" )
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "----------------------------------------------------------------------"
echo "Done $(date). Aggregate all cells so far:"
echo "  $PY experiments/aggregate_results.py --dataset $DATASET"
