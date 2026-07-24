#!/usr/bin/env bash
# ONE ADKGD experiment-matrix cell (NEG_SOURCE x TEST_SOURCE) on AWS -- plain
# bash, no scheduler. Mirrors experiments/slurm/exp_cell.slurm.
#
#   NEG_SOURCE  in {random, gan}   -- training negatives
#   TEST_SOURCE in {random, gan}   -- injected eval anomalies
#
#   NEG_SOURCE=random TEST_SOURCE=random DATASET=FB15K-237 SEED=0 bash deploy/aws/run_cell.sh
#   NEG_SOURCE=gan TEST_SOURCE=gan DATASET=FB15K-237 SEED=0 \
#       GAN_CKPT=experiments/kgsage/outputs/checkpoints/run_fb15k237_s0.ep06.pt \
#       bash deploy/aws/run_cell.sh
set -eo pipefail

CONDA_ENV="${CONDA_ENV:-$HOME/envs/adkgd}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-FB15K-237}"
SEED="${SEED:-0}"
MAX_EPOCH="${MAX_EPOCH:-1}"
ANOMALY_RATIO="${ANOMALY_RATIO:-0.05}"
NEG_SOURCE="${NEG_SOURCE:-random}"
TEST_SOURCE="${TEST_SOURCE:-random}"
GAN_CKPT="${GAN_CKPT:-}"

echo "Host $(hostname) starting $(date)"
echo "Cell: dataset=$DATASET neg=$NEG_SOURCE test=$TEST_SOURCE seed=$SEED max_epoch=$MAX_EPOCH ratio=$ANOMALY_RATIO"
echo "----------------------------------------------------------------------"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || echo "WARNING: conda activate failed; using absolute python."
PY="$CONDA_ENV/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: $PY missing (run deploy/aws/bootstrap.sh first)." >&2; exit 1; }

# NB: deliberately do NOT export OMP_NUM_THREADS -- run_experiment.py setdefaults
# it to 1 (the mandatory ADKGD segfault fix). Setting it here would clobber that.
export PYTHONUNBUFFERED=1

CMD=( "$PY" experiments/run_experiment.py
      --dataset             "$DATASET"
      --seed                "$SEED"
      --max_epoch           "$MAX_EPOCH"
      --anomaly_ratio       "$ANOMALY_RATIO"
      --neg_source          "$NEG_SOURCE"
      --test_anomaly_source "$TEST_SOURCE" )

if [[ "$NEG_SOURCE" == "gan" || "$TEST_SOURCE" == "gan" ]]; then
    [[ -n "$GAN_CKPT" && -f "$GAN_CKPT" ]] || {
        echo "ERROR: GAN_CKPT must point at a trained generator for gan cells." >&2
        echo "  Train one first:  DATASET=... bash deploy/aws/run.sh" >&2
        exit 1; }
    CMD+=( --gan_path "$GAN_CKPT" )
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "----------------------------------------------------------------------"
echo "Done $(date). Aggregate all cells so far:"
echo "  $PY experiments/aggregate_results.py --dataset $DATASET"
