#!/bin/bash
# Train the KGSAGE Encoder (RGCN + DistMult) on FB15K-237.
#
# This is PHASE 1 of the thesis pipeline. The checkpoint produced here
# (entity + relation embeddings) feeds Phase 2 (KGSAGE Generator +
# Discriminator), which will get its own SLURM script later.
#
#   sbatch experiments/kgsage/slurm/train_encoder_fb15k237.slurm
#
# Prerequisites (run ONCE on the login node):
#   1. Conda env with torch+CUDA wheel (see RUNNING_ON_DEEPTHOUGHT.md).
#   2. PyTorch Geometric installed in that env:
#        pip install torch_geometric
#   3. FB15K-237 TSVs at data/FB15K-237/{train,valid,test}.txt
#   4. Phase 1 Test 1.3 (dataset audit) has passed on the login node:
#        python -m kgsage.cli.audit_dataset --dataset fb15k237
#
# Override hyperparameters via env vars (no script edits required):
#   EPOCHS=300 sbatch experiments/kgsage/slurm/train_encoder_fb15k237.slurm
#   DIM=256 BATCH_SIZE=1024 sbatch experiments/kgsage/slurm/train_encoder_fb15k237.slurm
#
# To train on a different dataset, use one of the sibling SLURM scripts:
#   experiments/kgsage/slurm/train_encoder_wn18rr.slurm   (when written)
#   experiments/kgsage/slurm/train_encoder_nell995.slurm  (when written)
# OR override DATASET via env var on this script (uses FB15K-237 hyperparams):
#   DATASET=wn18rr sbatch experiments/kgsage/slurm/train_encoder_fb15k237.slurm

#SBATCH --job-name=kgsage_train_encoder_fb15k237
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

# ---- Training knobs (override per submission via env vars) ----
DATASET="${DATASET:-fb15k237}"
CKPT_PATH="${CKPT_PATH:-experiments/kgsage/outputs/${DATASET}_encoder.pt}"
# Hyperparameter overrides — if unset, kgsage.data.datasets supplies the
# dataset's recommended default. Setting any of these forces an override.
EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
DIM="${DIM:-}"
N_LAYERS="${N_LAYERS:-}"
NUM_BASES="${NUM_BASES:-}"
MARGIN="${MARGIN:-}"
EVAL_EVERY="${EVAL_EVERY:-}"
SEED="${SEED:-0}"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Dataset      : $DATASET"
echo "Checkpoint   : $CKPT_PATH"
echo "Seed         : $SEED"
echo "Overrides    : EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE LR=$LR DIM=$DIM"
echo "               N_LAYERS=$N_LAYERS NUM_BASES=$NUM_BASES MARGIN=$MARGIN"
echo "               EVAL_EVERY=$EVAL_EVERY"
echo "               (empty = use dataset's recommended default)"
echo "----------------------------------------------------------------------"

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
# Force unbuffered Python output so training progress streams live to the log.
export PYTHONUNBUFFERED=1

# Fail fast if torch can't see the GPU.
$PY -c "import torch; assert torch.cuda.is_available(), 'torch cannot see a GPU: install the CUDA wheel'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| cuda', torch.version.cuda)"

# Fail fast if PyG isn't installed (KGSAGE-specific dependency).
# We import RGCNConv specifically because that's what the encoder needs.
$PY -c "from torch_geometric.nn import RGCNConv; import torch_geometric; print('PyG OK | torch_geometric', torch_geometric.__version__)"

cd "$PROJECT_DIR"

# Make `import kgsage` resolve to experiments/kgsage/ without pip-installing.
export PYTHONPATH="$PROJECT_DIR/experiments:$PYTHONPATH"

mkdir -p "$(dirname "$CKPT_PATH")"

# Build the python command. Optional hyperparameter overrides only get added
# to the command line if the env var is non-empty — empty means "use the
# dataset's recommended default from kgsage.data.datasets".
CMD=( "$PY" -m kgsage.cli.train_encoder
      --dataset "$DATASET"
      --out     "$CKPT_PATH"
      --seed    "$SEED"
      --device  cuda )

[[ -n "$EPOCHS"     ]] && CMD+=( --epochs     "$EPOCHS" )
[[ -n "$BATCH_SIZE" ]] && CMD+=( --batch_size "$BATCH_SIZE" )
[[ -n "$LR"         ]] && CMD+=( --lr         "$LR" )
[[ -n "$MARGIN"     ]] && CMD+=( --margin     "$MARGIN" )
[[ -n "$DIM"        ]] && CMD+=( --dim        "$DIM" )
[[ -n "$N_LAYERS"   ]] && CMD+=( --n_layers   "$N_LAYERS" )
[[ -n "$NUM_BASES"  ]] && CMD+=( --num_bases  "$NUM_BASES" )
[[ -n "$EVAL_EVERY" ]] && CMD+=( --eval_every "$EVAL_EVERY" )

echo "[$(date)] Training KGSAGE Encoder + DistMult Decoder -> $CKPT_PATH"
echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "----------------------------------------------------------------------"
echo "Done $(date). Checkpoint: $CKPT_PATH"
echo ""
echo "Next steps (run on the LOGIN NODE, not as another SLURM job — they're cheap):"
echo ""
echo "  Test 1.1 — Link prediction MRR (Pass: MRR >= dataset's expected_mrr)"
echo "    cd \$PROJECT_DIR"
echo "    PYTHONPATH=\$PROJECT_DIR/experiments \$PY -m kgsage.cli.evaluate_encoder \\"
echo "        --dataset $DATASET \\"
echo "        --ckpt    $CKPT_PATH"
echo ""
echo "  Test 1.2 — Anti-symmetric signal in relation embeddings (Pass: p < 0.05)"
echo "    PYTHONPATH=\$PROJECT_DIR/experiments \$PY -m kgsage.cli.audit_embeddings \\"
echo "        --dataset $DATASET \\"
echo "        --ckpt    $CKPT_PATH"
echo ""
echo "  (Both tests run in seconds on a single GPU. If you want to be thorough"
echo "   about it, scp the checkpoint to your laptop and run there.)"
echo ""
echo "Phase 1 complete once all three Phase 1 tests pass. Phase 2 (KGSAGE"
echo "Generator + Discriminator) will get its own SLURM script later — it"
echo "loads this checkpoint as initialisation."
