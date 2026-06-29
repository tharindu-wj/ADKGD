#!/bin/bash
# Train the PHASE 2 pair-aware KGSAGE GAN (KGSAGEGenerator + KGSAGEDiscriminator)
# on FB15K-237 — one V100 allocation.
#
# This is the ROLE-SWAP contradiction generator. It loads the Phase 1 encoder
# checkpoint (post-RGCN entity embeddings + DistMult relation embeddings),
# conditions the generator on them, and trains adversarially against the
# partner-template positives. Output: experiments/kgsage/outputs/checkpoints/
# kgsage_fb15k237.pt + the Test 2.2 / 2.3 generation probes in the log.
#
#   sbatch experiments/kgsage/slurm/train_kgsage_gan_fb15k237.slurm
#
# This is DISTINCT from train_gan_fb15k237.slurm, which trains the SIMPLE
# single-slot-corruption GAN (ADKGD's --neg_source gan baseline).
#
# Prerequisites (run ONCE on the login node):
#   1. Conda env with torch+CUDA wheel (see RUNNING_ON_DEEPTHOUGHT.md).
#   2. PyTorch Geometric installed in that env (the encoder-embedding loader
#      imports RGCNConv):
#        pip install torch_geometric
#   3. FB15K-237 TSVs at data/FB15K-237/{train,valid,test}.txt.
#   4. PHASE 1 ENCODER ALREADY TRAINED — its checkpoint must exist:
#        experiments/kgsage/outputs/fb15k237_encoder.pt
#      (produced by train_encoder_fb15k237.slurm; Tests 1.1/1.2 should pass.)
#
# Override hyperparameters via env vars (no script edits required):
#   EPOCHS=100 BATCH_SIZE=1024 sbatch .../train_kgsage_gan_fb15k237.slurm
#   ENTROPY_WEIGHT=0.1 sbatch .../train_kgsage_gan_fb15k237.slurm   # if Test 2.2 mode-collapses
#   TEMPLATE=observed sbatch .../train_kgsage_gan_fb15k237.slurm    # v1 positives instead of v2-A

#SBATCH --job-name=kgsage_train_pairgan_fb15k237
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
DATASET_DIR="${DATASET_DIR:-data/FB15K-237}"
ENCODER_CKPT="${ENCODER_CKPT:-experiments/kgsage/outputs/fb15k237_encoder.pt}"
CKPT_PATH="${CKPT_PATH:-experiments/kgsage/outputs/checkpoints/kgsage_fb15k237.pt}"
EPOCHS="${EPOCHS:-50}"
# 512 is a safe V100 default (fits 32GB HBM2, good GPU utilisation). Bump to
# 1024 for more throughput; back off to 256 if the model grows and OOMs.
BATCH_SIZE="${BATCH_SIZE:-512}"
# Optional overrides — if unset, the trainer's defaults apply (template=rule,
# min_support=MIN_SUPPORT=100, recon_weight=1.0, entropy_weight=0.0, tau=0.5).
TEMPLATE="${TEMPLATE:-}"
MIN_SUPPORT="${MIN_SUPPORT:-}"
RECON_WEIGHT="${RECON_WEIGHT:-}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-}"
TAU="${TAU:-}"
EVAL_SAMPLES="${EVAL_SAMPLES:-}"
SEED="${SEED:-0}"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Dataset       : $DATASET_DIR"
echo "Encoder ckpt  : $ENCODER_CKPT"
echo "Output ckpt   : $CKPT_PATH"
echo "Epochs        : $EPOCHS    Batch size: $BATCH_SIZE    Seed: $SEED"
echo "Overrides     : TEMPLATE=$TEMPLATE MIN_SUPPORT=$MIN_SUPPORT RECON_WEIGHT=$RECON_WEIGHT"
echo "                ENTROPY_WEIGHT=$ENTROPY_WEIGHT TAU=$TAU EVAL_SAMPLES=$EVAL_SAMPLES"
echo "                (empty = use the trainer's default)"
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
# Force unbuffered Python output so training progress streams live to the log.
export PYTHONUNBUFFERED=1

# Fail fast if torch can't see the GPU.
$PY -c "import torch; assert torch.cuda.is_available(), 'torch cannot see a GPU: install the CUDA wheel'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| cuda', torch.version.cuda)"

# Fail fast if PyG isn't installed — load_encoder_embeddings imports RGCNConv.
$PY -c "from torch_geometric.nn import RGCNConv; import torch_geometric; print('PyG OK | torch_geometric', torch_geometric.__version__)"

cd "$PROJECT_DIR"

# Make `import kgsage` resolve to experiments/kgsage/ without pip-installing.
export PYTHONPATH="$PROJECT_DIR/experiments:$PYTHONPATH"

# Fail fast if the PHASE 1 encoder checkpoint is missing — Phase 2 cannot run
# without it. (Run train_encoder_fb15k237.slurm first.)
if [[ ! -f "$ENCODER_CKPT" ]]; then
    echo "ERROR: encoder checkpoint not found: $ENCODER_CKPT" >&2
    echo "       Train Phase 1 first:" >&2
    echo "         sbatch experiments/kgsage/slurm/train_encoder_fb15k237.slurm" >&2
    echo "       (or set ENCODER_CKPT=<path> if your checkpoint lives elsewhere)." >&2
    exit 1
fi

mkdir -p "$(dirname "$CKPT_PATH")"

# Build the python command. Optional hyperparameter overrides only get appended
# if the env var is non-empty — empty means "use the trainer's default".
CMD=( "$PY" -m kgsage.cli.train_kgsage_gan
      --data         "$DATASET_DIR"
      --encoder_ckpt "$ENCODER_CKPT"
      --out          "$CKPT_PATH"
      --epochs       "$EPOCHS"
      --batch_size   "$BATCH_SIZE"
      --seed         "$SEED"
      --device       cuda )

[[ -n "$TEMPLATE"       ]] && CMD+=( --template       "$TEMPLATE" )
[[ -n "$MIN_SUPPORT"    ]] && CMD+=( --min_support    "$MIN_SUPPORT" )
[[ -n "$RECON_WEIGHT"   ]] && CMD+=( --recon_weight   "$RECON_WEIGHT" )
[[ -n "$ENTROPY_WEIGHT" ]] && CMD+=( --entropy_weight "$ENTROPY_WEIGHT" )
[[ -n "$TAU"            ]] && CMD+=( --tau            "$TAU" )
[[ -n "$EVAL_SAMPLES"   ]] && CMD+=( --eval_samples   "$EVAL_SAMPLES" )

echo "[$(date)] Training Phase 2 pair-aware KGSAGE GAN -> $CKPT_PATH"
echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "----------------------------------------------------------------------"
echo "Done $(date). Checkpoint: $CKPT_PATH"
echo ""
echo "Inspect the Test 2.2 (partner-relation entropy > 1.5 nats) and Test 2.3"
echo "(syntactic validity >= 95%) lines printed above. If Test 2.2 reports mode"
echo "collapse (entropy < 0.5), resubmit with ENTROPY_WEIGHT=0.1."
echo ""
echo "Next steps:"
echo "  * Phase 3: implement + run kgsage.cli.evaluate_gan (generation quality)."
echo "  * Feed ADKGD: update kgsage_bridge.bridge to load this checkpoint via"
echo "    inference.load_kgsage_checkpoint, then flip the generate_contradictions"
echo "    alias to generate_kgsage_partners."
