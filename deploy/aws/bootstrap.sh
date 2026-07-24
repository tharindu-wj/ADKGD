#!/usr/bin/env bash
# One-time setup on a fresh g4dn.xlarge (recommended AMI: AWS Deep Learning Base
# OSS Nvidia Driver GPU AMI, Ubuntu 22.04 -- ships the NVIDIA driver + CUDA, so
# you never hand-install the driver). Reproduces the DeepThought environment:
#   ~/ADKGD           (code + committed data, from git)
#   ~/envs/adkgd      (conda env: py3.11, torch cu121, torch_geometric, sklearn)
#   run.sh / run_cell.sh made executable (plain-bash runners -- no SLURM)
#
#   curl -fsSL https://raw.githubusercontent.com/tharindu-wj/ADKGD/dev_gan_2/deploy/aws/bootstrap.sh | bash
# or, if you cloned first:
#   bash ~/ADKGD/deploy/aws/bootstrap.sh
#
# The renamed KGSAGE-2 launchers (train.slurm, kgsage.gan.train) live on the
# dev_gan_2 branch -- BRANCH pins the clone/checkout to it. Push that branch
# (with deploy/ committed) to origin before running the curl one-liner.
set -euo pipefail

REPO="${REPO:-https://github.com/tharindu-wj/ADKGD.git}"
BRANCH="${BRANCH:-dev_gan_2}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/ADKGD}"
CONDA_ENV="${CONDA_ENV:-$HOME/envs/adkgd}"
CUDA_WHL="${CUDA_WHL:-cu121}"   # T4 (Turing) on the DLAMI driver -> cu121, matching DeepThought

echo "=== ADKGD AWS bootstrap ==="
echo "repo=$REPO  branch=$BRANCH  project=$PROJECT_DIR  env=$CONDA_ENV  wheel=$CUDA_WHL"

# 0) conda present? (DLAMI ships it.) If not, install Miniconda and init bash.
if ! command -v conda >/dev/null 2>&1; then
  echo "--- conda not found; installing Miniconda ---"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
  "$HOME/miniconda3/bin/conda" init bash
  export PATH="$HOME/miniconda3/bin:$PATH"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1) code + committed data (~31MB: FB15K-237 + WN18RR are tracked in the repo)
if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  echo "--- cloning $REPO ($BRANCH) -> $PROJECT_DIR ---"
  git clone --branch "$BRANCH" --single-branch "$REPO" "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

# 2) env -- mirror RUNNING_ON_DEEPTHOUGHT.md step 1 (EC2 has internet, so no
#    login-node pre-staging dance is needed; install straight from PyPI).
if [[ ! -x "$CONDA_ENV/bin/python" ]]; then
  echo "--- creating conda env $CONDA_ENV (python 3.11) ---"
  conda create -y -p "$CONDA_ENV" python=3.11
fi
conda activate "$CONDA_ENV"
python -m pip install --upgrade pip
python -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_WHL}"
python -m pip install torch_geometric
python -m pip install numpy scikit-learn matplotlib

# 3) make the AWS runners executable (strip any CRLF from a Windows clone first)
RUN_DIR="$PROJECT_DIR/deploy/aws"
sed -i 's/\r$//' "$RUN_DIR"/*.sh 2>/dev/null || true
chmod +x "$RUN_DIR"/*.sh

# 4) prove the box can actually run the job (same asserts train.slurm makes)
echo "--- verifying GPU + PyG ---"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU (wrong CUDA wheel or driver)"
cc = torch.cuda.get_device_capability(0)
print(f"GPU: {torch.cuda.get_device_name(0)} | capability {cc} "
      f"| torch {torch.__version__} | cuda build {torch.version.cuda}")
assert cc >= (7, 0), f"compute capability {cc} is below PyTorch 2.x support (need Turing 7.5+; T4 is fine)"
from torch_geometric.nn import RGCNConv  # noqa: F401
print("PyG OK (RGCNConv importable)")
PY

cat <<EOF
----------------------------------------------------------------------
Bootstrap complete.

Open a NEW shell (picks up conda), then:
  conda activate $CONDA_ENV
  cd $PROJECT_DIR
  DATASET=fb15k237 bash deploy/aws/run.sh

If RGCNConv failed to import, your torch build needs PyG's compiled extensions:
  TVER=\$(python -c "import torch;print(torch.__version__.split('+')[0])")
  pip install pyg-lib torch-scatter -f https://data.pyg.org/whl/torch-\${TVER}+${CUDA_WHL}.html
----------------------------------------------------------------------
EOF
