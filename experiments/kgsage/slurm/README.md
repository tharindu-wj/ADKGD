# KGSAGE SLURM scripts

HPC job launchers for the KGSAGE pipeline. These scripts live here (inside
the `kgsage/` package) rather than alongside the ADKGD-side launchers in
`experiments/slurm/`, because KGSAGE is package-portable — when KGSAGE is
eventually released as a standalone library, this folder ships with it.

## Available scripts

| Script | Phase | Dataset | Submit with |
|---|---|---|---|
| `train_encoder_fb15k237.slurm` | Phase 1 — encoder pretraining | FB15K-237 | `sbatch experiments/kgsage/slurm/train_encoder_fb15k237.slurm` |

## Future scripts (Phase 2+)

| Script | Phase | Dataset |
|---|---|---|
| `train_encoder_wn18rr.slurm` | Phase 1 — encoder | WN18RR |
| `train_encoder_nell995.slurm` | Phase 1 — encoder | NELL-995 |
| `train_gan_fb15k237.slurm` | Phase 2 — pair-aware GAN | FB15K-237 |
| `train_gan_wn18rr.slurm` | Phase 2 — pair-aware GAN | WN18RR |
| `train_gan_nell995.slurm` | Phase 2 — pair-aware GAN | NELL-995 |

## SLURM scripts NOT in this folder

ADKGD-side jobs (which use KGSAGE only indirectly, via the bridge in
`experiments/kgsage_bridge/`) live in `experiments/slurm/`:

| Script | Purpose |
|---|---|
| `run_baseline_fb15k237.slurm` | ADKGD baseline (random negatives) |
| `run_baseline_with_gan_fb15k237.slurm` | ADKGD + simple conditional GAN |
| `run_adkgd_with_kgsage_fb15k237.slurm` | ADKGD + KGSAGE (future, Phase 4) |

Why the split:
- `kgsage/slurm/` jobs only touch the KGSAGE package; portable.
- `experiments/slurm/` jobs invoke ADKGD's training pipeline; not portable.

## Overriding hyperparameters

All KGSAGE SLURM scripts accept env-var overrides. Set the var on the
`sbatch` line; the script picks it up. Example:

```bash
# Train with 300 epochs instead of the dataset default:
EPOCHS=300 sbatch experiments/kgsage/slurm/train_encoder_fb15k237.slurm

# Train with a custom checkpoint name (useful for ablations):
CKPT_PATH=experiments/kgsage/outputs/fb15k237_encoder_seed1.pt \
    SEED=1 sbatch experiments/kgsage/slurm/train_encoder_fb15k237.slurm
```

Leave any override empty to inherit the dataset's recommended default
from `kgsage/data/datasets.py`.

## Editing paths

Each script has two paths at the top you'll need to set once:

```bash
PROJECT_DIR="$HOME/ADKGD"           # repo checkout location
CONDA_ENV="$HOME/envs/adkgd"        # conda env with torch+CUDA+PyG
```

See `../../RUNNING_ON_DEEPTHOUGHT.md` for HPC setup details.
