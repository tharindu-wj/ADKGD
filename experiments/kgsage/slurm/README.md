# KGSAGE SLURM scripts

HPC job launchers for the KGSAGE pipeline. These scripts live here (inside
the `kgsage/` package) rather than alongside the ADKGD-side launchers in
`experiments/slurm/`, because KGSAGE is package-portable — when KGSAGE is
eventually released as a standalone library, this folder ships with it.

## Available scripts

| Script | Dataset | Submit with |
|---|---|---|
| `train_gan_fb15k237.slurm` | FB15K-237 | `sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm` |

## SLURM scripts NOT in this folder

ADKGD-side jobs (which use KGSAGE only indirectly, via the bridge in
`experiments/kgsage_bridge/`) live in `experiments/slurm/`:

| Script | Purpose |
|---|---|
| `run_baseline_fb15k237.slurm`             | ADKGD baseline (random negatives) on FB15K-237 |
| `run_baseline_wn18rr.slurm`               | ADKGD baseline on WN18RR |
| `run_baseline_with_kgsage_fb15k237.slurm` | ADKGD + KGSAGE GAN negatives on FB15K-237 |

Why the split:
- `kgsage/slurm/` jobs only touch the KGSAGE package; portable.
- `experiments/slurm/` jobs invoke ADKGD's training pipeline; require both worlds.

## Overriding hyperparameters

All KGSAGE SLURM scripts accept env-var overrides. Set the var on the
`sbatch` line; the script picks it up. Example:

```bash
# Train with 600 epochs:
EPOCHS=600 sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm

# Train with a custom checkpoint name (useful for ablations):
CKPT_PATH=experiments/kgsage/outputs/checkpoints/fb15k237_seed1.pt \
    SEED=1 sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm
```

## Editing paths

Each script has two paths at the top you'll need to set once:

```bash
PROJECT_DIR="$HOME/ADKGD"           # repo checkout location
CONDA_ENV="$HOME/envs/adkgd"        # conda env with torch (+CUDA wheel)
```

See `../../RUNNING_ON_DEEPTHOUGHT.md` for HPC setup details.
