# KGSAGE SLURM scripts

HPC launchers for the generator side. They live inside `kgsage/` (not
`experiments/slurm/`) because KGSAGE is package-portable; the ADKGD-side
cell launchers stay in `experiments/slurm/`.

| script | what it trains | typical use |
|---|---|---|
| `train_aii.slurm` | **PRIMARY**: the A-ii generator (`kgsage.gan.train_aii`) | `DATASET=fb15k237\|wn18rr SEED=n sbatch …` — derives data dir + frozen-LP artifacts from `DATASET`; fails fast if `kgsage.cli.fetch_lp` has not been run on the login node |
| `train_gan_fb15k237.slurm` | legacy B1a GAN (ablation arm) | env knobs in header |
| `train_gan_wn18rr.slurm` | legacy B1a GAN (ablation arm) | env knobs in header |

After a generator job finishes, run the checkpoint through
`python -m kgsage.cli.inspect_gan_lp` before spending matrix compute, then
launch cells with `experiments/slurm/exp_cell.slurm`
(`NEG_SOURCE=gan GAN_CKPT=<the .pt> …`).
