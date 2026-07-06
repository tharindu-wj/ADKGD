# KGSAGE SLURM scripts

HPC launchers for the generator side. They live inside `kgsage/` (not
`experiments/slurm/`) because KGSAGE is package-portable; the ADKGD-side
cell launcher stays in `experiments/slurm/exp_cell.slurm`.

| script | what it does |
|---|---|
| `train.slurm` | trains the KGSAGE generator (`kgsage.gan.train`). `DATASET=fb15k237\|wn18rr SEED=n sbatch …` — derives data dir + frozen-LP artifacts from `DATASET`; fails fast if `kgsage.cli.fetch_lp` has not been run on the login node |

After a training job finishes, run the checkpoint through
`python -m kgsage.cli.inspect_gan_lp` (or corrupt your own triples with
`python -m kgsage.cli.corrupt`) before spending matrix compute, then launch
cells: `NEG_SOURCE=gan GAN_CKPT=<the .pt> … sbatch experiments/slurm/exp_cell.slurm`.
