# KGSAGE SLURM scripts

HPC launchers for the generator side. They live inside `kgsage/` (not
`experiments/slurm/`) because KGSAGE is package-portable; the ADKGD-side
cell launcher stays in `experiments/slurm/exp_cell.slurm`.

| script | what it does |
|---|---|
| `train.slurm` | trains the generator on a GPU node (`kgsage.gan.train`). `DATASET=fb15k237\|wn18rr SEED=n sbatch …` — derives the data dir from `DATASET`; saves a snapshot every adversarial epoch |
| `train_cpu.slurm` | CPU-node variant that skips the RGCN warm-up by reusing a cached E' (`--init_context_from`); pass the partition on the `sbatch` command line |

After a training job finishes, select the snapshot with
`python experiments/kgsage/cli/knockout_eval.py --ckpt <each .epNN.pt> --data …`
(lowest mean knockout J@10 wins), promote the winner to `generator_<dataset>.pt`,
then launch cells: `NEG_SOURCE=gan GAN_CKPT=<the .pt> … sbatch experiments/slurm/exp_cell.slurm`.
