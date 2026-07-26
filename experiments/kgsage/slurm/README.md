# KGSAGE SLURM scripts

HPC launchers for the generator side — they run Phase 1 (Neighbourhood Context
Encoding) and Phase 2 (Adversarial Generator Training) in one job. They live
inside `kgsage/` (not `experiments/slurm/`) because KGSAGE is package-portable;
the ADKGD-side cell launcher stays in `experiments/slurm/exp_cell.slurm`.

| script | what it does |
|---|---|
| `train.slurm` | trains the generator on a GPU node (`kgsage.gan.train`). `DATASET=fb15k237\|wn18rr\|yago45 SEED=n sbatch …` — derives the data dir from `DATASET`; saves a snapshot every adversarial epoch |
| `train_cpu.slurm` | CPU-node variant that skips the Phase-1 RGCN warm-up when a cached-E' donor exists (`--init_context_from`), and otherwise warms up on CPU; pass the partition on the `sbatch` command line |

The `DATASET` values (`fb15k237`/`wn18rr`/`yago45`) and the data directories
they select (`data/FB15K-237`/`data/WN18RR`/`data/YAGO4.5`) are frozen literals
shared with the dataset registry — change the `case` labels and lookups break.

After a training job finishes, select the snapshot with
`python experiments/kgsage/cli/knockout_eval.py --ckpt <each .epNN.pt> --data …`
(lowest mean knockout J@10 wins — the most anchor-specific generator), promote
the winner to `generator_<dataset>.pt`, then launch cells:
`NEG_SOURCE=gan GAN_CKPT=<the .pt> … sbatch experiments/slurm/exp_cell.slurm`
(`NEG_SOURCE`/`GAN_CKPT` are a frozen contract with `run_experiment.py`).

Keep each job's `.out.txt`: the `corr-pick=` / `alpha=` / `D-acc=` /
`dm-online=` trajectories are the training-health record. Those log tokens are
frozen for comparability with every run already collected — `corr-pick=` is the
**corroborated**-pick fraction (not "corrupted"), `D-acc=` is the plausibility
discriminator's accuracy on real versus generated triples, and `dm-online=` /
`g_match=` belong to the neighbourhood discriminator (the BCE of its online
update, and its mean fit score for the candidates the generator picked).
