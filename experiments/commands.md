# ADKGD — copy-paste command reference

Detector-side commands only. **This repo does not train a generator.**

KGSAGE is a separate package in its own repository. Producing a checkpoint —
adding a dataset (including the YAGO 4.5 conversion), Phase 1–2 training,
snapshot selection by knockout J@10, and the corruption CSVs used for the
qualitative arms — is documented there, in its `README.md` and `slurm/README.md`.

The hand-off between the two repos is a single file: a `.pt` checkpoint copied
into `artifacts/kgsage/`. See [`artifacts/kgsage/README.md`](../artifacts/kgsage/README.md).

Names follow `experiments/docs/KGSAGE_glossary.md`.

## 0. One-time setup

```bash
# Only needed for the `gan` cells. The `random` baseline needs nothing.
pip install -e /path/to/kgsage

# Then copy in a trained generator (produced in the KGSAGE repo):
cp /path/to/kgsage/outputs/checkpoints/run_fb15k237_s0.ep06.pt artifacts/kgsage/
```

## 1. The 2x2 matrix

Detector-side vocabulary: the corruptions arrive as **negatives** (training) and
**anomalies** (evaluation). `NEG_SOURCE`/`TEST_SOURCE`/`GAN_CKPT` and the
`random`|`gan` values are a frozen contract between `exp_cell.slurm`,
`run_experiment.py` and the detector — do not rename them.

```bash
NEG_SOURCE=<random|gan> TEST_SOURCE=<random|gan> DATASET=<FB15K-237|WN18RR> \
    SEED=0 GAN_CKPT=artifacts/kgsage/run_fb15k237_s0.ep06.pt \
    sbatch experiments/slurm/exp_cell.slurm

python experiments/aggregate_results.py --dataset FB15K-237
```

`GAN_CKPT` is read only for `gan` cells. `KGSAGE_CKPT` in the environment
overrides `--gan_path`, which is how a job points at a checkpoint on scratch
without editing the launcher.

The four cells:

| Train / Test | random | gan |
|---|---|---|
| **random** | Exp 1 (baseline) | Exp 3 (gap) |
| **gan** | Exp 2 (cross-check) | Exp 4 (proposed) |

## 2. Baseline without KGSAGE

Exp 1 needs neither the package nor a checkpoint — useful for confirming the two
repos are genuinely decoupled:

```bash
NEG_SOURCE=random TEST_SOURCE=random DATASET=WN18RR SEED=0 \
    sbatch experiments/slurm/exp_cell.slurm
```

## Smoke test

```bash
# Bridge round-trip (needs kgsage installed + a checkpoint in artifacts/kgsage/)
PYTHONPATH=experiments python experiments/kgsage_bridge/smoke_test.py
```
