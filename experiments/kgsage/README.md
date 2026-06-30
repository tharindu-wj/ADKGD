# KGSAGE — Knowledge Graph Synthetic Anomaly Generator

A standalone-ready Python package that trains a conditional GAN to generate
synthetic knowledge-graph anomalies (single-slot-corruption negatives), used to
train and evaluate per-triple anomaly detectors such as ADKGD.

## What this package does in one paragraph

A conditional GAN consumes a real triple together with a noise vector and
produces a fake-but-plausible triple — same shape, slightly wrong content (one
slot: head, relation, or tail). The generator learns its own entity/relation
embedding tables; the discriminator scores a `(real, candidate)` pair. Replacing
ADKGD's uniform-random corrupter with this learned generator yields harder,
type-consistent negatives.

ADKGD integration (running the detector with KGSAGE-generated negatives) lives
in the sibling folder `experiments/kgsage_bridge/`, not in this package —
keeping `kgsage/` ADKGD-agnostic and standalone-extractable.

## Package layout

```
kgsage/
├── __init__.py             <- public API (load_kg, KGSAGEGenerator, generate_negatives, ...)
├── README.md               <- this file
│
├── data/                   <- KG loading + dataset registry
│   ├── loaders.py          <- load_kg(path) -> integer triples + vocab maps
│   └── datasets.py         <- KNOWN_DATASETS + resolve_dataset()
│
├── gan/                    <- the conditional GAN
│   ├── models.py           <- KGSAGEGenerator (3-head) + KGSAGEDiscriminator + helpers
│   └── train.py            <- adversarial training loop (BCE + reconstruction)
│
├── inference.py            <- generation API: generate_negatives / load_checkpoint / render_stats
│
├── cli/                    <- command-line entry points (thin shims)
│   └── train_gan.py
│
└── slurm/                  <- HPC job launchers
    ├── README.md
    └── train_gan_fb15k237.slurm
```

## How it works

**Generator** `KGSAGEGenerator(h, r, t, z)` — looks up the triple's embeddings,
concatenates the noise `z`, runs a 2-layer MLP, and emits three logit heads:
one over entities (new head), one over relations, one over entities (new tail).

**Discriminator** `KGSAGEDiscriminator(real_emb, candidate_emb)` — scores whether
the candidate looks like a real fact given the real triple.

**Training** (`gan/train.py`): for every real triple, a target is built by
randomly corrupting one slot. The discriminator learns to tell the real-vs-target
pair from the real-vs-generated pair (BCE); the generator's loss is the
adversarial term plus a `10 ×` cross-entropy reconstruction term against the
target. Gumbel-Softmax keeps the categorical sampling differentiable.

**Inference** (`inference.py::generate_negatives`): per real triple, pick a slot
at random, mask the original value, Gumbel-argmax a replacement, and reject any
candidate that is a self-loop or already in the real graph (retry, then fall
back to uniform random). One negative per input, in ADKGD's ID space.

## Quick start

```bash
# Make `import kgsage` work without pip-installing
export PYTHONPATH="$(pwd)/experiments:$PYTHONPATH"

# Train the GAN (dummy KG, CPU, ~minutes)
python -m kgsage.cli.train_gan \
    --data data/dummy_kg \
    --epochs 50 \
    --device cpu \
    --out experiments/kgsage/outputs/checkpoints/kgsage_dummy.pt

# FB15K-237 on HPC (V100)
sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm
```

## ADKGD integration

After training the GAN, point ADKGD at the checkpoint:

```bash
python experiments/run_experiment.py \
    --dataset dummy_kg --anomaly_ratio 0.05 --max_epoch 1 \
    --neg_source gan \
    --gan_path experiments/kgsage/outputs/checkpoints/kgsage_dummy.pt
```

ADKGD's `Reader._gan_negatives` imports `kgsage_bridge.bridge`, which calls
`kgsage.inference.generate_negatives(...)` — running the loaded generator in
`torch.no_grad()` mode per training batch to produce one negative per positive.

## Datasets

The loader reads any `data/<NAME>/{train,valid,test}.txt` (tab-separated triples).
`KNOWN_DATASETS` in [data/datasets.py](data/datasets.py) maps short names
(`fb15k237`, `wn18rr`, `nell995`, `kinship`, `yago`, `kg20c`, `dummy_kg`) to their
directories; any other directory can be passed by path. Training hyperparameters
(`--dim`, `--epochs`, `--batch_size`, `--lr`, ...) are CLI flags on the trainer.

## What gets saved

| Path | Content |
|---|---|
| `experiments/kgsage/outputs/checkpoints/kgsage_<dataset>.pt` | Trained KGSAGEGenerator weights + vocab maps + real-triple set. |

## Going standalone someday

`kgsage/` imports nothing from outside the `kgsage.*` namespace. To release as a
pip package: `git mv experiments/kgsage ./kgsage`, add a `pyproject.toml`, move
tests to `tests/`, and drop the sys.path shim in the CLI wrappers. The ADKGD
bridge in `experiments/kgsage_bridge/` stays in this thesis codebase — it's
application glue, not library code.
