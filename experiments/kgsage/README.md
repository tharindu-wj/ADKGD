# KGSAGE — Knowledge Graph Semantic Anomaly Generator

A standalone-ready Python package for generating role-swap contradiction
anomalies in knowledge graphs. Designed to extend per-triple anomaly detectors
(like ADKGD) to multi-triple anomaly categories — specifically TAXO category
**#5 Contradictions, role-swap sub-class**.

Master's thesis pipeline ([THESIS_PLAN](../docs/THESIS_PLAN_pairgan_contradictions.md)).

## What this package does in one paragraph

KGSAGE has two phases:

- **Phase 1 — Encoder pretraining**. An RGCN backbone + DistMult decoder
  learn entity and relation embeddings that capture anti-symmetric predicate
  structure. Implemented in `kgsage.encoder`.
- **Phase 2 — Adversarial generation**. The pair-aware `KGSAGEGenerator` +
  `KGSAGEDiscriminator`, conditioned on the Phase 1 encoder embeddings, learn
  to emit role-swap contradiction partners `(t, r', h)` that ADKGD trains
  against. Lives in `kgsage.gan`.

ADKGD integration (Phase 4 — using KGSAGE-generated negatives in the ADKGD
detector) lives in the sibling folder `experiments/kgsage_bridge/`, not in
this package — keeping `kgsage/` ADKGD-agnostic and standalone-extractable.

## Package layout

```
kgsage/
├── __init__.py             <- public API (load_kg, KGSAGE*, generate_partners, ...)
├── README.md               <- this file
│
├── data/                   <- KG loading + dataset registry + dataset-level audit
│   ├── loaders.py          <- load_kg(path)
│   ├── datasets.py         <- KNOWN_DATASETS + resolve_dataset()
│   └── audit_dataset.py    <- Test 1.3 (dataset anti-symmetric pair density)
│
├── encoder/                <- Phase 1: RGCN + DistMult
│   ├── models.py           <- KGSAGEEncoder, KGSAGEDistMultDecoder, KGSAGELinkPredictor
│   ├── train.py            <- Phase 1 training loop
│   ├── evaluate.py         <- Test 1.1 (link prediction MRR)
│   └── audit_embeddings.py <- Test 1.2 (anti-symmetric signal in trained embeddings)
│
├── gan/                    <- Phase 2: the pair-aware role-swap GAN
│   ├── models.py           <- KGSAGEGenerator + KGSAGEDiscriminator + helpers
│   ├── train.py            <- adversarial training loop
│   └── partner_templates.py<- mine_partner_templates (discriminator supervision)
│
├── inference.py            <- public generation API:
│                              load_kgsage_checkpoint / generate_partners /
│                              render_partner_stats
│
├── cli/                    <- command-line entry points (thin wrappers)
│   ├── audit_dataset.py
│   ├── train_encoder.py
│   ├── evaluate_encoder.py
│   ├── audit_embeddings.py
│   └── train_gan.py
│
└── slurm/                  <- HPC job launchers
    ├── README.md
    ├── train_encoder_fb15k237.slurm
    └── train_gan_fb15k237.slurm
```

## Quick start

```bash
# 0. Install dependencies (one-time)
pip install torch torch_geometric

# 1. Make `import kgsage` work without pip-installing
export PYTHONPATH="$(pwd)/experiments:$PYTHONPATH"

# 2. Test 1.3 — does the dataset have the anti-symmetric signal we need?
python -m kgsage.cli.audit_dataset --dataset fb15k237

# 3. Train the encoder (Phase 1; ~30 min on a V100)
python -m kgsage.cli.train_encoder --dataset fb15k237

# 4. Test 1.1 — link prediction MRR
python -m kgsage.cli.evaluate_encoder --dataset fb15k237

# 5. Test 1.2 — anti-symmetric signal in relation embeddings
python -m kgsage.cli.audit_embeddings --dataset fb15k237

# 6. Train the GAN (Phase 2) — add --encoder_ckpt <fb15k237_encoder.pt> for the real run
python -m kgsage.cli.train_gan \
    --data data/dummy_kg \
    --epochs 30 \
    --device cpu \
    --out experiments/kgsage/outputs/checkpoints/kgsage_dummy.pt
```

## ADKGD integration

After training the GAN, point ADKGD at the checkpoint:

```bash
python experiments/run_experiment.py \
    --dataset dummy_kg --anomaly_ratio 0.15 --max_epoch 1 \
    --neg_source gan \
    --gan_path experiments/kgsage/outputs/checkpoints/kgsage_dummy.pt
```

ADKGD's `Reader._gan_negatives` imports `kgsage_bridge.bridge`, which calls
`kgsage.inference.generate_kgsage_partners(...)`, running the loaded generator
in `torch.no_grad()` mode per training batch to emit one role-swap
contradiction `(t, r', h)` per positive.

## Dataset extension

To use KGSAGE with a new dataset, add an entry to `KNOWN_DATASETS` in
[data/datasets.py](data/datasets.py):

```python
KNOWN_DATASETS["mydataset"] = {
    "default_path": "data/MyDataset",
    "n_relations": 50,
    "epochs": 200,
    "dim": 200,
    "n_layers": 2,
    "num_bases": 30,
    "batch_size": 2048,
    "lr": 1e-3,
    "margin": 1.0,
    "eval_every": 10,
    "expected_mrr": 0.30,
    "antisym_min_pairs": 30,
}
```

Then use the short name everywhere:
```bash
python -m kgsage.cli.audit_dataset --dataset mydataset
python -m kgsage.cli.train_encoder --dataset mydataset
```

For one-off datasets that don't need a registry entry, pass a filesystem
path directly:
```bash
python -m kgsage.cli.audit_dataset --dataset /path/to/custom_kg
```

The CLI accepts both short names and paths via the same `--dataset` flag.

## Currently supported datasets

| Short name | Path | Status |
|---|---|---|
| `fb15k237` | `data/FB15K-237` | Test 1.3 PASSED (247 anti-sym pairs) |
| `wn18rr`   | `data/WN18RR`    | Defaults present; not yet audited |
| `nell995`  | `data/NELL-995`  | Defaults present; not yet audited |
| `dummy_kg` | `data/dummy_kg`  | Smoke-test fixture (no decision gates) |

## What gets saved

| Path template | Content |
|---|---|
| `experiments/kgsage/outputs/<dataset>_density_audit.json` | Test 1.3 output. Consumed by Test 1.2. |
| `experiments/kgsage/outputs/<dataset>_encoder.pt`         | Trained encoder + decoder weights, vocab maps. |
| `experiments/kgsage/outputs/checkpoints/kgsage_<dataset>.pt` | Trained GAN: KGSAGEGenerator weights + vocab + real triples. |

## Decision gates

Each test has a pass/fail criterion documented in the thesis plan. If any
test fails, **stop and pivot before architecting Phase 2** — the failure
modes (data too sparse, model can't learn anti-symmetric signal) are
exactly what Phase 1 was designed to detect cheaply.

| Test | Pass | Fail action |
|---|---|---|
| 1.3 (dataset audit)        | ≥ dataset's `antisym_min_pairs` (FB15K-237: 30)  | Try a different dataset or pivot to rule-mining |
| 1.1 (link prediction MRR)  | MRR ≥ dataset's `expected_mrr` (FB15K-237: 0.30) | Debug training |
| 1.2 (anti-symmetric signal)| Mann-Whitney U test p < 0.05 + direction correct | Try ConvE decoder, longer training, or escalate to CompGCN |

## Why RGCN + DistMult (and not CompGCN)?

CompGCN ([Vashishth 2020](https://arxiv.org/abs/1911.03082)) is theoretically
nicer — it jointly embeds nodes and relations. But it's not in PyTorch
Geometric, and porting [malllabiisc/CompGCN](https://github.com/malllabiisc/CompGCN)
from PyTorch 1.0 to modern PyG takes ~2 weeks.

RGCN ([Schlichtkrull 2018](https://arxiv.org/abs/1703.06103)) is in PyG
natively as `torch_geometric.nn.RGCNConv`. Paired with DistMult decoder,
we get both entity embeddings (from the encoder) and relation embeddings
(from `decoder.rel_emb`) — same outputs CompGCN would give. CompGCN remains
Plan B if Test 1.2 fails on RGCN+DistMult.

See [THESIS_PLAN_pairgan_contradictions.md §2.2](../docs/THESIS_PLAN_pairgan_contradictions.md)
for the full encoder-choice rationale.

## Going standalone someday

KGSAGE is structurally a standalone package — nothing inside `kgsage/`
imports from outside the `kgsage.*` namespace. To release as a pip
package eventually:

1. `git mv experiments/kgsage ./kgsage`
2. Add a `pyproject.toml` with the public API exported in `kgsage/__init__.py`
3. Move tests to `tests/`
4. `pip install -e .` and test it works without the sys.path trick in the
   CLI shims

The ADKGD bridge in `experiments/kgsage_bridge/` stays in this thesis
codebase — it's application code, not library code.
