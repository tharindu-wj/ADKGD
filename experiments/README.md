# experiments/ — research pipeline

Orchestration layer that runs ADKGD with two negative-sample sources and
compares them: random corruption (baseline **B0**) versus kggan-generated
negatives (variant **B1**, called in-process).

Pipeline at a glance:

```
  Step 1: train kggan          (one-time per dataset)  →  .pt checkpoint
  Step 2: run ADKGD              repeat per experiment  →  RESULTS table
            ├─ B0: --neg_source random   (baseline)
            └─ B1: --neg_source gan      (consumes the checkpoint from step 1)
  Step 3: compare the two RESULTS tables
```

Three codebases coexist in this repo:

| Codebase | Location | Status |
|---|---|---|
| **ADKGD** (anomaly detector) | repo root — `Our_TopK%_RankingList.py`, `model.py`, `dataset.py`, `create_batch.py`, `score.py` | upstream — untouched except for the `--neg_source gan` dispatch in `dataset.py` |
| **kggan** (negative generator) | `experiments/gan/src/`, `experiments/gan/scripts/` | upstream — untouched |
| **Orchestration glue** | `experiments/run_experiment.py`, `experiments/slurm/`, `experiments/gan/adkgd_bridge.py` | ours |

---

## Folder map

```
experiments/
├── README.md                              ← you are here
├── RUNNING_ON_DEEPTHOUGHT.md              ← HPC operator guide
├── run_experiment.py                      ← ADKGD orchestrator (train+test+RESULTS)
│
├── slurm/                                 ← ALL HPC launchers
│   ├── train_gan_fb15k.slurm              ← step 1: train kggan
│   ├── run_baseline_fb15k.slurm           ← step 2a: B0 baseline
│   └── run_gan_fb15k.slurm                ← step 2b: B1 with kggan
│
└── gan/
    ├── adkgd_bridge.py                    ← OUR boundary file (kggan ↔ ADKGD adapter)
    ├── src/                               ← kggan source — untouched
    │   ├── corruption_strategies/
    │   ├── dataset_builders/
    │   ├── kg_data/                       (KnowledgeGraph loader)
    │   ├── models/                        (TripleGenerator + embeddings)
    │   ├── sampling/                      (masked-decode helpers)
    │   ├── training/                      (train loop, TrainConfig)
    │   └── validation/
    ├── scripts/                           ← kggan operator CLIs
    │   ├── train_gan.py                   ← used by train_gan_fb15k.slurm
    │   └── build_pseudo_types.py          ← prerequisite for FB15K kggan training
    └── outputs/checkpoints/               ← kggan checkpoint drop zone
        └── dummy.pt                       ← bundled fixture (dummy_kg, 30 epochs)
```

---

## Step 1 — Setup

**Local** (Windows/macOS/Linux):

```powershell
pip install torch numpy scikit-learn matplotlib
```

On Windows local CPU, PyTorch segfaults under multi-threaded OpenMP/MKL.
`experiments/run_experiment.py` sets `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
KMP_DUPLICATE_LIB_OK=TRUE` automatically; no action needed.

**HPC** (Flinders DeepThought, Tesla V100): see [RUNNING_ON_DEEPTHOUGHT.md](RUNNING_ON_DEEPTHOUGHT.md)
for one-time conda env creation with the CUDA wheel.

All commands below run from the **repo root**.

---

## Step 2 — Train kggan (one-time per dataset)

The kggan generator is trained once per dataset; the resulting checkpoint
feeds every subsequent ADKGD-with-GAN run. **Skip this step entirely** if
you already have a checkpoint at the expected path (e.g.
`experiments/gan/outputs/checkpoints/dummy.pt` ships bundled).

### 2a. Build entity pseudo-types (FB15K only — prerequisite)

Fast — login node, no GPU:

```bash
python experiments/gan/scripts/build_pseudo_types.py --data data/FB15K
```

Writes `data/FB15K/entity_metadata.txt`, used by kggan for type-coherent
corruption operators. `dummy_kg` already ships with metadata.

### 2b. Train the generator

Local (dummy KG, CPU, ~minutes):

```powershell
python experiments/gan/scripts/train_gan.py `
    --data data/dummy_kg `
    --epochs 30 `
    --batch-size 32 `
    --device cpu `
    --out experiments/gan/outputs/checkpoints/dummy.pt
```

HPC (FB15K-237, V100):

```bash
sbatch experiments/slurm/train_gan_fb15k.slurm
# → experiments/gan/outputs/checkpoints/fb15k.pt
```

Override hyperparameters via env vars:

```bash
EPOCHS=600 BATCH_SIZE=512 sbatch experiments/slurm/train_gan_fb15k.slurm
DATASET_DIR=data/other_kg \
    CKPT_PATH=experiments/gan/outputs/checkpoints/other.pt \
    sbatch experiments/slurm/train_gan_fb15k.slurm
```

---

## Step 3 — Run ADKGD

Same orchestrator, same RESULTS table for both variants. The **only**
difference is `--neg_source`.

### 3a. Baseline (B0) — random negatives

Local:

```powershell
python experiments/run_experiment.py --dataset dummy_kg --anomaly_ratio 0.15 --max_epoch 1
```

HPC:

```bash
sbatch experiments/slurm/run_baseline_fb15k.slurm
```

### 3b. Variant (B1) — kggan negatives (in-process)

Local (uses the bundled dummy checkpoint):

```powershell
python experiments/run_experiment.py --dataset dummy_kg --anomaly_ratio 0.15 --max_epoch 1 `
    --neg_source gan `
    --gan_path experiments/gan/outputs/checkpoints/dummy.pt
```

HPC (after step 2b produced `fb15k.pt`):

```bash
sbatch experiments/slurm/run_gan_fb15k.slurm
```

### What B1 prints (diagnostic)

```
[GAN] loaded checkpoint from experiments/gan/outputs/checkpoints/dummy.pt (device=cpu)
[GAN] processed=1,242  retries=374  uniform_fallbacks=0
       slot_distribution: head=399/1242(32.1%) rel=395/1242(31.8%) tail=448/1242(36.1%)
```

| Field | Meaning |
|---|---|
| `processed` | Every entry in `bp_triples` got a kggan negative — real positives AND injected eval anomalies, treated uniformly |
| `retries` | kggan's masked decode hit a real-graph collision and was re-rolled with fresh Gumbel noise |
| `uniform_fallbacks` | Retries exhausted → fell back to uniform-random replacement for that one slot |
| `slot_distribution` | Slot pick is uniform random per-positive (≈ 1/3 each, matches baseline) |

A healthy run has `uniform_fallbacks` near zero and slot distribution close to uniform.

---

## Step 4 — Compare results

Each `run_experiment.py` invocation prints a 5-row RESULTS table. Drop B0
and B1 side by side:

| K | B0 (random) | B1 (kggan) | Δ |
|---|---|---|---|
| 1% | 0.9581 (paper 0.951) | _from B1 .out.txt_ | _to fill_ |
| 2% | 0.8836 | _from B1 .out.txt_ | _to fill_ |
| 3% | 0.7852 | _from B1 .out.txt_ | _to fill_ |
| 4% | 0.6929 | _from B1 .out.txt_ | _to fill_ |
| 5% | 0.6148 | _from B1 .out.txt_ | _to fill_ |

Numbers above are from a single seed=0 run on V100; multi-seed mean±std
populates the final research-paper table.

---

## Architecture

```
Step 1 — kggan training (one-time)
  experiments/slurm/train_gan_fb15k.slurm
        └─ experiments/gan/scripts/train_gan.py
                └─ experiments/gan/src/training/train_triple_gan.py
                        └─ writes experiments/gan/outputs/checkpoints/<name>.pt

Step 2 — ADKGD run (per experiment, B0 or B1)
  experiments/slurm/run_{baseline,gan}_fb15k.slurm
        └─ experiments/run_experiment.py            ← orchestrator (ours)
                ├─ subprocess: Our_TopK%_RankingList.py --mode train   (ADKGD upstream)
                │       └─ dataset.py:Reader.get_data()
                │               ├─ neg_source=random → generate_anomalous_triples()
                │               └─ neg_source=gan    → Reader._gan_negatives()
                │                       └─ experiments/gan/adkgd_bridge.py
                │                               ├─ adds src/ to sys.path
                │                               ├─ load_checkpoint() the .pt
                │                               └─ batched generator forward pass
                ├─ subprocess: Our_TopK%_RankingList.py --mode test    (ADKGD upstream)
                └─ parses logs → prints RESULTS table
```

### Why this boundary

- **kggan is invoked only via `experiments/gan/adkgd_bridge.py`**. ADKGD's `dataset.py` knows nothing about kggan's internals — it calls `generate_negatives(...)` and gets ADKGD-ID negatives back. The bridge handles `sys.path` setup, KG construction, vocab string round-trips.
- **B0 and B1 share the orchestrator, RESULTS parser, and comparison table** — any metric delta is unambiguously attributable to the negative source.
- **kggan and ADKGD evolve independently**. Retrain kggan without touching ADKGD; change ADKGD without touching kggan.

---

## Datasets

| Dataset | Files | Triples | Purpose |
|---|---|---|---|
| `dummy_kg` | `data/dummy_kg/{train,valid,test}.txt` | **18 unique × 60 = 1,080** | Smoke-test fixture (6 people, 3 relations, 4 countries). Replication forces `K=0.1%` math to produce ≥ 1. |
| `FB15K` | `data/FB15K/{train,valid,test}.txt` | 310,116 | Paper benchmark (FB15K-237). Real research runs. |

---

## Common operator commands

```bash
# Tail a live HPC log (output and stderr merged — one file)
tail -f kggan_train_fb15k-<jobid>.out.txt           # kggan training
tail -f adkgd_fb15k-<jobid>.out.txt                 # ADKGD baseline (B0)
tail -f adkgd_gan_fb15k-<jobid>.out.txt             # ADKGD with kggan (B1)

# Pull Precision/Recall numbers from ADKGD's detailed log
grep -E "Precision 0\.050000 -- 0\.0[12345]0000|Recall  0\.050000-- 0\.0[12345]0000" \
    checkpoints/FB15K/ADKGD_FB15K_0.05_Neighbors39__log.txt

# Inspect epoch durations
cat checkpoints/FB15K/ADKGD_FB15K_epoch_times.txt

# Verify the slurm submitter's account (DeepThought needs --account=cse)
sacct -j <jobid> --format=JobID,JobName,Account,Partition,State

# Cancel a job
scancel <jobid>
```

---

## Local smoke test

A constructor-only test that confirms the `--neg_source=gan` wiring without
running training:

```powershell
& "$env:USERPROFILE\miniconda3\envs\pytorch\python.exe" temp/smoke_gan.py
```

Expected: `[GAN] loaded checkpoint ...` and `[GAN] processed=1,242` with
slot distribution near 1/3 each.

---

## See also

- [RUNNING_ON_DEEPTHOUGHT.md](RUNNING_ON_DEEPTHOUGHT.md) — HPC setup, env creation, troubleshooting.
- ADKGD upstream — repo root: `Our_TopK%_RankingList.py` (entry), `dataset.py` (Reader + `_gan_negatives` dispatch), `model.py` (BiLSTM_Attention).
- kggan upstream — `experiments/gan/src/training/train_triple_gan.py`, `experiments/gan/src/models/triple_gan.py`.
- Bridge — [experiments/gan/adkgd_bridge.py](gan/adkgd_bridge.py) (the only file that knows about both worlds).
