# experiments/ — KGSAGE research pipeline

**KGSAGE** (Knowledge Graph Semantic Anomaly GEnerator) — adversarial negative generator that replaces ADKGD's random negative sampling with type-coherent semantically-plausible anomalies (Category 5). Implements the CGSP framework (Concept-Guided Generative Sampling Paradigm, Tong et al. 2026, DAMI) inside ADKGD's experiment harness.

Canonical design reference: **[PIPELINE.md](PIPELINE.md)** — locked decisions, output formats, validation gates per phase.

## Pipeline at a glance

Three-phase pipeline, each phase a folder under `experiments/gan/`:

```
   data/FB15K-237/train.txt
              │
              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  PHASE 1 - concept/    schema + pools + cardinality             │
   │   adapters/freebase.py + concept_pools.py + cardinality.py      │
   │   -> data/<DATASET>/entity_types.tsv                            │
   │   -> data/<DATASET>/entity_types_metadata.json                  │
   │   -> experiments/gan/outputs/concept_pools/<DATASET>.pkl        │
   └─────────────────────────────────────────────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  PHASE 2 - adversarial/    train KGSAGE (REINFORCE)             │
   │   discriminator.py (TransE D) + generator.py (MLP G)            │
   │   + candidate_pool.py + train.py                                │
   │   -> experiments/gan/outputs/checkpoints/<DATASET>_kgsage.pt      │
   │   -> experiments/gan/outputs/logs/<DATASET>_kgsage_training.json  │
   └─────────────────────────────────────────────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  PHASE 3 - corruption/    produce anomalies (NOT YET BUILT)     │
   │   api.py (KGCorrupter primitive) + infer.py + adkgd_bridge.py   │
   │   -> in-memory negatives per corrupt(triple) call               │
   └─────────────────────────────────────────────────────────────────┘
              │
              ▼
   ADKGD consumes negatives via corruption/adkgd_bridge.py
   (replaces legacy gan/adkgd_bridge.py once Phase 3 lands)
```

## Scope

| Aspect | Decision |
|---|---|
| Method | KGSAGE (CGSP framework) — REINFORCE + concept + cardinality |
| First dataset | FB15K-237 (in-place rewrite, incremental phases) |
| Future datasets | WN18RR, YAGO 4.5 (after FB validates end-to-end) |
| Anomaly focus | Category 5 (type-coherent, semantically wrong) |
| Bulk injector | Parked - defer to future work |
| Out of scope | Kinship, KG20C (removed); universal/zero-shot corrupter |

## Folder map

```
experiments/
├── README.md                          ← you are here
├── PIPELINE.md                        ← locked design reference
├── RUNNING_ON_DEEPTHOUGHT.md          ← HPC operator guide
├── run_experiment.py                  ← ADKGD orchestrator (train+test+RESULTS)
│
├── slurm/                             ← HPC launchers
│   ├── train_kgsage_fb15k237.slurm         ← NEW - train KGSAGE (Phase 1+2)
│   ├── run_baseline_fb15k237.slurm        ← B0 baseline (random negs)
│   ├── run_baseline_with_gan_fb15k237.slurm  ← will become KGSAGE B2 after Phase 3
│   ├── train_gan_fb15k237.slurm           ← LEGACY (Gumbel GAN), deprecated
│   ├── train_gan_wn18rr.slurm             ← LEGACY (WN18RR not yet migrated)
│   ├── run_baseline_wn18rr.slurm          ← B0 baseline for WN18RR
│   └── run_baseline_with_gan_wn18rr.slurm ← WN18RR variant
│
└── gan/
    ├── concept/                       ← Phase 1 - Concept Module
    │   ├── adapters/
    │   │   ├── base.py                ←   BaseKBAdapter ABC
    │   │   └── freebase.py            ←   FreebaseAdapter (covers FB15K family)
    │   ├── concept_pools.py           ← builds headPool, tailPool, vocab
    │   ├── cardinality.py             ← 1-1 / 1-N / N-1 / N-N classifier
    │   └── preprocess.py              ← Phase 1 CLI orchestrator
    │
    ├── adversarial/                   ← Phase 2 - Adversarial Module
    │   ├── discriminator.py           ← TransE D (shares E + R with G)
    │   ├── generator.py               ← CandidateScorer MLP (G)
    │   ├── candidate_pool.py          ← per-positive candidate builder
    │   └── train.py                   ← REINFORCE training CLI
    │
    ├── corruption/                    ← Phase 3 - NOT YET BUILT
    │
    ├── outputs/
    │   ├── concept_pools/<DATASET>.pkl       ← Phase 1 cache
    │   ├── checkpoints/<DATASET>_kgsage.pt     ← Phase 2 checkpoint
    │   └── logs/<DATASET>_kgsage_training.json ← Phase 2 training log
    │
    ├── README.md                      ← per-codebase quickstart
    ├── data.py                        ← shared KG loader (legacy + new)
    ├── adkgd_bridge.py                ← LEGACY bridge (Phase 3 supersedes)
    ├── gan_model.py                   ← LEGACY Gumbel G (Phase 2.2 superseded)
    ├── train.py                       ← LEGACY Gumbel training (Phase 2.3 superseded)
    └── corrupt_triples.py             ← LEGACY Gumbel inference (Phase 3.1 will supersede)
```

## Setup

**Local** (Windows/macOS/Linux):

```powershell
pip install torch numpy scikit-learn matplotlib
```

On Windows local CPU, PyTorch segfaults under multi-threaded OpenMP/MKL.
`experiments/run_experiment.py` and `adversarial/train.py` both set
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` automatically.

**HPC** (Flinders DeepThought, Tesla V100): see [RUNNING_ON_DEEPTHOUGHT.md](RUNNING_ON_DEEPTHOUGHT.md)
for one-time conda env creation with the CUDA wheel.

All commands below run from the **repo root**.

---

## Phase 1 - Build concept pools

Cheap (~1 minute on FB15K-237). Run on the login node before submitting any GPU job.

```bash
python -m experiments.gan.concept.preprocess --dataset FB15K-237 --family freebase
```

Produces:
- `data/FB15K-237/entity_types.tsv`         (NTriples-style, YAGO convention)
- `data/FB15K-237/entity_types_metadata.json` (provenance + statistics)
- `experiments/gan/outputs/concept_pools/FB15K-237.pkl` (Phase 2/3 cache)

Successful output ends with a summary block showing entity coverage, median pool sizes, and cardinality distribution.

---

## Phase 2 - Train KGSAGE

### Local smoke test (CPU, small fraction)

For correctness verification only:

```powershell
python -m experiments.gan.adversarial.train `
    --dataset FB15K-237 `
    --warmup_epochs 2 `
    --total_epochs 5 `
    --batch_size 128
```

### HPC full training (FB15K-237, V100, ~2 hours)

The SLURM script runs Phase 1 (preprocess) AND Phase 2 (REINFORCE training) end-to-end:

```bash
sbatch experiments/slurm/train_kgsage_fb15k237.slurm
# -> experiments/gan/outputs/checkpoints/FB15K-237_kgsage.pt
# -> experiments/gan/outputs/logs/FB15K-237_kgsage_training.json
```

Override hyperparameters via env vars (no script edits required):

```bash
TOTAL_EPOCHS=200 sbatch experiments/slurm/train_kgsage_fb15k237.slurm
BATCH_SIZE=256 EMBEDDING_DIM=200 sbatch experiments/slurm/train_kgsage_fb15k237.slurm
```

### Hyperparameter defaults (from PIPELINE.md)

| Knob | Default | Env var |
|---|---|---|
| Warmup epochs (D-only) | 5 | `WARMUP_EPOCHS` |
| Total epochs | 100 | `TOTAL_EPOCHS` |
| Batch size | 128 | `BATCH_SIZE` |
| Embedding dim | 100 | `EMBEDDING_DIM` |
| Generator MLP hidden dim | 256 | `HIDDEN_DIM` |
| Candidate pool size (N_S) | 64 | `N_S` |
| G learning rate | 1e-4 | `G_LR` |
| D learning rate | 1e-3 | `D_LR` |
| Random seed | 0 | `SEED` |

### Convergence checks

After the SLURM job completes, inspect the training log:

```python
import json
with open("experiments/gan/outputs/logs/FB15K-237_kgsage_training.json") as f:
    log = json.load(f)
for e in log["epochs"]:
    print(e["epoch"], e["phase"], e["d_loss"], e["g_loss"], e["baseline"])
```

Expected pattern: D and G losses trend down; baseline_ema rises slowly and stays finite; mean_reward stabilises.

---

## Phase 3 - Corruption

**Not yet built.** Will provide the `KGCorrupter.corrupt(triple) -> triple` primitive that ADKGD consumes for runtime negative sampling. Tracking todo: `Phase 3: corruption/ - infer.py + api.py + adkgd_bridge.py`.

Once landed, this section will document:
- Running ADKGD with KGSAGE negatives (`--neg_source gan`)
- The new `experiments/gan/corruption/adkgd_bridge.py` (replaces legacy bridge)
- B0 vs B2 comparison procedure

---

## Run ADKGD baseline (B0)

The B0 baseline (random negatives) still works through the existing orchestrator and SLURM scripts. It's the reference point KGSAGE gets compared against.

Local:

```powershell
python experiments/run_experiment.py --dataset FB15K-237 --anomaly_ratio 0.05 --max_epoch 1
```

HPC:

```bash
sbatch experiments/slurm/run_baseline_fb15k237.slurm
# -> checkpoints/FB15K-237/ADKGD_FB15K-237_0.05_Neighbors39__log.txt
```

---

## Compare results (planned)

Once Phase 3 lands, the comparison table populated by repeated `run_experiment.py` invocations:

| K | B0 (random) | B2 (KGSAGE) | Δ |
|---|---|---|---|
| 1% | 0.9581 (paper 0.951) | _from B2 .out.txt_ | _to fill_ |
| 2% | 0.8836 | _to fill_ | _to fill_ |
| 3% | 0.7852 | _to fill_ | _to fill_ |
| 4% | 0.6929 | _to fill_ | _to fill_ |
| 5% | 0.6148 | _to fill_ | _to fill_ |

(Numbers above are from a single seed=0 V100 run; multi-seed mean±std for the paper table.)

---

## Datasets

| Dataset | Files | Triples | Status |
|---|---|---|---|
| `dummy_kg` | `data/dummy_kg/{train,valid,test}.txt` | 1,080 | Smoke-test fixture (6 people, 3 relations, 4 countries). |
| `FB15K-237` | `data/FB15K-237/{train,valid,test}.txt` | 310,116 | Primary research dataset. Covered by `FreebaseAdapter`. |
| `WN18RR` | `data/WN18RR/{train,valid,test}.txt` | 93,003 | Future - `WordNetAdapter` not yet built. |
| `YAGO 4.5` | `data/YAGO4.5/` (TBD) | TBD | Future - acquisition + `YagoAdapter` not yet built. |

Dropped from scope: Kinship, KG20C (single-type entities; no useful concept structure for KGSAGE).

---

## Where the HPC logs live

Every SLURM script merges stdout and stderr into a single file via `#SBATCH --output=%x-%j.out.txt`, where `%x` is the job name and `%j` is the SLURM job id. The file lands in whichever directory you ran `sbatch` from (usually `~/ADKGD`).

| SLURM script | `--job-name` | Log filename pattern |
|---|---|---|
| `train_kgsage_fb15k237.slurm` | `kgsage_train_fb15k237` | `kgsage_train_fb15k237-<jobid>.out.txt` |
| `run_baseline_fb15k237.slurm` | `adkgd_fb15k237` | `adkgd_fb15k237-<jobid>.out.txt` |
| `run_baseline_with_gan_fb15k237.slurm` | `adkgd_baseline_with_gan_fb15k237` | `adkgd_baseline_with_gan_fb15k237-<jobid>.out.txt` |
| `train_gan_fb15k237.slurm` (legacy) | `gan_train_fb15k237` | `gan_train_fb15k237-<jobid>.out.txt` |
| `run_baseline_wn18rr.slurm` | `adkgd_wn18rr` | `adkgd_wn18rr-<jobid>.out.txt` |

### Useful tailing commands

```bash
# Tail the latest KGSAGE training log without typing the job id
cd ~/ADKGD
tail -f "$(ls -t kgsage_train_fb15k237-*.out.txt | head -1)"

# Tail a specific job id
tail -f kgsage_train_fb15k237-2886370.out.txt

# List the latest few logs across all SLURMs
ls -t *-*.out.txt | head -10
```

### What's in the log vs in `experiments/gan/outputs/`

| Where | What it contains |
|---|---|
| `~/ADKGD/<jobname>-<jobid>.out.txt` | SLURM stdout: GPU pre-flight, env activation, all `print()`/`echo` output, Python tracebacks, per-epoch loss lines |
| `experiments/gan/outputs/logs/<DATASET>_kgsage_training.json` | Structured per-epoch loss curves (parseable for plotting) |
| `experiments/gan/outputs/checkpoints/<DATASET>_kgsage.pt` | Trained G + D weights + metadata |
| `checkpoints/<dataset>/ADKGD_<dataset>_...log.txt` | ADKGD-internal (after B0 or B2 run): Precision/Recall per K cutoff |

---

## Operator commands

```bash
# Job state (works even after the job finishes - squeue only shows running jobs)
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,Reason

# All your recent job states today
sacct -u $USER --starttime=today --format=JobID,JobName,State,ExitCode,Elapsed

# Pull Precision/Recall numbers from ADKGD's detailed log
grep -E "Precision 0\.050000 -- 0\.0[12345]0000|Recall  0\.050000-- 0\.0[12345]0000" \
    checkpoints/FB15K-237/ADKGD_FB15K-237_0.05_Neighbors39__log.txt

# Inspect epoch durations
cat checkpoints/FB15K-237/ADKGD_FB15K-237_epoch_times.txt

# Cancel a running or pending job
scancel <jobid>
```

---

## Migration status from legacy Gumbel-Softmax GAN

| Legacy file (still on disk) | Replacement | Status |
|---|---|---|
| `gan/gan_model.py` | `gan/adversarial/generator.py` | Superseded by Phase 2.2 |
| `gan/train.py` | `gan/adversarial/train.py` | Superseded by Phase 2.3 |
| `gan/corrupt_triples.py` | `gan/corruption/infer.py` | To be replaced in Phase 3.1 |
| `gan/adkgd_bridge.py` | `gan/corruption/adkgd_bridge.py` | To be moved in Phase 3.2 |
| `slurm/train_gan_fb15k237.slurm` | `slurm/train_kgsage_fb15k237.slurm` | Superseded; old script left as reference |

Legacy files are deleted at end of Phase 3 once the new pipeline is end-to-end validated.

---

## See also

- [PIPELINE.md](PIPELINE.md) — Locked design reference: scope, file layout, output formats, per-phase validation gates, hyperparameters.
- [RUNNING_ON_DEEPTHOUGHT.md](RUNNING_ON_DEEPTHOUGHT.md) — HPC setup, env creation, troubleshooting.
- ADKGD upstream — repo root: `Our_TopK%_RankingList.py`, `dataset.py`, `model.py`, `create_batch.py`, `score.py`.
- CGSP paper — Tong et al. 2026, DAMI: "A framework fusing entity concepts and GAN negative sampling for improving knowledge reasoning."
