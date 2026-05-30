# experiments/ — research pipeline

Orchestration layer that runs ADKGD with two negative-sample sources and
compares them: random corruption (baseline **B0**) versus the GAN-generated
negatives (variant **B1**, called in-process).

Pipeline at a glance:

```
  Step 1: train the GAN          (one-time per dataset)  →  .pt checkpoint
  Step 2: run ADKGD              repeat per experiment  →  RESULTS table
            ├─ B0: --neg_source random   (baseline)
            └─ B1: --neg_source gan      (consumes the checkpoint from step 1)
  Step 3: compare the two RESULTS tables
```

Three codebases coexist in this repo:

| Codebase | Location | Status |
|---|---|---|
| **ADKGD** (anomaly detector) | repo root — `Our_TopK%_RankingList.py`, `model.py`, `dataset.py`, `create_batch.py`, `score.py` | upstream — untouched except for the `--neg_source gan` dispatch in `dataset.py` |
| **Simple GAN** (negative generator) | `experiments/gan/` — teaching-grade rewrite (~600 lines) | ours |
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
│   ├── train_gan_fb15k.slurm              ← step 1: train the GAN
│   ├── run_baseline_fb15k.slurm           ← step 2a: B0 baseline
│   └── run_gan_fb15k.slurm                ← step 2b: B1 with the GAN
│
└── gan/                                   ← simple GAN (teaching version)
    ├── README.md                          ← per-codebase quickstart + diagram
    ├── data.py                            ← KG loader (~70 lines)
    ├── model.py                           ← Generator + Discriminator (plain MLPs, ~110 lines)
    ├── train.py                           ← training CLI used by train_gan_fb15k.slurm
    ├── generate.py                        ← in-process negative generation
    ├── adkgd_bridge.py                    ← OUR boundary file (GAN ↔ ADKGD adapter)
    └── outputs/checkpoints/               ← .pt drop zone
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

## Step 2 — Train the GAN (one-time per dataset)

The GAN is trained once per dataset; the resulting checkpoint feeds every
subsequent ADKGD-with-GAN run. **Skip this step entirely** if you already
have a checkpoint at the expected path (e.g.
`experiments/gan/outputs/checkpoints/dummy.pt` ships bundled).

Local (dummy KG, CPU, ~minutes):

```powershell
python experiments/gan/train.py `
    --data data/dummy_kg `
    --epochs 30 `
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
EPOCHS=200 BATCH_SIZE=256 sbatch experiments/slurm/train_gan_fb15k.slurm
DATASET_DIR=data/other_kg \
    CKPT_PATH=experiments/gan/outputs/checkpoints/other.pt \
    sbatch experiments/slurm/train_gan_fb15k.slurm
```

See [gan/README.md](gan/README.md) for the GAN architecture diagram, loss
functions, and the per-file walkthrough.

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

### 3b. Variant (B1) — the GAN negatives (in-process)

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
| `processed` | Every entry in `bp_triples` got a the GAN negative — real positives AND injected eval anomalies, treated uniformly |
| `retries` | the GAN's masked decode hit a real-graph collision and was re-rolled with fresh Gumbel noise |
| `uniform_fallbacks` | Retries exhausted → fell back to uniform-random replacement for that one slot |
| `slot_distribution` | Slot pick is uniform random per-positive (≈ 1/3 each, matches baseline) |

A healthy run has `uniform_fallbacks` near zero and slot distribution close to uniform.

---

## Step 4 — Compare results

Each `run_experiment.py` invocation prints a 5-row RESULTS table. Drop B0
and B1 side by side:

| K | B0 (random) | B1 (the GAN) | Δ |
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
Step 1 — GAN training (one-time)
  experiments/slurm/train_gan_fb15k.slurm
        └─ experiments/gan/train.py
                ├─ data.py     (load KG)
                ├─ model.py    (Generator + Discriminator)
                └─ writes experiments/gan/outputs/checkpoints/<name>.pt

Step 2 — ADKGD run (per experiment, B0 or B1)
  experiments/slurm/run_{baseline,gan}_fb15k.slurm
        └─ experiments/run_experiment.py            ← orchestrator (ours)
                ├─ subprocess: Our_TopK%_RankingList.py --mode train   (ADKGD upstream)
                │       └─ dataset.py:Reader.get_data()
                │               ├─ neg_source=random → generate_anomalous_triples()
                │               └─ neg_source=gan    → Reader._gan_negatives()
                │                       └─ experiments/gan/adkgd_bridge.py
                │                               ├─ generate.py:load_checkpoint() the .pt
                │                               └─ generate.py:generate_negatives() — batched forward pass
                ├─ subprocess: Our_TopK%_RankingList.py --mode test    (ADKGD upstream)
                └─ parses logs → prints RESULTS table
```

### Why this boundary

- **The GAN is invoked only via `experiments/gan/adkgd_bridge.py`**. ADKGD's `dataset.py` knows nothing about the GAN's internals — it calls `generate(...)` and gets ADKGD-ID negatives back. The bridge handles `sys.path` setup and vocab string round-trips.
- **B0 and B1 share the orchestrator, RESULTS parser, and comparison table** — any metric delta is unambiguously attributable to the negative source.
- **The GAN and ADKGD evolve independently**. Retrain the GAN without touching ADKGD; change ADKGD without touching the GAN.

---

## Datasets

| Dataset | Files | Triples | Purpose |
|---|---|---|---|
| `dummy_kg` | `data/dummy_kg/{train,valid,test}.txt` | **18 unique × 60 = 1,080** | Smoke-test fixture (6 people, 3 relations, 4 countries). Replication forces `K=0.1%` math to produce ≥ 1. |
| `FB15K` | `data/FB15K/{train,valid,test}.txt` | 310,116 | Paper benchmark (FB15K-237). Real research runs. |

---

## Where the HPC logs live

Every slurm script in this repo merges stdout and stderr into a single file
with the header `#SBATCH --output=%x-%j.out.txt`, where `%x` is the job name
and `%j` is the slurm job id.

**The file lands in whichever directory you ran `sbatch` from** (usually
`~/ADKGD`). It is NOT in `checkpoints/` and NOT in `experiments/`.

### Naming convention per slurm

| Slurm script | `--job-name` | Log filename |
|---|---|---|
| `train_gan_fb15k.slurm` | `gan_train_fb15k` | `gan_train_fb15k-<jobid>.out.txt` |
| `run_baseline_fb15k.slurm` | `adkgd_fb15k` | `adkgd_fb15k-<jobid>.out.txt` |
| `run_gan_fb15k.slurm` | `adkgd_gan_fb15k` | `adkgd_gan_fb15k-<jobid>.out.txt` |

### Three useful commands

```bash
# 1. Tail the latest log for a given slurm WITHOUT typing the job id
cd ~/ADKGD
tail -f "$(ls -t gan_train_fb15k-*.out.txt | head -1)"

# 2. Tail a specific job id (you get this from `sbatch` or `squeue`)
tail -f gan_train_fb15k-2886370.out.txt

# 3. List the latest few logs across all slurms
ls -t *-*.out.txt | head -10
```

### What's in the log vs in checkpoints/

| Where | What it contains |
|---|---|
| `~/ADKGD/<jobname>-<jobid>.out.txt` | **Slurm-level**: GPU pre-flight, env activation, all `print()`/`echo` output, Python tracebacks, the RESULTS table at the end |
| `~/ADKGD/checkpoints/<dataset>/ADKGD_<dataset>_<ratio>_Neighbors39__log.txt` | **ADKGD-internal**: every Precision/Recall line per K cutoff, per-batch losses (this is what `run_experiment.py` greps to build the RESULTS table) |
| `~/ADKGD/checkpoints/<dataset>/ADKGD_<dataset>_epoch_times.txt` | Per-epoch training duration in seconds |

## Other common operator commands

```bash
# Job state (works even after the job finishes — squeue only shows running jobs)
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,Reason

# All your recent job states today
sacct -u $USER --starttime=today --format=JobID,JobName,State,ExitCode,Elapsed

# Pull Precision/Recall numbers from ADKGD's detailed log
grep -E "Precision 0\.050000 -- 0\.0[12345]0000|Recall  0\.050000-- 0\.0[12345]0000" \
    checkpoints/FB15K/ADKGD_FB15K_0.05_Neighbors39__log.txt

# Inspect epoch durations
cat checkpoints/FB15K/ADKGD_FB15K_epoch_times.txt

# Cancel a running or pending job
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
- Simple GAN — [experiments/gan/](gan/) (data.py, model.py, train.py, generate.py).
- Bridge — [experiments/gan/adkgd_bridge.py](gan/adkgd_bridge.py) (the only file that knows about both worlds).
