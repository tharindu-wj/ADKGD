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
| **KGSAGE** (GAN package) | `experiments/kgsage/` — conditional GAN + inference | ours |
| **Orchestration glue** | `experiments/run_experiment.py`, `experiments/slurm/`, `experiments/kgsage_bridge/bridge.py` | ours |

---

## Folder map

```
experiments/
├── README.md                              ← you are here
├── RUNNING_ON_DEEPTHOUGHT.md              ← HPC operator guide
├── run_experiment.py                      ← ADKGD orchestrator (train+test+RESULTS)
│
├── slurm/                                 ← ADKGD-side launchers
│   ├── run_baseline_fb15k237.slurm              ← FB15K-237 — ADKGD baseline (random negatives, B0)
│   ├── run_baseline_with_kgsage_fb15k237.slurm  ← FB15K-237 — ADKGD + KGSAGE negatives (B1)
│   └── run_baseline_wn18rr.slurm                ← WN18RR    — ADKGD baseline (B0)
│
├── kgsage/                                ← standalone-ready GAN package
│   ├── README.md                          ← package overview, run order, datasets
│   ├── __init__.py                        ← public API
│   │
│   ├── data/                              ← KG loading + dataset registry
│   │   ├── loaders.py                     ← load_kg(path) -> integer triples + vocab
│   │   └── datasets.py                    ← KNOWN_DATASETS + resolve_dataset()
│   │
│   ├── gan/                               ← the conditional GAN
│   │   ├── models.py                      ← KGSAGEGenerator (3-head) + KGSAGEDiscriminator + helpers
│   │   └── train.py                       ← adversarial training loop (BCE + reconstruction)
│   │
│   ├── inference.py                       ← public generation API (generate_negatives)
│   │
│   ├── cli/                               ← command-line entry points (thin shims)
│   │   └── train_gan.py
│   │
│   ├── slurm/                             ← KGSAGE-only HPC launchers
│   │   ├── README.md
│   │   └── train_gan_fb15k237.slurm
│   │
│   └── outputs/checkpoints/               ← .pt drop zone
│       └── kgsage_dummy.pt                ← bundled fixture (dummy_kg smoke)
│
└── kgsage_bridge/                         ← OUR boundary file (KGSAGE ↔ ADKGD adapter)
    ├── README.md                          ← integration rationale + contract
    ├── __init__.py                        ← re-exports load_gan/generate/render_stats
    └── bridge.py                          ← implementation
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
`experiments/kgsage/outputs/checkpoints/kgsage_dummy.pt` ships bundled).

Local (dummy KG, CPU, ~minutes):

```powershell
$env:PYTHONPATH = "experiments"
python -m kgsage.cli.train_gan `
    --data data/dummy_kg `
    --epochs 30 `
    --device cpu `
    --out experiments/kgsage/outputs/checkpoints/kgsage_dummy.pt
```

HPC (FB15K-237, V100):

```bash
sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm
# → experiments/kgsage/outputs/checkpoints/kgsage_fb15k237.pt
```

Override run knobs via env vars (EPOCHS, SEED, ENCODER_CKPT, CKPT_PATH):

```bash
EPOCHS=200 BATCH_SIZE=256 sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm
DATASET_DIR=data/other_kg \
    CKPT_PATH=experiments/kgsage/outputs/checkpoints/other.pt \
    sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm
```

See [kgsage/README.md](kgsage/README.md) for the package architecture,
training command surface, and dataset extension story.

---

## Step 3 — Run ADKGD

Same orchestrator, same RESULTS table for both variants. The **only**
difference is `--neg_source`.

### 3a. Baseline (B0) — random negatives

Local:

```powershell
python experiments/run_experiment.py --dataset dummy_kg --anomaly_ratio 0.15 --max_epoch 1
```

HPC (FB15K-237):

```bash
sbatch experiments/slurm/run_baseline_fb15k237.slurm
```

HPC (WN18RR):

```bash
sbatch experiments/slurm/run_baseline_wn18rr.slurm
```

### 3b. Variant (B1) — the GAN negatives (in-process)

Local (uses the bundled dummy checkpoint):

```powershell
python experiments/run_experiment.py --dataset dummy_kg --anomaly_ratio 0.15 --max_epoch 1 `
    --neg_source gan `
    --gan_path experiments/kgsage/outputs/checkpoints/kgsage_dummy.pt
```

HPC FB15K-237 (after step 2 produced `kgsage_fb15k237.pt`):

```bash
sbatch experiments/slurm/run_baseline_with_kgsage_fb15k237.slurm
```

(A WN18RR `+KGSAGE` launcher is not wired yet — copy the FB15K-237 one and
point `GAN_CKPT` at a WN18RR checkpoint when needed.)

### What B1 prints (diagnostic)

```
[GAN] loaded checkpoint from experiments/kgsage/outputs/checkpoints/kgsage_fb15k237.pt (device=cuda)
[GAN] processed=325,620  retries=30,587  uniform_fallbacks=2,177  slot_distribution: head=108116/325620(33.2%) rel=108940/325620(33.5%) tail=108564/325620(33.3%)
```

| Field | Meaning |
|---|---|
| `processed` | Every entry in `bp_triples` got a GAN negative — real positives AND injected eval anomalies, treated uniformly |
| `retries` | The generator's masked decode hit a real-graph collision and was re-rolled with fresh Gumbel noise |
| `uniform_fallbacks` | Retries exhausted → fell back to uniform-random replacement for that one slot |
| `slot_distribution` | Which slot was corrupted (head / relation / tail) — uniform random per positive (≈ 1/3 each) |

A healthy run has `uniform_fallbacks` near zero and a roughly uniform slot distribution.

---

## Step 4 — Compare results

Each `run_experiment.py` invocation prints a 5-row RESULTS table. Drop B0
and B1 side by side:

| K | B0 (random) | B1 (KGSAGE) | Δ |
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
  experiments/kgsage/slurm/train_gan_fb15k237.slurm
        └─ python -m kgsage.cli.train_gan
                └─ kgsage.gan.train.main()
                        ├─ kgsage.data.loaders.load_kg
                        ├─ kgsage.gan.models.{KGSAGEGenerator,KGSAGEDiscriminator}
                        └─ writes experiments/kgsage/outputs/checkpoints/kgsage_<name>.pt

Step 2 — ADKGD run (per experiment, B0 or B1)
  experiments/slurm/run_baseline{,_with_kgsage}_fb15k237.slurm
        └─ experiments/run_experiment.py            ← orchestrator (ours)
                ├─ subprocess: Our_TopK%_RankingList.py --mode train   (ADKGD upstream)
                │       └─ dataset.py:Reader.get_data()
                │               ├─ neg_source=random → generate_anomalous_triples()
                │               └─ neg_source=gan    → Reader._gan_negatives()
                │                       └─ experiments/kgsage_bridge/bridge.py
                │                               ├─ kgsage.inference.load_checkpoint() the .pt
                │                               └─ kgsage.inference.generate_negatives() — one single-slot corruption per positive
                ├─ subprocess: Our_TopK%_RankingList.py --mode test    (ADKGD upstream)
                └─ parses logs → prints RESULTS table
```

### Why this boundary

- **The GAN is invoked only via `experiments/kgsage_bridge/bridge.py`**. ADKGD's `dataset.py` knows nothing about the GAN's internals — it calls `generate(...)` and gets ADKGD-ID negatives back. The bridge handles `sys.path` setup and vocab string round-trips.
- **B0 and B1 share the orchestrator, RESULTS parser, and comparison table** — any metric delta is unambiguously attributable to the negative source.
- **The GAN and ADKGD evolve independently**. Retrain the GAN without touching ADKGD; change ADKGD without touching the GAN.

---

## Datasets

| Dataset | Files | Triples | Purpose |
|---|---|---|---|
| `dummy_kg` | `data/dummy_kg/{train,valid,test}.txt` | **18 unique × 60 = 1,080** | Smoke-test fixture (6 people, 3 relations, 4 countries). Replication forces `K=0.1%` math to produce ≥ 1. |
| `FB15K-237` | `data/FB15K-237/{train,valid,test}.txt` | 310,116 | Paper benchmark — Freebase 15K with 237 relations (inverse relations removed to prevent test leakage). Real research runs. |
| `WN18RR` | `data/WN18RR/{train,valid,test}.txt` | 93,003 | Paper benchmark (WordNet 18 with restricted relations: 40,943 entities, 11 relations). Real research runs. |

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
| `kgsage/slurm/train_gan_fb15k237.slurm` | `kgsage_train_gan_fb15k237` | `kgsage_train_gan_fb15k237-<jobid>.out.txt` |
| `slurm/run_baseline_fb15k237.slurm` | `adkgd_fb15k237` | `adkgd_fb15k237-<jobid>.out.txt` |
| `slurm/run_baseline_with_kgsage_fb15k237.slurm` | `adkgd_baseline_with_kgsage_fb15k237` | `adkgd_baseline_with_kgsage_fb15k237-<jobid>.out.txt` |
| `slurm/run_baseline_wn18rr.slurm` | `adkgd_wn18rr` | `adkgd_wn18rr-<jobid>.out.txt` |

### Three useful commands

```bash
# 1. Tail the latest log for a given slurm WITHOUT typing the job id
cd ~/ADKGD
tail -f "$(ls -t gan_train_fb15k237-*.out.txt | head -1)"

# 2. Tail a specific job id (you get this from `sbatch` or `squeue`)
tail -f gan_train_fb15k237-2886370.out.txt

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
    checkpoints/FB15K-237/ADKGD_FB15K-237_0.05_Neighbors39__log.txt

# Inspect epoch durations
cat checkpoints/FB15K-237/ADKGD_FB15K-237_epoch_times.txt

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
- KGSAGE package — [experiments/kgsage/](kgsage/) — GAN + inference, all in one.
- Bridge — [experiments/kgsage_bridge/bridge.py](kgsage_bridge/bridge.py) — the only file that knows about both worlds.
- KGSAGE (future) — [experiments/kgsage/](kgsage/), bridge at [experiments/kgsage_bridge/](kgsage_bridge/).
