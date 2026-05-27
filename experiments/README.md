# experiments/ — our wrapper layer over ADKGD

Everything in this folder is code **we authored** for running ADKGD as a
research baseline. ADKGD's own source (the model, the data reader, the
training/test loop) lives at the repo root and is **untouched here**.

## The boundary

| Layer | Lives where | Responsible for |
|---|---|---|
| **ADKGD upstream** | repo root — `Our_TopK%_RankingList.py`, `model.py`, `dataset.py`, `create_batch.py`, `data/` | training, scoring, writing logs/checkpoints |
| **Our wrapper** | this folder — `run_experiment.py`, `slurm/`, `RUNNING_ON_DEEPTHOUGHT.md` | orchestration (run train, then test), parsing logs into a clean RESULTS table, HPC launch |

A single experiment's data flow:

```
experiments/slurm/run_baseline_fb15k.slurm        ← HPC launcher (ours)
        │
        └─ $PY experiments/run_experiment.py …    ← orchestrator (ours)
                │
                ├─ subprocess: Our_TopK%_RankingList.py --mode train   (ADKGD's)
                │      → writes .ckpt + log.txt + epoch_times.txt under checkpoints/
                ├─ subprocess: Our_TopK%_RankingList.py --mode test    (ADKGD's)
                │      → appends Precision/Recall lines to log.txt
                └─ parses those files → prints the 5-row RESULTS table (ours)
```

## Datasets at a glance

| Dataset | Files | Triples | Purpose |
|---|---|---|---|
| `dummy_kg` | `data/dummy_kg/{train,valid,test}.txt` | **18 unique facts, replicated to 1,080** | Tiny smoke-test dataset (6 people, 3 relations, 4 countries — eyeball-friendly). The 60× replication is purely so ADKGD's `K=0.1%` cutoff math (`int(0.001 × num_original) >= 1`) doesn't crash. The vocabulary stays minimal so you can read the GAN's output and tell at a glance whether it's producing sensible "wrong-but-plausible" triples. |
| `FB15K` | `data/FB15K/{train,valid,test}.txt` | 310,116 | The paper's benchmark (FB15K-237). Real research runs. |
| `FB15K-mini` | `data/FB15K-mini/{train,valid,test}.txt` | 2,600 | Legacy 2k-triple subset; superseded by `dummy_kg` for local iteration but still works. |

## How to run

**Always from the repo root** (one directory above this one).

### Baseline (B0) — random training negatives, default behaviour

```bash
# Local CPU smoke test on dummy_kg (18 unique facts, replicated to 1,080 triples for
# ADKGD's K=0.1% math). OMP_NUM_THREADS=1 is set automatically by the script.
python experiments/run_experiment.py --dataset dummy_kg --anomaly_ratio 0.15 --max_epoch 1

# HPC GPU run (full FB15K-237, ~12 min/epoch on V100)
sbatch experiments/slurm/run_baseline_fb15k.slurm
```

### GAN variant (B1) — training negatives loaded from a TSV

```bash
# Local smoke against a stub TSV (proves the loader path; ~5 seconds end to end)
python experiments/run_experiment.py --dataset dummy_kg --anomaly_ratio 0.15 --max_epoch 1 \
    --neg_source gan --gan_neg_path data/dummy_kg/gan_negatives_stub.tsv

# HPC GPU run (expects data/FB15K/gan_negatives.tsv to exist)
sbatch experiments/slurm/run_gan_fb15k.slurm
```

The script writes its artifacts to `<repo-root>/checkpoints/<dataset>/`, not
under `experiments/`. That's intentional — ADKGD's own paths assume the repo
root as cwd, and we honour that.

## What's in this folder

| File | Purpose |
|---|---|
| [run_experiment.py](run_experiment.py) | The orchestrator. Wipes stale logs, invokes ADKGD's train then test, parses the resulting log files, prints the RESULTS table. Accepts `--neg_source {random,gan}` and `--gan_neg_path`. |
| [slurm/run_baseline_fb15k.slurm](slurm/run_baseline_fb15k.slurm) | DeepThought launcher for the **B0 baseline** (random negatives). Tesla V100, env activation, GPU pre-flight, calls `run_experiment.py` with defaults. |
| [slurm/run_gan_fb15k.slurm](slurm/run_gan_fb15k.slurm) | DeepThought launcher for the **B1 GAN variant**. Same as baseline but adds `--neg_source gan --gan_neg_path data/FB15K/gan_negatives.tsv`, plus a pre-flight check that the TSV exists. |
| [gan/make_stub_negatives.py](gan/make_stub_negatives.py) | One-off helper: produces a stub TSV using ADKGD's own random corrupter. Lets you rehearse the B1 pipeline end-to-end **before** the real GAN exists. |
| [gan/validate_gan_tsv.py](gan/validate_gan_tsv.py) | Offline pre-flight validator. Reads a TSV, checks vocab match, real-graph collisions, uniqueness, pool size. Run on the login node before any `sbatch` to fail fast at zero compute cost. |
| [RUNNING_ON_DEEPTHOUGHT.md](RUNNING_ON_DEEPTHOUGHT.md) | Step-by-step HPC guide: env setup, submission, monitoring, troubleshooting, CUDA-wheel selection, CPU-fallback variant. |

## GAN workflow (Phase B)

The full sequence from "I have a TSV" to "I have the B1 row of the results table." All commands run from the repo root.

### 1. Build (or refresh) the negatives TSV

If you have a real GAN, point ADKGD at its TSV directly. For rehearsal — before the real GAN exists — generate a stub:

```bash
# Local (dummy_kg, tiny, instant)
python experiments/gan/make_stub_negatives.py
# = python experiments/gan/make_stub_negatives.py --dataset dummy_kg --count 200
# → writes data/dummy_kg/gan_negatives_stub.tsv

# Cluster (full FB15K-237, ~30 s on the login node)
python experiments/gan/make_stub_negatives.py --dataset FB15K --count 400000 \
    --out data/FB15K/gan_negatives.tsv
```

`--count 400000` is sized for the largest anomaly ratio we'll run (15%) on FB15K-237, with headroom after collision filtering. The dummy_kg default of 200 is plenty for its 6-entity × 3-relation vocabulary (~90 unique anomalous triples possible in total).

### 2. Validate the TSV (5 seconds, login node)

```bash
python experiments/gan/validate_gan_tsv.py --dataset FB15K \
    --tsv data/FB15K/gan_negatives.tsv
```

Expected verdict: `OK to use.` (exit 0). If it prints `NOT READY` with specific reasons (unknown vocab > 5%, mode collapse, pool too small, real-graph leakage), fix the upstream generator before continuing.

### 3. Submit the GAN run

```bash
sbatch experiments/slurm/run_gan_fb15k.slurm
# → prints a job id, e.g. 2884600
squeue -u $USER
tail -f adkgd_gan_fb15k-<jobid>.out.txt
```

Watch for, in order:

- `Using GAN negatives: data/FB15K/gan_negatives.tsv (400000 lines)` — slurm pre-flight is happy
- `[GAN loader] pool: <X> valid | skipped: <Y> unknown-vocab, <Z> collide-with-original | sampling 325621` — ADKGD's loader is reading your file
- per-batch `Epoch: 0-N, pos_loss, neg_loss, Loss` lines
- the RESULTS block at the bottom (5 rows of Precision@K / Recall@K + total train time)

### 4. Compare against B0

| K | B0 (run_baseline_fb15k.slurm) | B1 (run_gan_fb15k.slurm) | Delta |
|---|---|---|---|
| 1% | 0.9581 (paper 0.951) | _from B1 .out.txt_ | _to fill_ |
| 2% | 0.8836 | _from B1 .out.txt_ | _to fill_ |
| 3% | 0.7852 | _from B1 .out.txt_ | _to fill_ |
| 4% | 0.6929 | _from B1 .out.txt_ | _to fill_ |
| 5% | 0.6148 | _from B1 .out.txt_ | _to fill_ |

### Common commands you'll reach for

```bash
# What account did your job use? (for slurm --account= troubleshooting)
sacct -j <jobid> --format=JobID,JobName,Account,Partition,State

# Tail the live training log (slurm output merges stderr → one file)
tail -f adkgd_gan_fb15k-<jobid>.out.txt

# Pull just the metric lines from ADKGD's detailed log
grep -E "Precision 0\.050000 -- 0\.0[12345]0000|Recall  0\.050000-- 0\.0[12345]0000" \
    checkpoints/FB15K/ADKGD_FB15K_0.05_Neighbors39__log.txt

# Inspect epoch durations
cat checkpoints/FB15K/ADKGD_FB15K_epoch_times.txt

# Cancel a submitted job
scancel <jobid>

# Inspect how many unique triples in your TSV (mode-collapse check)
wc -l data/FB15K/gan_negatives.tsv
sort -u data/FB15K/gan_negatives.tsv | wc -l   # should be ≥ 95% of the above

# Spot-check the file (do entries look like plausible-but-wrong KG facts?)
head -10 data/FB15K/gan_negatives.tsv
shuf -n 20 data/FB15K/gan_negatives.tsv         # if you have GNU shuf on the cluster
```

### Backward compatibility

Default `--neg_source` is `random`, so `experiments/slurm/run_baseline_fb15k.slurm` works exactly as before — no GAN file required. The GAN code path is only entered when `--neg_source gan` is explicitly passed.

## Why the separation matters

The research goal is to **swap ADKGD's random training negatives for
GAN-generated ones** (Phase B, implemented). That swap is **one conditional
branch inside `dataset.py`** (`Reader.get_data()` checks `args.neg_source`)
and **one new method** (`Reader.load_gan_negatives`) — it doesn't change
anything in this folder beyond the new flag plumbing. Keeping the wrapper
isolated means the orchestration + reporting pipeline is shared by both
variants: same `run_experiment.py`, same RESULTS-table parser, same comparison
table, so any difference in metrics is unambiguously attributable to the
training-negatives source.
