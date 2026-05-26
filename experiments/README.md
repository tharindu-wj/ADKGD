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

## How to run

**Always from the repo root** (one directory above this one):

```bash
# Local CPU smoke test (Windows: set OMP_NUM_THREADS=1 first; the script also defaults to it)
python experiments/run_experiment.py --dataset FB15K-mini --anomaly_ratio 0.15 --max_epoch 1

# HPC GPU run
sbatch experiments/slurm/run_baseline_fb15k.slurm
```

The script writes its artifacts to `<repo-root>/checkpoints/<dataset>/`, not
under `experiments/`. That's intentional — ADKGD's own paths assume the repo
root as cwd, and we honour that.

## What's in this folder

| File | Purpose |
|---|---|
| [run_experiment.py](run_experiment.py) | The orchestrator. Wipes stale logs, invokes ADKGD's train then test, parses the resulting log files, prints the RESULTS table. |
| [slurm/run_baseline_fb15k.slurm](slurm/run_baseline_fb15k.slurm) | DeepThought (Flinders HPC) launcher: Tesla V100 partition, env activation, GPU pre-flight, then `python experiments/run_experiment.py …`. |
| [RUNNING_ON_DEEPTHOUGHT.md](RUNNING_ON_DEEPTHOUGHT.md) | Step-by-step HPC guide: env setup, submission, monitoring, troubleshooting, CUDA-wheel selection, CPU-fallback variant. |

## Why the separation matters

The research goal is to **swap ADKGD's random training negatives for
GAN-generated ones** (Phase B). That swap is **one line inside `dataset.py`
upstream** — it does not change anything in this folder. Keeping the wrapper
isolated means Phase B is a focused, reviewable change to upstream code, and
the orchestration + reporting pipeline stays the same for both the baseline
(B0) and the GAN variant (B1).
