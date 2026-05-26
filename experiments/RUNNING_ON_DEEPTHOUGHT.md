# Running ADKGD baseline on DeepThought (Flinders HPC)

How to reproduce the FB15K-237 column of the ADKGD paper's Table 2 on the
DeepThought HPC as a SLURM batch job on the **GPU partition (Tesla V100)**,
using [`run_experiment.py`](run_experiment.py) as the orchestrator and
[`experiments/slurm/run_baseline_fb15k.slurm`](slurm/run_baseline_fb15k.slurm) as the
launcher.

## Why batch (not the login node)

The login node (`hpc-head01`) is for editing and submitting jobs only — its
`python`/`python3` are Python 2 / 3.6 (too old for PyTorch 2.x), it has no GPU,
and heavy compute there is not allowed. Real work runs on compute nodes via
SLURM, inside a Miniconda environment you build yourself.

**Compute nodes have no internet**, so every dependency must be installed on
the login node *before* you submit. The job only invokes already-installed
packages — it never reaches out to PyPI mid-run.

## What one job produces

One submission of `experiments/slurm/run_baseline_fb15k.slurm` trains ADKGD on FB15K-237 at
`anomaly_ratio=0.05`, seed 0, for 1 epoch, then evaluates. The job's
`.out.txt` ends with:

```
============================================================
  RESULTS  (FB15K @ anomaly_ratio=0.05, seed=0)
============================================================
     K   Precision@K    Recall@K
------  ------------  ----------
    1%        0.9xxx      0.1xxx
    2%        0.8xxx      0.3xxx
    ...
    5%        0.6xxx      0.6xxx
Total train time: XX.XX minutes (1 epoch(s))
```

Five Precision@K + five Recall@K + one Total-train-time number per run. That
is the entire "ADKGD (Baseline)" row of your comparison table.

## Prerequisites

- HPC access + SSH to `deepthought.flinders.edu.au` (Flinders VPN if off-campus).
- This repo cloned on the HPC, e.g. `~/ADKGD`.

## 1. One-time environment setup (login node — has internet)

```bash
module load Miniconda3                                # if not found: `module avail miniconda`
conda create -y -p $HOME/envs/adkgd python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $HOME/envs/adkgd

pip install --upgrade pip
# CUDA torch wheel matching the V100 node's driver (cu121 is the safe default; see step 4 if torch can't see the GPU):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy scikit-learn matplotlib

# sanity check
python -c "import torch, numpy, sklearn, matplotlib; print('ok torch', torch.__version__, '| cuda build', torch.version.cuda)"
```

Python 3.11 is fine because this is a batch job — no kernel version constraint
to worry about.

## 2. Point the job script at your paths

Edit the two variables at the top of
[`experiments/slurm/run_baseline_fb15k.slurm`](slurm/run_baseline_fb15k.slurm) if your
layout differs:

```bash
PROJECT_DIR="$HOME/ADKGD"
CONDA_ENV="$HOME/envs/adkgd"
```

Change experiment parameters (`--anomaly_ratio`, `--seed`, `--max_epoch`,
`--dataset`) at the **single `python run_experiment.py` line** at the bottom of
the script — never inside the `#SBATCH` header.

## 3. Submit, monitor, collect

```bash
cd $HOME/ADKGD
git pull                                              # get latest run_experiment.py / slurm

sbatch --test-only experiments/slurm/run_baseline_fb15k.slurm     # dry-run: validate the script
sbatch experiments/slurm/run_baseline_fb15k.slurm                 # real submit → prints a job id

squeue -u $USER                                       # PD = pending, R = running
tail -f adkgd_fb15k-<jobid>.out.txt                   # live log (training progress)
```

**Success** = job ends `COMPLETED`, the GPU pre-flight printed
`GPU: Tesla V100-...`, and the bottom of `adkgd_fb15k-<jobid>.out.txt` shows
the RESULTS block (5 P@K + 5 R@K values + Total train time). The trained model
plus raw logs sit under `checkpoints/FB15K/` in the project directory
(gitignored).

> HPC storage is not backed up. Once you have the RESULTS block, save your
> `*.out.txt` somewhere durable — `/RDrive`, locally via `scp`, or pasted into
> your write-up.

## Useful commands cheat-sheet (while a job is running / after it finishes)

Two files hold the action. Pick whichever's more convenient:

| File | What's in it | Who wrote it |
|---|---|---|
| `adkgd_fb15k-<jobid>.out.txt` (in the repo root) | The slurm job's combined stdout+stderr: GPU pre-flight, every per-batch loss, every Precision/Recall log line, and the **RESULTS block at the very end**. | SLURM (everything from the job) |
| `checkpoints/FB15K/ADKGD_FB15K_0.05_Neighbors39__log.txt` | ADKGD's own application log: per-batch losses + every `Precision/Recall <ratio> -- <K> : <val>` line (the 39 = `--num_neighbor=39`, ADKGD's subgraph size). | ADKGD's `logging.info(...)` via FileHandler |

**While the job is running** — pick one to tail:

```bash
# Slurm output: everything the job is producing, RESULTS will land at the end
tail -f adkgd_fb15k-<jobid>.out.txt

# ADKGD's own log: same training info, but isolated to just this run's data
tail -f checkpoints/FB15K/ADKGD_FB15K_0.05_Neighbors39__log.txt
```

**After the job finishes** — pull the headline numbers:

```bash
# The 5-row RESULTS table + total train time
tail -n 15 adkgd_fb15k-<jobid>.out.txt

# Just the Precision/Recall lines for K = 1..5% (skips the noisy array(…) dump)
grep -E "Precision 0\.050000 -- 0\.0[12345]0000|Recall  0\.050000-- 0\.0[12345]0000" \
    checkpoints/FB15K/ADKGD_FB15K_0.05_Neighbors39__log.txt

# Train time in seconds, one line per epoch
cat checkpoints/FB15K/ADKGD_FB15K_epoch_times.txt

# Final job state (definitive — was it COMPLETED, FAILED, TIMEOUT, OOM?)
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

**Did the GPU actually get used?** (catches a silent CPU fallback):

```bash
head -n 30 adkgd_fb15k-<jobid>.out.txt | grep -E "GPU:|cuda"
# Want a line like:  GPU: Tesla V100-PCIE-32GB | torch 2.4.1+cu121 | cuda 12.1
```

**Copy artifacts to your laptop** (from your laptop terminal, not the HPC):

```bash
# The job's stdout (the RESULTS block lives at the bottom)
scp wick0167@deepthought.flinders.edu.au:/home/wick0167/ADKGD/adkgd_fb15k-<jobid>.out.txt ./

# The whole checkpoints/FB15K dir (raw log + epoch_times + .ckpt)
scp -r wick0167@deepthought.flinders.edu.au:/home/wick0167/ADKGD/checkpoints/FB15K ./checkpoints/

# Incremental sync (best if doing this repeatedly):
rsync -avz wick0167@deepthought.flinders.edu.au:/home/wick0167/ADKGD/checkpoints/FB15K/ ./checkpoints/FB15K/
```

> The `Neighbors39` infix in the log filename is ADKGD's encoding of
> `--num_neighbor=39` (its default subgraph size: 39 head + 39 tail neighbors
> per triple). We never vary it, so that name is constant across all our runs.

## 4. Picking the right CUDA wheel

`cu121` is the safe default for the current V100 nodes. If the pre-flight
assert in the slurm script fails with
`torch cannot see a GPU: install the CUDA wheel`, check what the GPU node
exposes and reinstall:

```bash
# 1. Check the GPU node's driver-supported CUDA version (note "CUDA Version: XX.X" top-right):
srun --partition=gpu --gres=gpu:tesla_v100:1 --time=00:05:00 --pty nvidia-smi

# 2. Reinstall torch with the matching wheel (login node):
conda activate $HOME/envs/adkgd
python -m pip uninstall -y torch
#   driver shows CUDA >= 12.1  →  cu121 ;  CUDA 11.x  →  cu118
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda)"

# 3. Resubmit:
sbatch experiments/slurm/run_baseline_fb15k.slurm
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `module: command not found` or `Miniconda3` missing | `module avail miniconda` and use the exact name (maybe `miniconda/3.0`). |
| `CommandNotFoundError: conda activate` | The `source "$(conda info --base)/etc/profile.d/conda.sh"` line must run before `conda activate`. |
| `/bin/bash^M: bad interpreter` | CRLF line endings from a Windows checkout. `dos2unix experiments/slurm/run_baseline_fb15k.slurm` on the HPC. |
| Pre-flight assert: `torch cannot see a GPU` | CPU-only torch wheel got installed. Reinstall per step 4 above. |
| `ModuleNotFoundError` for torch / numpy / sklearn / matplotlib | The env wasn't built or wasn't activated — redo step 1; confirm `CONDA_ENV` path in the slurm script. |
| Job killed, `oom-kill` in log | Raise `--mem` in the script (e.g. 16G → 32G). FB15K-237 at batch 256 should fit in 16G; only an issue if you bump batch size. |
| Job pending forever | GPU partition is busy (only 5 V100s cluster-wide, heavy Fairshare weight). `squeue -u $USER --start` shows the predicted start time; lowering `--time` helps backfill. |
| RESULTS block missing from `.out.txt` | The Python script crashed before printing. Check `adkgd_fb15k-<jobid>.err.txt` for the traceback; also scroll up in the `.out.txt`. Usually a missing dep or `PROJECT_DIR` mismatch. |
| K columns are 2/4/6/8/10% instead of 1/2/3/4/5% | You set `--anomaly_ratio 0.10`. The K cutoffs scale to `anomaly_ratio · i/5` for `i ∈ 1..5`: at 10% they're 2..10%, at 15% they're 3..15%. (Paper convention; reflected in `run_experiment.py`.) |
| Segfault very early in training (rare on GPU) | If torch fell back to CPU silently, an OMP/MKL conflict can crash it. `run_experiment.py` already sets defensive defaults via `os.environ.setdefault`; if it still bites, explicitly `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` before the python line. |
| `dataset FB15K not found` / FileNotFoundError on `data/FB15K/...` | `PROJECT_DIR` is wrong, or `data/FB15K/` wasn't pulled. Check `ls $PROJECT_DIR/data/FB15K/` shows `train.txt valid.txt test.txt`. |

## Running on CPU (general partition) — optional

The CPU `general` partition is much more available than GPU, but ADKGD on
FB15K-237 at batch 256 is genuinely slow on CPU (paper reports ~13 min/epoch
on V100; expect 5–10× slower on CPU). It's useful for:

- **Smoke-testing** the full pipeline on `FB15K-mini` (the 2,000-triple subset
  committed to the repo); a single epoch finishes in ~1.5 min.
- Sanity-checking your env without burning a scarce V100 slot.

To make a CPU variant, copy `run_baseline_fb15k.slurm` and:

- Set `#SBATCH --partition=general` (drop the `gpu` partition line)
- Delete `#SBATCH --gres=gpu:tesla_v100:1`
- Remove the `python -c "import torch; assert torch.cuda.is_available() ..."` pre-flight (it would fail)
- Install the CPU wheel into the env: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Optionally swap `--dataset FB15K` for `--dataset FB15K-mini` for a fast sanity test

`run_experiment.py` already sets OMP/MKL thread defaults that prevent the CPU
torch from segfaulting on multi-core nodes.

## What this baseline row feeds into

This document covers Phase A — establishing the ADKGD baseline. The eventual
Phase B (replace the random training negatives with a GAN-generated pool)
re-uses the same slurm + `run_experiment.py` setup with one extra flag pair
(`--neg_source gan --gan_neg_path ...`), so once the Phase A row is in your
table the GAN row drops in alongside it without further infrastructure work.
