# Running ADKGD baseline on DeepThought (Flinders HPC)

How to reproduce the FB15K-237 column of the ADKGD paper's Table 2 on the
DeepThought HPC as a SLURM batch job on the **GPU partition (Tesla V100)**,
using [`run_experiment.py`](run_experiment.py) as the orchestrator and
[`experiments/slurm/run_baseline_fb15k237.slurm`](slurm/run_baseline_fb15k237.slurm) as the
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

One submission of `experiments/slurm/run_baseline_fb15k237.slurm` trains ADKGD on FB15K-237 at
`anomaly_ratio=0.05`, seed 0, for 1 epoch, then evaluates. The job's
`.out.txt` ends with:

```
============================================================
  RESULTS  (FB15K-237 @ anomaly_ratio=0.05, seed=0)
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
[`experiments/slurm/run_baseline_fb15k237.slurm`](slurm/run_baseline_fb15k237.slurm) if your
layout differs:

```bash
PROJECT_DIR="$HOME/ADKGD"
CONDA_ENV="$HOME/envs/adkgd"
```

Change experiment parameters (`--anomaly_ratio`, `--seed`, `--max_epoch`,
`--dataset`) at the **single `python run_experiment.py` line** at the bottom of
the script — never inside the `#SBATCH` header.

## 3. Submit, monitor, collect

Three slurm launchers per dataset. Submit them in this order:

| # | Launcher | Location | When |
|---|---|---|---|
| 1 | `train_gan_fb15k237.slurm` | `experiments/kgsage/slurm/` | **One-time** per dataset — produces the GAN checkpoint used by B1. Skip if a checkpoint already exists. |
| 2 | `run_baseline_fb15k237.slurm` | `experiments/slurm/` | **B0** — ADKGD baseline with random negatives (reproduces Wu et al. 2024 Table 2). |
| 3 | `run_baseline_with_kgsage_fb15k237.slurm` | `experiments/slurm/` | **B1** — ADKGD baseline, but with the trained KGSAGE GAN supplying single-slot-corruption negatives in-process. Requires step 1 to have produced `experiments/kgsage/outputs/checkpoints/kgsage_fb15k237.pt`. |

Step 2 (B0) and step 3 (B1) are independent — submit them in either order.
Step 1 must happen before step 3.

```bash
cd $HOME/ADKGD
git pull                                              # get latest run_experiment.py / slurm

# (one-time, only if no checkpoint yet) Train the GAN:
sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm

# B0 baseline
sbatch --test-only experiments/slurm/run_baseline_fb15k237.slurm     # dry-run: validate the script
sbatch experiments/slurm/run_baseline_fb15k237.slurm                 # real submit → prints a job id

# B1 baseline + the KGSAGE GAN negatives (after step 1 has finished)
sbatch experiments/slurm/run_baseline_with_kgsage_fb15k237.slurm

squeue -u $USER                                       # PD = pending, R = running
tail -f adkgd_fb15k237-<jobid>.out.txt                   # live log (training progress)
```

**Success** = job ends `COMPLETED`, the GPU pre-flight printed
`GPU: Tesla V100-...`, and the bottom of `adkgd_fb15k237-<jobid>.out.txt` (or
`adkgd_baseline_with_gan_fb15k237-<jobid>.out.txt` for B1) shows the RESULTS
block (5 P@K + 5 R@K values + Total train time). The trained model plus raw
logs sit under `checkpoints/FB15K-237/` in the project directory (gitignored).

> HPC storage is not backed up. Once you have the RESULTS block, save your
> `*.out.txt` somewhere durable — `/RDrive`, locally via `scp`, or pasted into
> your write-up.

## Useful commands cheat-sheet (while a job is running / after it finishes)

Two files hold the action. Pick whichever's more convenient:

| File | What's in it | Who wrote it |
|---|---|---|
| `adkgd_fb15k237-<jobid>.out.txt` (in the repo root) | The slurm job's combined stdout+stderr: GPU pre-flight, every per-batch loss, every Precision/Recall log line, and the **RESULTS block at the very end**. | SLURM (everything from the job) |
| `checkpoints/FB15K-237/ADKGD_FB15K-237_0.05_Neighbors39__log.txt` | ADKGD's own application log: per-batch losses + every `Precision/Recall <ratio> -- <K> : <val>` line (the 39 = `--num_neighbor=39`, ADKGD's subgraph size). | ADKGD's `logging.info(...)` via FileHandler |

**While the job is running** — pick one to tail:

```bash
# Slurm output: everything the job is producing, RESULTS will land at the end
tail -f adkgd_fb15k237-<jobid>.out.txt

# ADKGD's own log: same training info, but isolated to just this run's data
tail -f checkpoints/FB15K-237/ADKGD_FB15K-237_0.05_Neighbors39__log.txt
```

**After the job finishes** — pull the headline numbers:

```bash
# The 5-row RESULTS table + total train time
tail -n 15 adkgd_fb15k237-<jobid>.out.txt

# Just the Precision/Recall lines for K = 1..5% (skips the noisy array(…) dump)
grep -E "Precision 0\.050000 -- 0\.0[12345]0000|Recall  0\.050000-- 0\.0[12345]0000" \
    checkpoints/FB15K-237/ADKGD_FB15K-237_0.05_Neighbors39__log.txt

# Train time in seconds, one line per epoch
cat checkpoints/FB15K-237/ADKGD_FB15K-237_epoch_times.txt

# Final job state (definitive — was it COMPLETED, FAILED, TIMEOUT, OOM?)
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

**Did the GPU actually get used?** (catches a silent CPU fallback):

```bash
head -n 30 adkgd_fb15k237-<jobid>.out.txt | grep -E "GPU:|cuda"
# Want a line like:  GPU: Tesla V100-PCIE-32GB | torch 2.4.1+cu121 | cuda 12.1
```

**Copy artifacts to your laptop** (from your laptop terminal, not the HPC):

```bash
# The job's stdout (the RESULTS block lives at the bottom)
scp wick0167@deepthought.flinders.edu.au:/home/wick0167/ADKGD/adkgd_fb15k237-<jobid>.out.txt ./

# The whole checkpoints/FB15K-237 dir (raw log + epoch_times + .ckpt)
scp -r wick0167@deepthought.flinders.edu.au:/home/wick0167/ADKGD/checkpoints/FB15K-237 ./checkpoints/

# Incremental sync (best if doing this repeatedly):
rsync -avz wick0167@deepthought.flinders.edu.au:/home/wick0167/ADKGD/checkpoints/FB15K-237/ ./checkpoints/FB15K-237/
```

> The `Neighbors39` infix in the log filename is ADKGD's encoding of
> `--num_neighbor=39` (its default subgraph size: 39 head + 39 tail neighbors
> per triple). We never vary it, so that name is constant across all our runs.

## Why is my GPU job stuck PENDING? (diagnose + fix)

DeepThought has only ~5 Tesla V100s cluster-wide and weights **Fairshare**
heavily on the `gpu` partition, so a pending KGSAGE/ADKGD GPU job is almost
always one of: (a) your priority is low, (b) the partition is full, (c) you hit
a QOS/association limit, (d) a reservation is holding the nodes, or (e) your
`--time` exceeds the partition cap. Work through the five angles below.

First, get the one-word **reason** SLURM attaches to your pending job — it tells
you which angle to chase:

```bash
squeue -u $USER -t PD -o "%.10i %.9P %.30j %.8T %.10M %.10l %.6D %R"
#                                                                    └ REASON in parens, e.g.
#  (Priority)        → other jobs outrank you            → angle A
#  (Resources)       → you're next, but no free V100 yet → angle B
#  (QOSMaxJobsPerUserLimit / QOSMax... / AssocMax...)     → angle C
#  (ReqNodeNotAvail, Reserved...)                         → angle D
#  (PartitionTimeLimit)  → your --time > partition MaxTime→ angle E
```

The `%R` field prints the reason for pending jobs in parentheses; map it to the
matching angle.

For the **authoritative, full picture of one job** — the reason plus its
priority, state, scheduling timestamps, and the exact resources it requested —
use `scontrol show job`:

```bash
scontrol show job <jobid>
#  JobState=PENDING  Reason=<code>      ← the definitive reason (matches %R above)
#  Priority=<int>                       ← compare against the running jobs
#  TRES=...,gres/gpu=1  Partition=gpu   ← confirm it really asked for a V100 on the gpu partition
#  EligibleTime in the FUTURE           ← a --begin hold ;  StartTime = estimate while pending
```

A silent trap this catches: a typo'd `--gres`/`--partition` so the job never
actually requests a V100 and pends forever for the wrong reason.

### A. "My priority is low" (reason: `Priority`)

See the weighted breakdown of your job's priority, then check whether your own
recent GPU usage is what's dragging the Fairshare component down:

```bash
sprio -j <jobid>           # this job's PRIORITY split into AGE / FAIRSHARE / QOS / JOBSIZE / PARTITION
sprio -l -u $USER          # all your pending jobs, long form, same breakdown
sprio -w                   # the cluster's weight for each factor (which factor dominates here)

sshare -U                  # YOUR fairshare line: RawShares, RawUsage, EffectvUsage, FairShare (0..1)
sshare -u $USER -a         # your account + siblings, to see if the account (not just you) is over-used
```

**What it reveals.** `sprio` shows whether the FAIRSHARE column (not AGE) is the
thing holding you down. In `sshare`, the **FairShare** factor runs 0..1: ~0.5 =
your fair share, **<0.5 = you have over-consumed** your GPU share recently (it
decays over ~30 days via `PriorityDecayHalfLife`), >0.5 = you're under your
share and should be favored. A high `RawUsage`/`EffectvUsage` after a burst of
V100 jobs is the classic cause of a low-priority pend.

**Remedy.** You can't out-argue Fairshare directly — but (1) **let AGE
accumulate**: don't cancel/resubmit, since requeuing resets the AGE factor that
would otherwise lift you over time; (2) **stop burning share** — move
non-GPU steps to the `general` partition (see Backfill tips) so they don't
charge against your GPU usage; (3) **space out** large V100 submissions so your
RawUsage decays back toward your fair share before the next batch.

### B. "The partition is full" (reason: `Resources`)

```bash
sinfo -p gpu -N -o "%.18N %.6t %.8O %.20G %.10e/%.10m"   # per-V100-node: state, CPU load, GRES, free/total mem
sinfo -p gpu -t idle,mix -o "%.18N %.6t %G"              # are ANY V100s actually free right now?
squeue -p gpu -t R -o "%.10i %.9u %.8a %.12L %.10l %.6D %R"  # who's RUNNING, their TIME-LEFT (%L) and TIME-LIMIT (%l)
```

**What it reveals.** `sinfo` node states: `idle`=free, `mix`=partly used,
`alloc`=full, `drain`/`down`=unavailable (so the effective V100 count is below
5). The `squeue -p gpu -t R` listing shows the *other* jobs sitting on the
V100s and — crucially — their `%L` **time-left** and `%l` **time-limit**: a
couple of multi-day jobs can wall off the whole partition.

**Remedy.** If every V100 is `alloc`/`drain`, the only fast lever you control is
making your job **fit a backfill gap before those running jobs end** — drop your
own `--time` (see Backfill tips). If nodes show `drain`/`down`, that's an admin
issue worth flagging; there's nothing you can submit your way around.

### C. "I hit a QOS / association limit" (reason: `QOSMax...` / `AssocMax...`)

```bash
sacctmgr show qos format=Name,Priority,MaxWall,MaxTRESPU,MaxJobsPU,MaxSubmitPU   # per-QOS caps
sacctmgr show assoc user=$USER format=Account,User,QOS,MaxJobs,GrpTRES           # what YOU are capped at
```

**What it reveals.** `MaxJobsPU` (max running jobs per user), `MaxSubmitPU` (max
running+pending per user), `MaxTRESPU` (e.g. a cap of `gres/gpu=1` per user),
and `MaxWall` (per-job walltime ceiling for that QOS). If `squeue` said
`QOSMaxJobsPerUserLimit` or `AssocMaxJobsLimit`, you've already hit one of these
— a second GPU job won't start until your first finishes.

**Remedy.** Respect the cap: run GPU jobs **one at a time** (don't queue B0 and
B1 to fight for the same single-GPU slot — chain them, or submit B1 only after
B0 starts). If `MaxWall` is below your `--time`, lower `--time` to comply. These
limits are policy; raising them needs an admin request.

### D. "A reservation is blocking the nodes" (reason: `ReqNodeNotAvail`/`Reserved`)

```bash
scontrol show reservation                       # any active/upcoming reservation, and which Nodes/Users it covers
scontrol show res -o | grep -i v100             # quick check whether a reservation grabs the V100 nodes
```

**What it reveals.** Maintenance or course/project reservations can lock the
V100 nodes to specific `Users=`/`Accounts=` during a window. If a reservation
covers the GPU nodes and your user isn't in its `Users=` list, your job waits
until `EndTime`, even though the hardware looks idle.

**Remedy.** Read the reservation's `EndTime` and either wait it out, or — if your
work is short — set `--time` small enough to **backfill before the reservation
StartTime**. If you were *supposed* to be granted that reservation, ask the
admins to add your user/account to it.

### E. "The partition rejects my --time" (reason: `PartitionTimeLimit`)

```bash
scontrol show partition gpu                     # MaxTime (hard cap) and DefaultTime for the gpu partition
```

**What it reveals.** `MaxTime` is the partition's hard walltime ceiling. If your
`#SBATCH --time` exceeds it, the job is rejected/held with
`PartitionTimeLimit` and will never start as-is.

**Remedy.** Lower `#SBATCH --time` to at or below `MaxTime`. A single ADKGD
FB15K-237 epoch is ~13 min on a V100, so a generous `--time=01:00:00` is plenty
and sits far under any sane partition cap — over-requesting walltime only hurts
your backfill chances anyway (next section).

### Backfill tips: get a contended GPU job to start sooner

SLURM's backfill scheduler will start a **lower-priority job early** if it fits
in the gap before a higher-priority job is due to start — but only if its
`--time` is short enough to finish within that gap. On a partition with ~5
V100s and long-running neighbors, your walltime estimate is your main lever.

1. **Cut `#SBATCH --time` to what the job actually needs.** One FB15K-237 epoch
   ≈ 13 min on a V100; `--time=00:30:00` (or `01:00:00` with margin) makes your
   job eligible for far more backfill windows than a default multi-hour request.
   This is the single biggest mover.

2. **Check whether a shorter walltime actually pulls the estimate earlier:**

   ```bash
   squeue -u $USER --start -o "%.10i %.9P %.20S %.10l %R"   # predicted StartTime (%S) for your pending job
   ```

   Lower `--time`, resubmit, and re-run this — if `%S` jumps earlier, backfill
   is rewarding you. (The estimate is a guideline, not a guarantee.)

3. **Request fewer resources.** Ask for exactly `--gres=gpu:tesla_v100:1`, one
   task, and only the memory/CPUs you need (`--mem=16G` is enough for ADKGD at
   batch 256). A smaller footprint fits more gaps; the JOBSIZE priority factor
   also gives small jobs a slight backfill boost.

4. **Run the non-GPU steps on `general`, not `gpu`.** The pipeline's CPU-only
   steps — e.g. any `dummy_kg` smoke test or a small-dataset GAN run — should go to the much
   more available `general` partition (`#SBATCH --partition=general`, drop the
   `--gres` line; see the "Running on CPU" section). This both starts those
   steps immediately **and** keeps them from charging against your GPU Fairshare
   usage, which protects your priority for the runs that genuinely need a V100.

5. **One GPU job at a time.** With a likely 1-GPU-per-user cap (angle C), a
   second pending GPU job just waits anyway — submit B1 after B0 has started so
   you're not self-blocking, and don't cancel/resubmit (it resets your AGE
   priority).

### After the fact: how long did a late job actually wait?

Once the job has started or finished, quantify the queue wait so you know whether
it's worth optimising:

```bash
sacct -j <jobid> -X --format=JobID,Partition,Submit,Eligible,Start,End,Elapsed,Planned,State,ExitCode
#  Submit -> Start  = the real queue wait      |  -X = job allocation only (hides .batch/.extern steps)
#  Eligible later than Submit = a hold/dependency delayed eligibility, not the scheduler
```

> On older Slurm the eligible-wait column is named `Reserved`, not `Planned`. If
> you get `invalid field name`, run `sacct --helpformat` and swap
> `Planned` → `Reserved`.

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
sbatch experiments/slurm/run_baseline_fb15k237.slurm
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `module: command not found` or `Miniconda3` missing | `module avail miniconda` and use the exact name (maybe `miniconda/3.0`). |
| `CommandNotFoundError: conda activate` | The `source "$(conda info --base)/etc/profile.d/conda.sh"` line must run before `conda activate`. |
| `/bin/bash^M: bad interpreter` | CRLF line endings from a Windows checkout. `dos2unix experiments/slurm/run_baseline_fb15k237.slurm` on the HPC. |
| Pre-flight assert: `torch cannot see a GPU` | CPU-only torch wheel got installed. Reinstall per step 4 above. |
| `ModuleNotFoundError` for torch / numpy / sklearn / matplotlib | The env wasn't built or wasn't activated — redo step 1; confirm `CONDA_ENV` path in the slurm script. |
| Job killed, `oom-kill` in log | Raise `--mem` in the script (e.g. 16G → 32G). FB15K-237 at batch 256 should fit in 16G; only an issue if you bump batch size. |
| Job pending forever | GPU partition is busy (only 5 V100s cluster-wide, heavy Fairshare weight). Read the `%R` reason with `squeue -u $USER -t PD -o "%.10i %.8T %R"`, then follow the matching angle in **Why is my GPU job stuck PENDING?**; lowering `--time` helps backfill, and `squeue -u $USER --start` shows the predicted start time. |
| RESULTS block missing from `.out.txt` | The Python script crashed before printing. Check `adkgd_fb15k237-<jobid>.err.txt` for the traceback; also scroll up in the `.out.txt`. Usually a missing dep or `PROJECT_DIR` mismatch. |
| K columns are 2/4/6/8/10% instead of 1/2/3/4/5% | You set `--anomaly_ratio 0.10`. The K cutoffs scale to `anomaly_ratio · i/5` for `i ∈ 1..5`: at 10% they're 2..10%, at 15% they're 3..15%. (Paper convention; reflected in `run_experiment.py`.) |
| Segfault very early in training (rare on GPU) | If torch fell back to CPU silently, an OMP/MKL conflict can crash it. `run_experiment.py` already sets defensive defaults via `os.environ.setdefault`; if it still bites, explicitly `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` before the python line. |
| `dataset FB15K-237 not found` / FileNotFoundError on `data/FB15K-237/...` | `PROJECT_DIR` is wrong, or `data/FB15K-237/` wasn't pulled. Check `ls $PROJECT_DIR/data/FB15K-237/` shows `train.txt valid.txt test.txt`. |

## Running on CPU (general partition) — optional

The CPU `general` partition is much more available than GPU, but ADKGD on
FB15K-237 at batch 256 is genuinely slow on CPU (paper reports ~13 min/epoch
on V100; expect 5–10× slower on CPU). It's useful for:

- **Smoke-testing** the full pipeline on `dummy_kg` (the 18-fact toy KG replicated
  to 1,080 triples; data/dummy_kg/{train,valid,test}.txt). A single epoch finishes
  in well under a minute.
- Sanity-checking your env without burning a scarce V100 slot.

To make a CPU variant, copy `run_baseline_fb15k237.slurm` and:

- Set `#SBATCH --partition=general` (drop the `gpu` partition line)
- Delete `#SBATCH --gres=gpu:tesla_v100:1`
- Remove the `python -c "import torch; assert torch.cuda.is_available() ..."` pre-flight (it would fail)
- Install the CPU wheel into the env: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Optionally swap `--dataset FB15K-237` for `--dataset dummy_kg` for a fast sanity test (the older `FB15K-mini` subset still works too if you prefer ~2,600 triples)

`run_experiment.py` already sets OMP/MKL thread defaults that prevent the CPU
torch from segfaulting on multi-core nodes.

## What this baseline row feeds into

This document covers the HPC operations for the full pipeline (ADKGD baseline
B0, GAN training, and ADKGD-with-GAN B1). The step-by-step research
walkthrough lives in [README.md](README.md); this doc focuses on the cluster
specifics: env, submission, monitoring, troubleshooting.

For the end-to-end research workflow (train the GAN → run B0 → run B1 → compare
RESULTS), follow the four steps in `README.md`. The slurm launchers in
`experiments/slurm/` map 1:1 to those steps.
