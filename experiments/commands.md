# KGSAGE — copy-paste command reference (GAN-only workflow)

Run these from the repo root on DeepThought (`cd ~/ADKGD` first). Each block
is self-contained — copy the whole block into your terminal.

Dataset naming gotcha: the two scripts spell the dataset differently.
- `train.slurm` wants the short tag: `fb15k237` / `wn18rr`
- `exp_cell.slurm` wants the data folder name: `FB15K-237` / `WN18RR`

---

## 0. One-time setup (login node)

```bash
cd ~/ADKGD
PYTHONPATH=experiments ~/envs/adkgd/bin/python -m kgsage.cli.fetch_lp --dataset all
# expect: GATE PASS fb15k237 mrr≈0.3477 · wn18rr mrr≈0.4749
```

---

## 1. Train the GAN → checkpoint (GPU job, one per dataset)

### FB15K-237

```bash
DATASET=fb15k237 SEED=0 sbatch experiments/kgsage/slurm/train.slurm
```

Produces `experiments/kgsage/outputs/checkpoints/kgsage_fb15k237_s0.pt`.
Watch it:

```bash
tail -f kgsage_train-<jobid>.out.txt
```

Healthy signs: `copy=0.00%` always, `D-acc real/fake` around 0.78/0.87 (not
pinned at 0.50 or 1.00), `distinct-picks` settling (not collapsing toward
single digits/low hundreds).

### WN18RR

```bash
DATASET=wn18rr SEED=0 sbatch experiments/kgsage/slurm/train.slurm
```

Produces `experiments/kgsage/outputs/checkpoints/kgsage_wn18rr_s0.pt`.

> **Known issue on WN18RR (from the 2026-07-03 run):** default settings
> mode-collapsed — `D-acc fake=1.00` from epoch 1, `fence-hit=0%`,
> `distinct-picks` fell from ~18,700 to ~110. If you see the same pattern,
> **do not use that checkpoint for gan cells.** Retrain with the escalation
> knobs (cheapest first):
>
> ```bash
> DATASET=wn18rr SEED=0 LAMBDA_H=0.05 sbatch experiments/kgsage/slurm/train.slurm
> # if still collapsed, also lower the discriminator's effective learning rate
> # by editing --lr_d in kgsage/gan/train.py, or raise BAND_TEMP:
> DATASET=wn18rr SEED=0 LAMBDA_H=0.05 BAND_TEMP=1.0 sbatch experiments/kgsage/slurm/train.slurm
> ```

### Second seed (needed later to de-circularize the "recovery" cell — optional for now)

```bash
DATASET=fb15k237 SEED=1 sbatch experiments/kgsage/slurm/train.slurm
DATASET=wn18rr   SEED=1 sbatch experiments/kgsage/slurm/train.slurm
```

---

## 2. Verify the checkpoint by hand (login node, CPU)

### Random-sample check

```bash
# FB15K-237
PYTHONPATH=experiments ~/envs/adkgd/bin/python -m kgsage.cli.inspect_gan_lp \
  --ckpt experiments/kgsage/outputs/checkpoints/kgsage_fb15k237_s0.pt \
  --data data/FB15K-237 \
  --lp_ckpt experiments/kgsage/outputs/lp/fb15k-237-complex.pt \
  --lp_ids  experiments/kgsage/outputs/lp/fb15k-237 --n 40

# WN18RR
PYTHONPATH=experiments ~/envs/adkgd/bin/python -m kgsage.cli.inspect_gan_lp \
  --ckpt experiments/kgsage/outputs/checkpoints/kgsage_wn18rr_s0.pt \
  --data data/WN18RR \
  --lp_ckpt experiments/kgsage/outputs/lp/wnrr-complex.pt \
  --lp_ids  experiments/kgsage/outputs/lp/wnrr --n 40
```

### Pick-your-own triples check

```bash
# FB15K-237 — grab a relation you can reason about, then corrupt it
grep place_of_birth data/FB15K-237/train.txt | head -8 > my_triples_fb.tsv
PYTHONPATH=experiments ~/envs/adkgd/bin/python -m kgsage.cli.corrupt \
  --ckpt experiments/kgsage/outputs/checkpoints/kgsage_fb15k237_s0.pt \
  --triples my_triples_fb.tsv \
  --lp_ckpt experiments/kgsage/outputs/lp/fb15k-237-complex.pt \
  --lp_ids  experiments/kgsage/outputs/lp/fb15k-237

# WN18RR — e.g. hypernym relation
grep _hypernym data/WN18RR/train.txt | head -8 > my_triples_wn.tsv
PYTHONPATH=experiments ~/envs/adkgd/bin/python -m kgsage.cli.corrupt \
  --ckpt experiments/kgsage/outputs/checkpoints/kgsage_wn18rr_s0.pt \
  --triples my_triples_wn.tsv \
  --lp_ckpt experiments/kgsage/outputs/lp/wnrr-complex.pt \
  --lp_ids  experiments/kgsage/outputs/lp/wnrr
```

Read: positive `gap` = corruption scores below the truth (good, it's false).
Negative `gap` / `ABOVE TRUE` flag = possible false negative.

---

## 3. Run the 4-cell matrix (GPU jobs)

### FB15K-237

```bash
CKPT=experiments/kgsage/outputs/checkpoints/kgsage_fb15k237_s0.pt

# ① baseline      — train random, test random   (no GAN needed)
NEG_SOURCE=random TEST_SOURCE=random DATASET=FB15K-237 SEED=0 \
  sbatch experiments/slurm/exp_cell.slurm

# ② the gap       — train random, test GAN
NEG_SOURCE=random TEST_SOURCE=gan DATASET=FB15K-237 SEED=0 GAN_CKPT=$CKPT \
  sbatch experiments/slurm/exp_cell.slurm

# ③ no-regression — train GAN, test random
NEG_SOURCE=gan TEST_SOURCE=random DATASET=FB15K-237 SEED=0 GAN_CKPT=$CKPT \
  sbatch experiments/slurm/exp_cell.slurm

# ④ the recovery  — train GAN, test GAN
NEG_SOURCE=gan TEST_SOURCE=gan DATASET=FB15K-237 SEED=0 GAN_CKPT=$CKPT \
  sbatch experiments/slurm/exp_cell.slurm
```

### WN18RR

```bash
CKPT=experiments/kgsage/outputs/checkpoints/kgsage_wn18rr_s0.pt

# ① baseline      — train random, test random   (no GAN needed)
NEG_SOURCE=random TEST_SOURCE=random DATASET=WN18RR SEED=0 \
  sbatch experiments/slurm/exp_cell.slurm

# ② the gap       — train random, test GAN
NEG_SOURCE=random TEST_SOURCE=gan DATASET=WN18RR SEED=0 GAN_CKPT=$CKPT \
  sbatch experiments/slurm/exp_cell.slurm

# ③ no-regression — train GAN, test random
NEG_SOURCE=gan TEST_SOURCE=random DATASET=WN18RR SEED=0 GAN_CKPT=$CKPT \
  sbatch experiments/slurm/exp_cell.slurm

# ④ the recovery  — train GAN, test GAN
NEG_SOURCE=gan TEST_SOURCE=gan DATASET=WN18RR SEED=0 GAN_CKPT=$CKPT \
  sbatch experiments/slurm/exp_cell.slurm
```

> Only run the WN18RR gan cells (②③④) once you have a WN18RR checkpoint that
> passed the quality-gate check above — a collapsed generator will make ②③④
> meaningless (it emits near-constant/degenerate corruptions).

Optional: add `MAX_EPOCH=5` to any cell above once the 1-epoch numbers look
sensible, for firmer results.

---

## 4. Read the results

```bash
~/envs/adkgd/bin/python experiments/aggregate_results.py --dataset FB15K-237
~/envs/adkgd/bin/python experiments/aggregate_results.py --dataset WN18RR
```

Expected shape per dataset: ① high (easy) → ② collapses (proves the
problem) → ③ stays near ① (no downside) → ④ climbs back up (GAN helps).

---

## Reference: env knobs

| Script | Knob | Default | Notes |
|---|---|---|---|
| `train.slurm` | `DATASET` | `fb15k237` | `fb15k237` \| `wn18rr` |
| | `SEED` | `0` | |
| | `WARMUP_EPOCHS` | `10` | |
| | `WARMSTART_EPOCHS` | `2` | |
| | `EPOCHS` | `30` | adversarial phase |
| | `BATCH_SIZE` | `512` | |
| | `TAU` | `0.5` | Gumbel temperature |
| | `BAND_K` | `10` | band-teacher width |
| | `FENCE_SIGMA` | `0.5` | truth-fence margin |
| | `LAMBDA_H` | `0.01` | entropy bonus — raise if mode collapse |
| `exp_cell.slurm` | `DATASET` | `FB15K-237` | `FB15K-237` \| `WN18RR` |
| | `SEED` | `0` | |
| | `MAX_EPOCH` | `1` | ADKGD detector epochs |
| | `ANOMALY_RATIO` | `0.05` | |
| | `NEG_SOURCE` | `random` | `random` \| `lp_band` \| `gan` |
| | `TEST_SOURCE` | `random` | `random` \| `lp_band` \| `gan` |
| | `GAN_CKPT` | — | required when either source is `gan` |
