# KGSAGE — copy-paste command reference (GAN-only workflow)

Run these from the repo root on DeepThought (`cd ~/ADKGD` first). Each block
is self-contained — copy the whole block into your terminal, **top to bottom**.

**Train on the train split only.** Every training command below sets
`TRAIN_SPLIT=train`. Never use `TRAIN_SPLIT=all`: it feeds valid+test edges into
the RGCN encoder and its warm-up link-predictor — that is **test leakage** and
invalidates any reported result. (All splits *are* still used for the known-true
falseness filter, but that happens automatically in the code, not via this knob.)

Dataset naming gotcha: the two scripts spell the dataset differently.
- `train.slurm` wants the short tag: `fb15k237` / `wn18rr`
- `exp_cell.slurm` wants the data folder name: `FB15K-237` / `WN18RR`

---

## 0. One-time setup (login node)

```bash
cd ~/ADKGD
git pull                     # sync latest scripts (renamed train.slurm + kgsage_<tag>_s0.pt names)
PYTHONPATH=experiments ~/envs/adkgd/bin/python -m kgsage.cli.fetch_lp --dataset all
# expect: GATE PASS fb15k237 mrr≈0.3477 · wn18rr mrr≈0.4749
```

---

## 1. Train the GAN → checkpoint (GPU job, one per dataset)

### FB15K-237

```bash
DATASET=fb15k237 SEED=0 TRAIN_SPLIT=train sbatch experiments/kgsage/slurm/train.slurm
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
DATASET=wn18rr SEED=0 TRAIN_SPLIT=train sbatch experiments/kgsage/slurm/train.slurm
```

Produces `experiments/kgsage/outputs/checkpoints/kgsage_wn18rr_s0.pt`.

> **Known issue on WN18RR (from the 2026-07-03 run):** default settings
> mode-collapsed — `D-acc fake=1.00` from epoch 1, `fence-hit=0%`,
> `distinct-picks` fell from ~18,700 to ~110. If you see the same pattern,
> **do not use that checkpoint for gan cells.** Retrain with the entropy bonus
> raised (still train-only):
>
> ```bash
> DATASET=wn18rr SEED=0 TRAIN_SPLIT=train LAMBDA_H=0.05 sbatch experiments/kgsage/slurm/train.slurm
> ```
>
> `LAMBDA_H` is the only escalation knob wired into the SLURM script; the deeper
> levers are **not** env knobs — edit `experiments/kgsage/gan/train.py` to lower
> `--lr_d` or raise `--band_temp`. Note that raising `LAMBDA_H` only diversifies
> the *training* sampler: the deployed decode is near-argmax, so the §2 quality
> report can still show collapse — always re-check its coverage before use.
>
> ⚠️ If an earlier run left a `kgsage_wn18rr_all_s0.pt` (trained with
> `TRAIN_SPLIT=all`), **delete it** — it saw valid+test edges (leakage) and must
> not be reported. Retrain train-only with the command above.

### Second seed (needed later to de-circularize the "recovery" cell — optional for now)

```bash
DATASET=fb15k237 SEED=1 TRAIN_SPLIT=train sbatch experiments/kgsage/slurm/train.slurm
DATASET=wn18rr   SEED=1 TRAIN_SPLIT=train sbatch experiments/kgsage/slurm/train.slurm
```

---

## 2. Verify the checkpoint by hand (login node, CPU)

Run the quality report: it samples 5% of the graph, corrupts every triple
through the DEPLOYED decode path, and writes a readable `.md` (metrics +
before→after pairs) plus a `.tsv` of every pair for an LLM / by-hand
fact-check. `entity2text.txt` / `relation2text.txt` are auto-detected inside
`--data`, and the `reports/` dir is created automatically.

```bash
# FB15K-237
PYTHONPATH=experiments ~/envs/adkgd/bin/python -m kgsage.cli.quality_report \
  --ckpt experiments/kgsage/outputs/checkpoints/kgsage_fb15k237_s0.pt \
  --data data/FB15K-237 \
  --sample_frac 0.05 --out reports/quality_fb15k237.md

# WN18RR
PYTHONPATH=experiments ~/envs/adkgd/bin/python -m kgsage.cli.quality_report \
  --ckpt experiments/kgsage/outputs/checkpoints/kgsage_wn18rr_s0.pt \
  --data data/WN18RR \
  --sample_frac 0.05 --out reports/quality_wn18rr.md
```

Read the `.md` "Quality at a glance" table:

- **Type-valid** and **Truly false** near 100% — replacements are legal fillers
  for the relation and not real facts in any split.
- **Null** near 0% — the generator almost always found a valid corruption.
- **Distinct-entity coverage** + **Entropy** are the mode-collapse tell. LOW
  coverage with a few entities dominating "Most-repeated replacements" means the
  generator collapsed (the WN18RR failure) — **do not use that checkpoint for
  gan cells.**

Then feed the `.tsv` to an LLM (or skim it) to judge how many corruptions are
believable-but-false vs obviously wrong or accidentally true (a real-world-true
row is a false negative).

> Optional — hardness (LP gap): the quality report is model-free by design. For
> the `s_f(true) − s_f(neg)` gap (positive = false; large = easy, small = hard)
> or to corrupt your own hand-picked triples, use `kgsage.cli.inspect_gan_lp`
> or `kgsage.cli.corrupt` (both take `--lp_ckpt`/`--lp_ids`).

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

> Only run the WN18RR gan cells (②③④) once the WN18RR quality report above
> shows healthy distinct-entity coverage (not collapsed) — a collapsed generator
> makes ②③④ meaningless (it emits near-constant/degenerate corruptions).

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
| | `TRAIN_SPLIT` | `train` | **keep `train`** — `all` leaks valid+test edges into the encoder; never use for reported results |
| | `WARMUP_EPOCHS` | `10` | RGCN link-prediction warm-up |
| | `WARMSTART_EPOCHS` | `2` | band-teacher copy epochs |
| | `EPOCHS` | `30` | adversarial phase |
| | `BATCH_SIZE` | `512` | facts per step |
| | `TAU` | `0.5` | Gumbel temperature (training) |
| | `BAND_K` | `10` | band-teacher shortlist — smaller = harder fakes |
| | `FENCE_SIGMA` | `0.5` | truth-fence margin — a ceiling only, no floor |
| | `LAMBDA_H` | `0.01` | entropy bonus — raise to fight collapse (WN18RR run used `0.05`) |
| | `CKPT_PATH` | derived | override the output `.pt` path |
| `exp_cell.slurm` | `DATASET` | `FB15K-237` | `FB15K-237` \| `WN18RR` |
| | `SEED` | `0` | |
| | `MAX_EPOCH` | `1` | ADKGD detector epochs |
| | `ANOMALY_RATIO` | `0.05` | |
| | `NEG_SOURCE` | `random` | `random` \| `lp_band` \| `gan` |
| | `TEST_SOURCE` | `random` | `random` \| `lp_band` \| `gan` |
| | `GAN_CKPT` | — | required when either source is `gan` |

> **Not exposed as SLURM env knobs** — change these by editing the `argparse`
> defaults in `experiments/kgsage/gan/train.py`: `--band_temp` (0.5),
> `--lr_g` (1e-4), `--lr_d` (3e-4), `--dim` (64), `--z_dim` (16),
> `--lambda_fence` (1.0), `--lambda_res` (1e-3), `--label_smoothing` (0.1),
> `--beta_residual` (1.0), `--num_bases` (30), `--encoder_layers` (2),
> `--warmup_batch` (4096). These are mostly "size & stability" dials; the
> tuning levers for our two known problems are `LAMBDA_H` and `BAND_K` (both
> env knobs) plus, for the **deployment** collapse, a decode temperature that
> is currently hardcoded at `0.5` in `kgsage/inference.py`.
