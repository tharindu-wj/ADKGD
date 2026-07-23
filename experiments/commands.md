# KGSAGE — copy-paste command reference (current dual-discriminator flow)

Run from the repo root (`cd ~/ADKGD` on DeepThought; local Windows works the
same with the pytorch conda env). `PYTHONPATH=experiments` throughout.

## 0. One-time: LP auditor checkpoints (evaluation only — training needs no LP)

```bash
PYTHONPATH=experiments python -m kgsage.cli.fetch_lp --dataset all
# expect GATE PASS: fb15k237 MRR ~0.3477, wn18rr ~0.4749
```

## 1. Train (per-epoch snapshots; short runs on purpose)

```bash
# GPU node:
DATASET=fb15k237 SEED=0 sbatch experiments/kgsage/slurm/train.slurm
# CPU node (skips the RGCN warm-up by reusing a cached E'):
CPU_PARTITION=<name> DATASET=fb15k237 sbatch --partition=$CPU_PARTITION \
    experiments/kgsage/slurm/train_cpu.slurm
# Direct (no SLURM):
PYTHONPATH=experiments nohup python -m kgsage.gan.train \
    --data data/FB15K-237 \
    --out experiments/kgsage/outputs/checkpoints/run_fb15k237_s0.pt \
    --init_context_from experiments/kgsage/outputs/checkpoints/kgsage_fb15k237_s0.pt \
    --dmatch_epochs 2 --dreal_pretrain_epochs 2 --alpha_warmup_epochs 2 \
    --epochs 8 --snapshot_every 1 --device cpu > run_fb.log 2>&1 &
```

## 2. Select the snapshot (anchor-knockout; LOWEST mean J@10 wins)

```bash
for f in experiments/kgsage/outputs/checkpoints/run_fb15k237_s0.ep0*.pt; do
  echo "== $f"
  PYTHONPATH=experiments python experiments/kgsage/cli/knockout_eval.py \
      --ckpt "$f" --data data/FB15K-237 | tail -4
done
# WN18RR: add --relations _hypernym _derivationally_related_form _member_meronym _has_part
# Promote the winner:  cp <winner.pt> experiments/kgsage/outputs/checkpoints/generator_fb15k237.pt
```

Locked artifacts in use: `generator_fb15k237.pt` (knockout J@10 0.582) and
`generator_wn18rr.pt` (0.082).

## 3. Evaluate (Section 7 pipeline)

```bash
# Stage 1 — one CSV of corruptions (shared by 7.3 LLM + 7.4 ego):
PYTHONPATH=experiments python experiments/kgsage/cli/gen_corruptions_csv.py \
    --ckpt experiments/kgsage/outputs/checkpoints/generator_fb15k237.pt \
    --data data/FB15K-237 --split test --per_rel 4 --seed 7 \
    --out experiments/kgsage/outputs/eval/fb_corruptions.csv

# 7.4 — ego graphs from the CSV (best exemplars first):
PYTHONPATH=experiments python experiments/kgsage/cli/ego_from_csv.py \
    --csv experiments/kgsage/outputs/eval/fb_corruptions.csv \
    --data data/FB15K-237 --out_dir experiments/kgsage/outputs/eval/ego --limit 6

# Optional LP audit of generated negatives:
PYTHONPATH=experiments python -m kgsage.cli.inspect_gan_lp \
    --ckpt experiments/kgsage/outputs/checkpoints/generator_fb15k237.pt --n 40
```

## 4. Downstream 2x2 matrix (ADKGD)

```bash
NEG_SOURCE=<random|gan> TEST_SOURCE=<random|gan> DATASET=<FB15K-237|WN18RR> \
    SEED=0 GAN_CKPT=experiments/kgsage/outputs/checkpoints/generator_fb15k237.pt \
    sbatch experiments/slurm/exp_cell.slurm
python experiments/aggregate_results.py --dataset FB15K-237
```

## Smoke tests

```bash
PYTHONPATH=experiments python experiments/kgsage/smoke_test.py
PYTHONPATH=experiments python experiments/kgsage_bridge/smoke_test.py
```
