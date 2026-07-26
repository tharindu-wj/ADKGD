# KGSAGE — copy-paste command reference (dual-discriminator architecture)

Run from the repo root (`cd ~/ADKGD` on DeepThought; local Windows works the
same with the pytorch conda env). `PYTHONPATH=experiments` throughout.

Names follow `experiments/docs/KGSAGE_glossary.md`.

## 0. (Optional) Add YAGO 4.5 as a dataset

The model code is dataset-agnostic — YAGO is purely a data-conversion step.
**Download the `-tiny` release, NOT the 12 GB full one**: KGSAGE builds
O(n_ent) structures (the membership sketch is ~8 GB per million entities), so
the graph must be shrunk to FB/WN scale with the converter's subsampling knobs.

```bash
# 1. download the TINY Turtle release (~200 MB) into data/
cd ~/ADKGD/data
wget https://yago-knowledge.org/data/yago4.5/yago-4.5.0.2-tiny.zip
cd ~/ADKGD

# 2. convert Turtle -> data/YAGO4.5/{train,valid,test}.txt (pure stdlib, no GPU).
#    --min_degree + --max_entities shrink YAGO to a dense, WN18RR-scale KG.
python experiments/kgsage/data/yago_to_tsv.py \
    --in  data/yago-4.5.0.2-tiny.zip \
    --out data/YAGO4.5 \
    --min_degree 5 --max_entities 30000
#    Optional: focus on specific relations for a cleaner story, e.g.
#    --relations nationality birthPlace spouse memberOf author director \
#                containedInPlace deathPlace
```

The converter accepts a `.zip`, a `.ttl`, or a directory. Read its summary: aim
for tens of thousands of entities and a few hundred thousand train triples. If
too few survive, lower `--min_degree`; if too many, lower `--max_entities`.
No `entity2text.txt` is needed — YAGO ids are already human-readable.

## 1. Train — Phases 1-2 (per-epoch snapshots; short runs on purpose)

One job covers Phase 1 (Neighbourhood Context Encoding: RGCN warm-up -> E'
frozen, membership sketches built) and Phase 2 (Adversarial Generator
Training: discriminator pretraining, then the dual-discriminator game).

```bash
# GPU node:
DATASET=fb15k237 SEED=0 sbatch experiments/kgsage/slurm/train.slurm
# CPU node (skips the Phase-1 RGCN warm-up by reusing a cached E'):
CPU_PARTITION=<name> DATASET=fb15k237 sbatch --partition=$CPU_PARTITION \
    experiments/kgsage/slurm/train_cpu.slurm
# Direct (no SLURM):
PYTHONPATH=experiments nohup python -m kgsage.gan.train \
    --data data/FB15K-237 \
    --out experiments/kgsage/outputs/checkpoints/run_fb15k237_s0.pt \
    --init_context_from experiments/kgsage/outputs/checkpoints/kgsage_fb15k237_s0.pt \
    --epochs 8 --snapshot_every 1 --device cpu > run_fb.log 2>&1 &
```

## 2. Select the snapshot (anchor-knockout; LOWEST mean knockout J@10 wins)

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
`generator_wn18rr.pt` (0.082). Lower = more anchor-specific.

## 3. Evaluate (Section 7 pipeline)

```bash
# Phase 3 (Corruption Generation) — one CSV of corruptions, shared by the
# 7.3 LLM arm and the 7.4 ego arm:
PYTHONPATH=experiments python experiments/kgsage/cli/gen_corruptions_csv.py \
    --ckpt experiments/kgsage/outputs/checkpoints/generator_fb15k237.pt \
    --data data/FB15K-237 --split test --per_rel 4 --seed 7 \
    --out experiments/kgsage/outputs/eval/fb_corruptions.csv

# 7.4 — ego graphs from the CSV (best exemplars first):
PYTHONPATH=experiments python experiments/kgsage/cli/ego_from_csv.py \
    --csv experiments/kgsage/outputs/eval/fb_corruptions.csv \
    --data data/FB15K-237 --out_dir experiments/kgsage/outputs/eval/ego --limit 6
```

## 4. Downstream 2x2 matrix (ADKGD)

Detector-side vocabulary: the corruptions from step 3 arrive as **negatives**
(training) and **anomalies** (evaluation). `NEG_SOURCE`/`TEST_SOURCE`/`GAN_CKPT`
and the `random`|`gan` values are a frozen contract between `exp_cell.slurm`,
`run_experiment.py` and the detector — do not rename them.

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
