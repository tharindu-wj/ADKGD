# KGSAGE — Phase 1: Encoder pretraining

Phase 1 of the thesis pipeline ([THESIS_PLAN_pairgan_contradictions.md](../docs/THESIS_PLAN_pairgan_contradictions.md)). This module trains the **KGSAGE Encoder** — an RGCN backbone + DistMult decoder — to learn entity and relation embeddings on FB15K-237 that capture anti-symmetric predicate structure.

Phase 2 (KGSAGE Generator + Discriminator), Phase 3 (generation evaluation), and Phase 4 (ADKGD integration) consume the embeddings produced here.

## What this module does in one paragraph

We train two networks jointly on FB15K-237 link prediction:

- An **RGCN encoder** (relation-specific message passing) produces a 200-dim vector for every entity.
- A **DistMult decoder** scores triples as `<h, r, t>` — and as a side effect, learns a 200-dim vector for every relation.

After training, we save both embedding tables. Phase 2's KGSAGE Generator uses them to predict contradicting partner relations; Phase 2's KGSAGE Discriminator uses them to score pair plausibility.

## Files in this module

Read and run them in this order:

| Step | File | What it does | When to run |
|---|---|---|---|
| 1 | [data.py](data.py) | Loads FB15K-237 into PyG-friendly tensors. Library; nothing to run directly. | Imported by everything else. |
| 2 | [audit_density.py](audit_density.py) | **Test 1.3** — counts anti-symmetric predicate pairs in train.txt. Decision gate: needs ≥ 30 pairs to proceed. | First. Cheap, no training. |
| 3 | [encoder.py](encoder.py) | `KGSAGEEncoder` (RGCN) + `KGSAGEDistMultDecoder`. Library. | Imported by training + evaluation. |
| 4 | [train_encoder.py](train_encoder.py) | Trains the encoder + decoder on FB15K-237. Saves checkpoint. | After Test 1.3 passes. |
| 5 | [evaluate_lp.py](evaluate_lp.py) | **Test 1.1** — link prediction MRR on the test set. Pass = MRR ≥ 0.30. | After training. |
| 6 | [test_antisym_signal.py](test_antisym_signal.py) | **Test 1.2** — Mann-Whitney U on cosine similarities of symmetric vs anti-symmetric relation pairs. Pass = p < 0.05. | After training. |

## Quick start

```bash
# 0. Install dependencies (one-time)
pip install torch torch_geometric

# 1. Decision gate — does FB15K-237 even have the anti-symmetric signal we need?
python -m experiments.kgsage.audit_density --data data/FB15K-237

# 2. Train the encoder (one-time, ~30 min on a V100)
python -m experiments.kgsage.train_encoder \
    --data data/FB15K-237 \
    --out experiments/kgsage/outputs/fb15k237_encoder.pt \
    --epochs 200 \
    --dim 200

# 3. Test 1.1 — link prediction MRR
python -m experiments.kgsage.evaluate_lp \
    --data data/FB15K-237 \
    --ckpt experiments/kgsage/outputs/fb15k237_encoder.pt

# 4. Test 1.2 — anti-symmetric signal in relation embeddings
python -m experiments.kgsage.test_antisym_signal \
    --data data/FB15K-237 \
    --ckpt experiments/kgsage/outputs/fb15k237_encoder.pt \
    --audit experiments/kgsage/outputs/density_audit.json
```

## What gets saved

| Path | Content |
|---|---|
| `experiments/kgsage/outputs/density_audit.json` | Phase 1 Test 1.3 output: anti-symmetric and symmetric predicate pairs |
| `experiments/kgsage/outputs/fb15k237_encoder.pt` | Trained encoder + decoder weights, vocab maps. Consumed by Phase 2. |

## Decision gates

Each test has a pass/fail criterion documented in the thesis plan. If any test fails, **stop and pivot before architecting Phase 2** — the failure modes (data too sparse, model can't learn anti-symmetric signal) are exactly what Phase 1 was designed to detect cheaply.

| Test | Pass | Fail action |
|---|---|---|
| 1.3 (data audit) | ≥ 30 anti-symmetric pairs with support ≥ 100 | Add NELL-995 + WN18RR, or pivot to rule-mining thesis |
| 1.1 (link prediction MRR) | MRR ≥ 0.30 | Debug training. Check edge_type tensor shapes, vanishing gradients, lr too high. |
| 1.2 (anti-symmetric signal) | Mann-Whitney U test p < 0.05 | Try ConvE decoder, longer training, or escalate to CompGCN (Plan B, +2 weeks) |

## Why RGCN + DistMult (and not CompGCN)?

CompGCN ([Vashishth 2020](https://arxiv.org/abs/1911.03082)) is theoretically nicer — it jointly embeds nodes and relations. But it's not in PyTorch Geometric, and porting [malllabiisc/CompGCN](https://github.com/malllabiisc/CompGCN) from PyTorch 1.0 to modern PyG takes ~2 weeks.

RGCN ([Schlichtkrull 2018](https://arxiv.org/abs/1703.06103)) is in PyG natively as `torch_geometric.nn.RGCNConv`. Paired with DistMult decoder, we get both entity embeddings (from the encoder) and relation embeddings (from `decoder.rel_emb`) — same outputs CompGCN would give. CompGCN remains Plan B if Test 1.2 fails on RGCN+DistMult.

See [THESIS_PLAN_pairgan_contradictions.md §2.2](../docs/THESIS_PLAN_pairgan_contradictions.md) for the full encoder-choice rationale.

## Style note

This module is intentionally written in the same simple, comment-heavy style as `experiments/gan/`. Goal: a Master's student reading the code should be able to learn from it without needing prior PyG familiarity. If you find a comment redundant or unclear, send a note — the docstring is the spec.
