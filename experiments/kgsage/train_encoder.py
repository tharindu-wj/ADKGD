"""Train the KGSAGE Encoder + DistMult decoder on FB15K-237.

The training task is link prediction:
  Given (h, r, ?) — predict the correct tail entity
  Given (?, r, t) — predict the correct head entity

We do this with the standard margin-ranking loss against random negatives:

  For each true triple (h, r, t):
    sample a corrupted triple (h', r, t) by replacing the head
    or                       (h, r, t') by replacing the tail
    push score(h, r, t) above score(h', r, t)  by at least `margin`

This is the same training objective Bordes et al. 2013 used for TransE
and Schlichtkrull et al. 2018 used for R-GCN. We're not innovating here —
we're doing the standard pretraining recipe so that:
  - the encoder learns good entity embeddings
  - the decoder learns good relation embeddings as a side effect
  - Phase 2 inherits both via the saved checkpoint

After training we save the model with `save_pretrained()`. Phase 2's
KGSAGE Generator and Discriminator will load it with `load_pretrained()`.

Usage (Local, dummy KG, CPU):
    python -m experiments.kgsage.train_encoder \\
        --data data/dummy_kg \\
        --epochs 50 \\
        --device cpu \\
        --out experiments/kgsage/outputs/dummy_encoder.pt

Usage (HPC, FB15K-237, V100):
    python -m experiments.kgsage.train_encoder \\
        --data data/FB15K-237 \\
        --epochs 200 \\
        --device cuda \\
        --out experiments/kgsage/outputs/fb15k237_encoder.pt
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# So the package import works whether you invoke as `python -m experiments.kgsage.train_encoder`
# or `python experiments/kgsage/train_encoder.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.kgsage.data import load_fb15k237
from experiments.kgsage.encoder import KGSAGELinkPredictor


def sample_negatives(positive_triples, n_ent, triple_set, rng):
    """For each positive triple, build one negative by corrupting head OR tail.

    This matches the standard KGE negative-sampling recipe:
      - flip a coin: corrupt head or tail
      - pick a uniform-random entity
      - reject if the resulting triple is actually in the real graph

    Inputs:
      positive_triples : (batch, 3) tensor of (h, r, t) IDs
      n_ent            : entity vocab size
      triple_set       : Python set of all (h, r, t) tuples for fast collision check
      rng              : random.Random instance, for reproducibility

    Returns: (batch, 3) tensor of corrupted triples.
    """
    n_positives = positive_triples.size(0)
    negatives = positive_triples.clone()

    for i in range(n_positives):
        h, r, t = int(negatives[i, 0]), int(negatives[i, 1]), int(negatives[i, 2])

        # ─── Coin flip: corrupt head (0) or tail (1) ──────────────────
        for _ in range(10):  # retry budget
            if rng.random() < 0.5:
                # Corrupt head: pick a different random entity.
                new_h = rng.randrange(n_ent)
                candidate = (new_h, r, t)
                slot = 0
                new_value = new_h
            else:
                # Corrupt tail.
                new_t = rng.randrange(n_ent)
                candidate = (h, r, new_t)
                slot = 2
                new_value = new_t

            # Reject if we accidentally hit a real triple — that would
            # send the wrong gradient signal (penalising a true fact).
            if candidate not in triple_set:
                negatives[i, slot] = new_value
                break
        # If we run out of retries we leave the (silently invalid)
        # last candidate; with n_ent = 14,541 this happens with
        # probability ~1e-5 so it's not worth a louder fallback.

    return negatives


def evaluate_mrr_quick(model, data, kg, device, n_eval=1000):
    """Cheap validation: random sample of `n_eval` valid triples → MRR.

    For each held-out triple (h, r, t):
      - replace t with EVERY entity in vocab
      - score all candidates
      - rank the true t
      - 1 / rank is the reciprocal rank for this triple

    MRR is the mean of those reciprocal ranks. Higher = better.

    We score n_eval triples instead of the full valid set so this runs
    in seconds during training. Final evaluation (Test 1.1) uses the
    full test set in evaluate_lp.py.
    """
    model.eval()
    valid_triples = kg["triples_valid"]
    n_ent = kg["n_ent"]
    edge_index = data["edge_index"]
    edge_type = data["edge_type"]

    if n_eval < len(valid_triples):
        sample = random.sample(valid_triples, n_eval)
    else:
        sample = valid_triples

    # Pre-encode the whole KG once — entity embeddings don't change
    # while we score this many candidates.
    with torch.no_grad():
        ent_emb_all = model.encoder(edge_index, edge_type)

    ranks = []
    with torch.no_grad():
        for h, r, t in sample:
            # Score (h, r, *) for every possible tail.
            h_emb = ent_emb_all[h].unsqueeze(0).expand(n_ent, -1)
            r_emb = model.decoder.rel_emb.weight[r].unsqueeze(0).expand(n_ent, -1)
            scores = (h_emb * r_emb * ent_emb_all).sum(dim=-1)

            # Rank of the true tail = 1 + number of entities scoring higher.
            true_score = scores[t]
            rank = 1 + (scores > true_score).sum().item()
            ranks.append(1.0 / rank)

    model.train()
    return sum(ranks) / len(ranks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/FB15K-237",
                    help="dataset folder under data/")
    ap.add_argument("--out", default="experiments/kgsage/outputs/fb15k237_encoder.pt",
                    help="where to save the trained checkpoint")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--dim", type=int, default=200)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--num_bases", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=10,
                    help="run quick validation MRR every N epochs")
    args = ap.parse_args()

    # Reproducibility ─────────────────────────────────────────────────
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    print(f"Loading {args.data}/...", flush=True)
    kg = load_fb15k237(args.data)
    print(f"  {kg['n_ent']:,} entities, {kg['n_rel']:,} relations", flush=True)
    print(f"  train={len(kg['triples_train']):,}  "
          f"valid={len(kg['triples_valid']):,}  "
          f"test={len(kg['triples_test']):,}", flush=True)
    print()

    device = torch.device(args.device)

    # Move the KG to the device once. The encoder reads these every step.
    data = {
        "edge_index": kg["edge_index"].to(device),
        "edge_type": kg["edge_type"].to(device),
    }
    all_train_triples = torch.tensor(kg["triples_train"], dtype=torch.long, device=device)

    # ─── Build the model ──────────────────────────────────────────────
    model = KGSAGELinkPredictor(
        n_ent=kg["n_ent"],
        n_rel=kg["n_rel"],
        dim=args.dim,
        n_layers=args.n_layers,
        num_bases=args.num_bases,
    ).to(device)
    print(f"Model parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}", flush=True)
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ─── Training loop ────────────────────────────────────────────────
    n_train = all_train_triples.size(0)
    best_mrr = 0.0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"Training for {args.epochs} epochs, batch_size={args.batch_size}", flush=True)
    print("─" * 70)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        # Shuffle the triples once per epoch.
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, n_train, args.batch_size):
            batch_idx = perm[batch_start:batch_start + args.batch_size]
            positives = all_train_triples[batch_idx]

            # Build one negative per positive.
            # We move negatives to CPU for sampling because the rejection
            # check is a Python set lookup — fastest in pure Python.
            positives_cpu = positives.cpu()
            negatives = sample_negatives(
                positives_cpu,
                kg["n_ent"],
                kg["triple_set_train"],
                rng,
            ).to(device)

            # Score positives and negatives in the same forward pass.
            # The encoder runs once per batch (across the whole KG); the
            # decoder picks up the specific (h, r, t) we care about.
            pos_scores = model(data["edge_index"], data["edge_type"], positives)
            neg_scores = model(data["edge_index"], data["edge_type"], negatives)

            # Margin ranking loss: push pos_score above neg_score by
            # at least `margin`. `relu` clamps the negative-margin case
            # to zero — once a pair is well-separated we stop training on it.
            loss = F.relu(args.margin - pos_scores + neg_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        # ─── Periodic validation MRR ───────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            mrr = evaluate_mrr_quick(model, data, kg, device, n_eval=1000)
            print(f"  epoch {epoch:>4}  loss={epoch_loss:.4f}  "
                  f"valid_MRR={mrr:.4f}  time={epoch_time:.1f}s",
                  flush=True)

            # Save the best checkpoint by validation MRR.
            if mrr > best_mrr:
                best_mrr = mrr
                model.save_pretrained(args.out, kg["ent2id"], kg["rel2id"])
                print(f"    → new best, saved to {args.out}", flush=True)
        else:
            # Quiet epochs: just print the loss and time.
            print(f"  epoch {epoch:>4}  loss={epoch_loss:.4f}  "
                  f"time={epoch_time:.1f}s", flush=True)

    print("─" * 70)
    print(f"Training done. Best validation MRR: {best_mrr:.4f}")
    print(f"Checkpoint saved at: {args.out}")
    print()
    print("Next steps:")
    print("  Test 1.1 (link prediction MRR):")
    print(f"    python -m experiments.kgsage.evaluate_lp --ckpt {args.out}")
    print("  Test 1.2 (anti-symmetric signal):")
    print(f"    python -m experiments.kgsage.test_antisym_signal --ckpt {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
