"""Train the simple GAN and save a checkpoint.

Run from the repo root:

  # Local quick-test on the bundled dummy KG (~minutes on CPU)
  python experiments/gan/train.py \
      --data data/dummy_kg \
      --epochs 50 \
      --device cpu \
      --out experiments/gan/outputs/checkpoints/dummy.pt

  # HPC on FB15K (V100)
  python experiments/gan/train.py \
      --data data/FB15K \
      --epochs 200 \
      --device cuda \
      --out experiments/gan/outputs/checkpoints/fb15k.pt

How training works (each epoch):
  1. For every real triple, build a "training pair" (real, target):
       real   = the actual triple
       target = the real triple with ONE slot replaced by a random in-vocab value
     The target is the kind of "wrong-but-plausible" output we want the
     generator to produce.

  2. For each batch of pairs, alternate two updates:

     Discriminator step:
       - Show D a real triple paired with its target (label = 1).
       - Show D a real triple paired with the generator's current output
         (label = 0).
       - D learns to tell them apart.

     Generator step:
       - Run G to get a candidate triple.
       - Use D's score on (real, candidate) as the "adversarial" loss
         (G wants D to think the candidate is real).
       - Also use cross-entropy between G's logits and the target as the
         "reconstruction" loss (G learns the shape of valid corruptions).
       - Total loss = adversarial + 10 * reconstruction.

The checkpoint at the end bundles:
  - the trained Generator weights
  - the vocab maps (ent2id, rel2id, ...)
  - the set of real triples (for collision filtering at inference)

So `generate.py` only needs the checkpoint, no separate data files.
"""
import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F

# Import sibling modules (data.py, model.py) without needing a package layout.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_kg                                            # noqa: E402
from model import Generator, Discriminator, gumbel_softmax, soft_embedding  # noqa: E402


def random_corrupt(h, r, t, n_ent, n_rel, rng):
    """Replace ONE slot of (h, r, t) with a random in-vocab value.

    This is the simplest possible "negative" training signal: pick a slot
    uniformly, swap in a random entity (or relation). The generator's job is
    to learn the conditional distribution of plausible corruptions.
    """
    slot = rng.randint(0, 2)  # 0 = head, 1 = relation, 2 = tail
    if slot == 0:
        return (rng.randint(0, n_ent - 1), r, t)
    if slot == 1:
        return (h, rng.randint(0, n_rel - 1), t)
    return (h, r, rng.randint(0, n_ent - 1))


def build_training_pairs(triples, n_ent, n_rel, rng):
    """For every real triple, generate one (real, target) pair to train on."""
    pairs = []
    for h, r, t in triples:
        target = random_corrupt(h, r, t, n_ent, n_rel, rng)
        pairs.append(((h, r, t), target))
    return pairs


def train_one_epoch(G, D, opt_G, opt_D, pairs, batch_size, device, recon_weight=10.0):
    """One pass over the training pairs. Returns (avg_D_loss, avg_G_loss)."""
    random.shuffle(pairs)
    total_d_loss = 0.0
    total_g_loss = 0.0
    n_batches = 0

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        if len(batch) < 2:
            continue

        # Pack the (real, target) pairs into tensors.
        real_triples = torch.tensor([p[0] for p in batch], dtype=torch.long, device=device)
        target_triples = torch.tensor([p[1] for p in batch], dtype=torch.long, device=device)
        real_h, real_r, real_t = real_triples[:, 0], real_triples[:, 1], real_triples[:, 2]
        target_h, target_r, target_t = target_triples[:, 0], target_triples[:, 1], target_triples[:, 2]
        n = len(batch)

        # ------------- Discriminator step -------------
        # Generate a "fake" candidate (no grad — D only updates D's weights).
        with torch.no_grad():
            z = torch.randn(n, G.z_dim, device=device)
            head_logits, rel_logits, tail_logits = G(real_h, real_r, real_t, z)
            # Gumbel-Softmax sample: near-one-hot, differentiable.
            soft_h = gumbel_softmax(head_logits)
            soft_r = gumbel_softmax(rel_logits)
            soft_t = gumbel_softmax(tail_logits)
            fake_emb = soft_embedding(
                soft_h, soft_r, soft_t,
                G.ent_emb.weight, G.rel_emb.weight,
            )
            # Get embeddings for real and target triples.
            real_emb = torch.stack(list(G.lookup(real_h, real_r, real_t)), dim=1)
            target_emb = torch.stack(list(G.lookup(target_h, target_r, target_t)), dim=1)

        opt_D.zero_grad()
        # D should give high score to (real, target) and low to (real, fake).
        score_target = D(real_emb, target_emb)
        score_fake = D(real_emb, fake_emb)
        loss_d = (
            F.binary_cross_entropy_with_logits(score_target, torch.ones_like(score_target))
            + F.binary_cross_entropy_with_logits(score_fake, torch.zeros_like(score_fake))
        )
        loss_d.backward()
        opt_D.step()

        # ------------- Generator step -------------
        opt_G.zero_grad()
        z = torch.randn(n, G.z_dim, device=device)
        head_logits, rel_logits, tail_logits = G(real_h, real_r, real_t, z)
        soft_h = gumbel_softmax(head_logits)
        soft_r = gumbel_softmax(rel_logits)
        soft_t = gumbel_softmax(tail_logits)
        fake_emb = soft_embedding(
            soft_h, soft_r, soft_t,
            G.ent_emb.weight, G.rel_emb.weight,
        )
        real_emb = torch.stack(list(G.lookup(real_h, real_r, real_t)), dim=1)

        # Adversarial loss: G wants D to think the candidate is REAL (label = 1).
        score_fake_for_g = D(real_emb, fake_emb)
        loss_adv = F.binary_cross_entropy_with_logits(
            score_fake_for_g, torch.ones_like(score_fake_for_g),
        )

        # Reconstruction loss: G's logits should match the target distribution.
        # This is a strong supervised signal that stabilises training.
        loss_recon = (
            F.cross_entropy(head_logits, target_h)
            + F.cross_entropy(rel_logits, target_r)
            + F.cross_entropy(tail_logits, target_t)
        )

        loss_g = loss_adv + recon_weight * loss_recon
        loss_g.backward()
        opt_G.step()

        total_d_loss += loss_d.item()
        total_g_loss += loss_g.item()
        n_batches += 1

    return total_d_loss / max(n_batches, 1), total_g_loss / max(n_batches, 1)


def save_checkpoint(G, kg, dim, z_dim, save_path):
    """Bundle everything `generate.py` needs into one .pt file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "generator_state": G.state_dict(),
        "ent2id": kg["ent2id"],
        "rel2id": kg["rel2id"],
        "id2ent": kg["id2ent"],
        "id2rel": kg["id2rel"],
        # Sets aren't pickle-friendly across versions; store as a list.
        "real_triples": list(kg["triple_set"]),
        "n_ent": kg["n_ent"],
        "n_rel": kg["n_rel"],
        "dim": dim,
        "z_dim": z_dim,
        "hidden": G.hidden,
    }, save_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Dataset directory")
    ap.add_argument("--out", required=True, help="Output checkpoint path (.pt)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--z_dim", type=int, default=16)
    ap.add_argument("--recon_weight", type=float, default=10.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}", flush=True)

    print(f"Loading KG from {args.data} ...", flush=True)
    kg = load_kg(args.data)
    print(f"  entities = {kg['n_ent']:,}  relations = {kg['n_rel']:,}  triples = {len(kg['triples']):,}", flush=True)

    print("Building (real, target) training pairs ...", flush=True)
    pairs = build_training_pairs(kg["triples"], kg["n_ent"], kg["n_rel"], rng)
    print(f"  pairs = {len(pairs):,}", flush=True)

    G = Generator(kg["n_ent"], kg["n_rel"], dim=args.dim, z_dim=args.z_dim).to(device)
    D = Discriminator(dim=args.dim).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr * 0.25, betas=(0.5, 0.999))

    print(f"Training: {args.epochs} epochs, batch_size = {args.batch_size}, recon_weight = {args.recon_weight}", flush=True)
    print("-" * 60, flush=True)
    for epoch in range(1, args.epochs + 1):
        d_loss, g_loss = train_one_epoch(
            G, D, opt_G, opt_D, pairs, args.batch_size, device, args.recon_weight,
        )
        # Print every epoch for the first 5, then every 5% of total.
        log_every = max(1, args.epochs // 20)
        if epoch <= 5 or epoch % log_every == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:4d}/{args.epochs}  D_loss = {d_loss:.4f}  G_loss = {g_loss:.4f}", flush=True)
    print("-" * 60, flush=True)

    save_checkpoint(G, kg, args.dim, args.z_dim, args.out)
    print(f"Saved checkpoint to {args.out}", flush=True)


if __name__ == "__main__":
    main()
