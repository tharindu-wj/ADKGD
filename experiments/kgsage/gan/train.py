"""Train the KGSAGE conditional GAN (encoder + generator + discriminator) and
save a checkpoint.

Run from the repo root:

  # Local quick-test on the bundled dummy KG (~minutes on CPU)
  PYTHONPATH=experiments python -m kgsage.cli.train_gan \
      --data data/dummy_kg \
      --epochs 50 \
      --device cpu \
      --out experiments/kgsage/outputs/checkpoints/kgsage_dummy.pt

  # HPC on FB15K-237 (V100)
  sbatch experiments/kgsage/slurm/train_gan_fb15k237.slurm

WHAT TRAINS (B1a)
  Three modules are trained together:
    - KGSAGEEncoder      : RGCN over the train graph -> context table E'
    - KGSAGEGenerator    : conditioned on E' -> corrupted-triple logits
    - KGSAGEDiscriminator : scores (real triple, candidate triple) pairs
  The encoder shares the generator's optimizer, so gradients that reach E'
  (generator -> E' -> encoder) train the encoder JOINTLY with the generator.

HOW ONE EPOCH WORKS
  1. For every real triple, build a "training pair" (real, target):
       real   = the actual triple
       target = the real triple with ONE slot corrupted.
     By default (--target_mode contradiction) the target is a TYPE-VALID but
     CONTEXT-DISTANT filler (kgsage.gan.targets.ContradictionTargetSampler), so
     the generator is trained to reach for the converging-context near-miss —
     this is what makes conditioning on E' actually matter. --target_mode random
     restores the simple random single-slot corruption as an ablation arm (E'
     stays inert because the target no longer depends on it).

  2. For each batch, recompute the context table E' once, then alternate:
     Discriminator step (E' detached):
       - Show D (real, target) with label 1 and (real, generator-candidate) with
         label 0. D learns to tell real-ish from fake.
     Generator + encoder step (E' grad-enabled):
       - Adversarial loss: G wants D to call its candidate real.
       - Reconstruction loss: G's logits should match the target distribution.
       - Total = adversarial + recon_weight * reconstruction. Backprop updates
         the generator AND the encoder (via E').

THE CHECKPOINT bundles the trained generator weights, the vocab maps, the set of
real triples (collision filter), and — the B1a addition — the CACHED context
table E'. Inference (kgsage.inference) replays E' and never touches PyG.
"""
import argparse
import os
import random
import time
from datetime import datetime

import torch
import torch.nn.functional as F

from kgsage.data.loaders import load_kg
from kgsage.gan.encoder import KGSAGEEncoder
from kgsage.gan.models import (
    KGSAGEGenerator, KGSAGEDiscriminator, gumbel_softmax, soft_embedding,
)
from kgsage.gan.targets import ContradictionTargetSampler


def random_corrupt(head, relation, tail, n_ent, n_rel, rng):
    """Replace ONE slot of (head, relation, tail) with a random in-vocab value.

    This is the simplest possible "negative" training signal: pick a slot
    uniformly, swap in a random entity (or relation). The generator's job is to
    learn the conditional distribution of plausible corruptions.
    """
    slot = rng.randint(0, 2)  # 0 = head, 1 = relation, 2 = tail
    if slot == 0:
        return (rng.randint(0, n_ent - 1), relation, tail)
    if slot == 1:
        return (head, rng.randint(0, n_rel - 1), tail)
    return (head, relation, rng.randint(0, n_ent - 1))


def train_one_epoch(encoder, generator, discriminator,
                    optimizer_generator, optimizer_discriminator,
                    real_all, target_all, edge_index, edge_type,
                    batch_size, device, recon_weight=10.0):
    """One pass over the training pairs. Returns (avg_D_loss, avg_G_loss).

    Each batch does ONE encoder forward to produce the context table E'. That
    tensor is:
      - DETACHED for the discriminator step (so only D updates there), and
      - reused GRAD-ENABLED for the generator step, so gradients flow
        generator -> E' -> encoder and the encoder trains jointly.

    `real_all` / `target_all` are pre-packed [N, 3] tensors; each epoch we just
    permute and slice — no Python lists, no per-batch tensor construction.
    """
    n_total = real_all.size(0)
    shuffle = torch.randperm(n_total, device=device)
    real_shuffled = real_all[shuffle]
    target_shuffled = target_all[shuffle]

    total_discriminator_loss = 0.0
    total_generator_loss = 0.0
    n_batches = 0

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        if end - start < 2:
            continue

        real_triples = real_shuffled[start:end]
        target_triples = target_shuffled[start:end]
        real_head, real_relation, real_tail = real_triples[:, 0], real_triples[:, 1], real_triples[:, 2]
        target_head, target_relation, target_tail = target_triples[:, 0], target_triples[:, 1], target_triples[:, 2]
        batch = end - start

        # Context table E' for THIS step (grad-enabled). We detach a copy for the
        # discriminator step and reuse this grad-enabled tensor for the generator
        # step, so the RGCN runs only ONCE per batch.
        entity_context = encoder(edge_index, edge_type)

        # ---------------- Discriminator step (E' detached) ----------------
        with torch.no_grad():
            context_detached = entity_context.detach()
            noise = torch.randn(batch, generator.z_dim, device=device)
            head_logits, relation_logits, tail_logits = generator(
                real_head, real_relation, real_tail, noise, context_detached,
            )
            # Gumbel-Softmax: near-one-hot samples we can embed via E'.
            soft_head = gumbel_softmax(head_logits)
            soft_relation = gumbel_softmax(relation_logits)
            soft_tail = gumbel_softmax(tail_logits)
            candidate_embedding = soft_embedding(
                soft_head, soft_relation, soft_tail,
                context_detached, generator.relation_embedding.weight,
            )
            real_embedding = torch.stack(list(generator.gather_conditioning_embeddings(
                real_head, real_relation, real_tail, context_detached)), dim=1)
            target_embedding = torch.stack(list(generator.gather_conditioning_embeddings(
                target_head, target_relation, target_tail, context_detached)), dim=1)

        optimizer_discriminator.zero_grad()
        # D should score (real, target) high (label 1) and (real, candidate) low (0).
        score_target = discriminator(real_embedding, target_embedding)
        score_candidate = discriminator(real_embedding, candidate_embedding)
        loss_discriminator = (
            F.binary_cross_entropy_with_logits(score_target, torch.ones_like(score_target))
            + F.binary_cross_entropy_with_logits(score_candidate, torch.zeros_like(score_candidate))
        )
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # ------------- Generator + encoder step (E' grad-enabled) -------------
        optimizer_generator.zero_grad()
        noise = torch.randn(batch, generator.z_dim, device=device)
        head_logits, relation_logits, tail_logits = generator(
            real_head, real_relation, real_tail, noise, entity_context,
        )
        soft_head = gumbel_softmax(head_logits)
        soft_relation = gumbel_softmax(relation_logits)
        soft_tail = gumbel_softmax(tail_logits)
        candidate_embedding = soft_embedding(
            soft_head, soft_relation, soft_tail,
            entity_context, generator.relation_embedding.weight,
        )
        real_embedding = torch.stack(list(generator.gather_conditioning_embeddings(
            real_head, real_relation, real_tail, entity_context)), dim=1)

        # Adversarial loss: G wants D to think the candidate is REAL (label 1).
        score_candidate_for_generator = discriminator(real_embedding, candidate_embedding)
        loss_adversarial = F.binary_cross_entropy_with_logits(
            score_candidate_for_generator, torch.ones_like(score_candidate_for_generator),
        )
        # Reconstruction loss: G's logits should match the target distribution.
        # A strong supervised signal that stabilises training.
        loss_reconstruction = (
            F.cross_entropy(head_logits, target_head)
            + F.cross_entropy(relation_logits, target_relation)
            + F.cross_entropy(tail_logits, target_tail)
        )
        loss_generator = loss_adversarial + recon_weight * loss_reconstruction
        loss_generator.backward()
        optimizer_generator.step()

        total_discriminator_loss += loss_discriminator.item()
        total_generator_loss += loss_generator.item()
        n_batches += 1

    return (total_discriminator_loss / max(n_batches, 1),
            total_generator_loss / max(n_batches, 1))


def save_checkpoint(generator, encoder, kg, dim, z_dim, edge_index, edge_type, save_path):
    """Bundle everything kgsage.inference.load_checkpoint needs into one .pt file.

    The B1a addition is `context_embeddings` (E'): the FINAL encoder output,
    cached so inference can look up E'[head]/E'[tail] without ever running the
    RGCN (or importing torch_geometric) again.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Freeze the trained context table for inference.
    context_embeddings = encoder.cache_embeddings(edge_index, edge_type)  # [n_ent, dim], CPU
    torch.save({
        "generator_state": generator.state_dict(),
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
        "hidden": generator.hidden,
        # --- B1a: cached RGCN context table (what the whole encoder is for) ---
        "context_embeddings": context_embeddings,
        "encoder_num_bases": encoder.num_bases,
        "encoder_layers": encoder.num_layers,
        "encoder_add_inverse": encoder.add_inverse,
    }, save_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Dataset directory")
    ap.add_argument("--out", required=True, help="Output checkpoint path (.pt)")
    ap.add_argument("--epochs", type=int, default=50)
    # 256 is a good default on both CPU and a single V100. Bigger batches
    # (512/1024) are faster on GPU; smaller (32/64) on a laptop CPU may help
    # the model learn finer distinctions but cost wall time.
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--z_dim", type=int, default=16)
    ap.add_argument("--recon_weight", type=float, default=10.0)
    # --- RGCN encoder hyper-parameters (B1a) ---
    ap.add_argument("--num_bases", type=int, default=30,
                    help="RGCN basis-decomposition rank (capped at 2*n_rel)")
    ap.add_argument("--encoder_layers", type=int, default=2,
                    help="Number of FastRGCNConv layers (hops of context)")
    ap.add_argument("--no_inverse", action="store_true",
                    help="Do NOT add inverse edges to the message-passing graph")
    # --- contradiction-bias target selection (B1a) ---
    ap.add_argument("--target_mode", choices=["contradiction", "random"],
                    default="contradiction",
                    help="contradiction = type-valid, context-distant targets (makes E' "
                         "matter); random = simple random single-slot corruption (ablation)")
    ap.add_argument("--k_candidates", type=int, default=20,
                    help="Type-valid fillers considered per triple (contradiction mode)")
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
    # The GAN trains on all available triples (train + valid + test), not the
    # train split alone. kgsage.data.loaders returns splits separately; combine
    # them here. triple_set_all is the matching set form.
    all_triples = list(kg["triples_train"]) + list(kg["triples_valid"]) + list(kg["triples_test"])
    kg["triples"] = all_triples
    kg["triple_set"] = kg["triple_set_all"]
    print(f"  entities = {kg['n_ent']:,}  relations = {kg['n_rel']:,}  triples = {len(all_triples):,}", flush=True)

    # The encoder's message-passing graph is the TRAIN graph (load_kg builds
    # edge_index over train only). This is the standard no-leakage choice: the
    # generator is conditioned on context derived from known training structure.
    edge_index, edge_type = KGSAGEEncoder.to_tensors(kg["edge_index"], kg["edge_type"], device)
    print(f"  message-passing edges = {edge_index.size(1):,} (train graph)", flush=True)

    # Pack the real triples into one [N, 3] tensor (no Python lists in the epoch
    # loop from here on — only tensor indexing).
    real_all = torch.tensor(all_triples, dtype=torch.long, device=device)

    # Target selection. `contradiction` (default) rebuilds context-distant,
    # type-valid targets each epoch from the CURRENT encoder — this is what makes
    # E' earn its place. `random` fixes simple random-corrupt targets once (the
    # ablation arm where E' stays inert because the target ignores it).
    if args.target_mode == "contradiction":
        target_sampler = ContradictionTargetSampler(
            real_all, kg["n_ent"], kg["n_rel"],
            k_candidates=args.k_candidates, seed=args.seed, device=device)
        fixed_targets = None
        print(f"Targets: contradiction bias (type-valid, context-distant), "
              f"k_candidates = {args.k_candidates}", flush=True)
    else:
        target_sampler = None
        fixed_targets = torch.tensor(
            [random_corrupt(h, r, t, kg["n_ent"], kg["n_rel"], rng) for (h, r, t) in all_triples],
            dtype=torch.long, device=device)
        print("Targets: random single-slot corruption (ablation)", flush=True)

    encoder = KGSAGEEncoder(
        kg["n_ent"], kg["n_rel"], dim=args.dim,
        num_bases=args.num_bases, num_layers=args.encoder_layers,
        add_inverse=not args.no_inverse,
    ).to(device)
    generator = KGSAGEGenerator(kg["n_ent"], kg["n_rel"], dim=args.dim, z_dim=args.z_dim).to(device)
    discriminator = KGSAGEDiscriminator(dim=args.dim).to(device)

    # The encoder shares the GENERATOR's optimizer, so it trains jointly with G.
    optimizer_generator = torch.optim.Adam(
        list(generator.parameters()) + list(encoder.parameters()),
        lr=args.lr, betas=(0.5, 0.999),
    )
    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr * 0.25, betas=(0.5, 0.999),
    )

    print(f"Encoder: FastRGCNConv x{args.encoder_layers}  "
          f"num_bases={min(args.num_bases, encoder.eff_rel)}  "
          f"inverse_edges={not args.no_inverse}  eff_relations={encoder.eff_rel}", flush=True)
    print(f"Training: {args.epochs} epochs, batch_size = {args.batch_size}, "
          f"recon_weight = {args.recon_weight}", flush=True)
    print("-" * 60, flush=True)

    # Wall-clock timing. datetime.now() for human-readable timestamps in the log;
    # time.perf_counter() for precise elapsed duration (immune to clock jumps).
    train_start_wall = datetime.now()
    train_start_perf = time.perf_counter()
    print(f"Training started at: {train_start_wall:%Y-%m-%d %H:%M:%S}", flush=True)

    for epoch in range(1, args.epochs + 1):
        # Contradiction mode: refresh targets against the CURRENT context table so
        # they track the improving encoder. Random mode: reuse the fixed targets.
        if target_sampler is not None:
            with torch.no_grad():
                context_snapshot = encoder(edge_index, edge_type)
            target_all = target_sampler.build_targets(context_snapshot)
        else:
            target_all = fixed_targets

        discriminator_loss, generator_loss = train_one_epoch(
            encoder, generator, discriminator,
            optimizer_generator, optimizer_discriminator,
            real_all, target_all, edge_index, edge_type,
            args.batch_size, device, args.recon_weight,
        )
        # Print every epoch for the first 5, then every 5% of total.
        log_every = max(1, args.epochs // 20)
        if epoch <= 5 or epoch % log_every == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:4d}/{args.epochs}  "
                  f"D_loss = {discriminator_loss:.4f}  G_loss = {generator_loss:.4f}", flush=True)

    train_end_wall = datetime.now()
    total_seconds = time.perf_counter() - train_start_perf
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Training finished at: {train_end_wall:%Y-%m-%d %H:%M:%S}", flush=True)
    print(f"Total training time : {total_seconds:.2f} s   "
          f"({hours}h {minutes}m {seconds}s)   "
          f"average {total_seconds / args.epochs:.2f} s/epoch", flush=True)
    print("-" * 60, flush=True)

    save_checkpoint(generator, encoder, kg, args.dim, args.z_dim, edge_index, edge_type, args.out)
    print(f"Saved checkpoint to {args.out}", flush=True)


if __name__ == "__main__":
    main()
