"""KGSAGE trainer: the dual-discriminator adversarial game (paper: Methodology).

The three phases of the method, in this one file:

  1. Neighbourhood Context Encoding — the RGCN encoder is warmed up with a
     DistMult decoder on link prediction, then its output E' (one context
     vector per entity) is FROZEN. Bloom membership sketches of every
     entity's 1-2 hop neighbourhood are built alongside.
  2. Adversarial Generator Training — the candidate-scoring generator plays
     against two discriminators over the frozen E':
       plausibility discriminator      "could this triple be real?"   G pushes HIGH
       neighbourhood discriminator  "does the filler fit THIS
                                   anchor's neighbourhood?"       G pushes LOW
     Generator loss:  L_G = -D_real + alpha * relu(D_match - margin).
     alpha is set by a PI controller so that the fraction of generated picks
     the training graph actually corroborates stays at CORROBORATION_TARGET.
  3. Corruption Generation lives in kgsage/corruption_generation.py — it
     replays the checkpoint saved here.

Everything learned (E', sketches, candidate pools, neighbour lists, both
discriminators) sees the TRAIN split only. The guarantee that emitted
corruptions are false against ALL splits comes from the masks applied at
corruption time, not from anything learned here.

The locked hyperparameters live as module-level constants below (edit there
to re-tune); only operational flags are on the CLI.

Usage (repo root, pytorch env):
  PYTHONPATH=experiments python -m kgsage.gan.train \
      --data data/FB15K-237 --out experiments/kgsage/outputs/checkpoints/run.pt \
      --epochs 8 --snapshot_every 1 [--init_context_from <ckpt> --device cpu]
"""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from kgsage.data.loaders import load_kg, build_edge_index
from kgsage.gan.neighbourhood_context_encoder import NeighbourhoodContextEncoder
from kgsage.gan.generator import CandidateScoringGenerator, gumbel_softmax
from kgsage.gan.plausibility_discriminator import PlausibilityDiscriminator
from kgsage.gan.neighbourhood_discriminator import NeighbourhoodDiscriminator
from kgsage.gan.membership_sketch import build_membership_sketches
from kgsage.gan.candidate_sampler import CandidateSampler

HEAD, TAIL = 0, 2

# ==========================================================================
# Locked configuration. Tuned once during development and fixed for every
# run; only operational args (data/out/device/seed/epochs/snapshots/E'-reuse)
# stay on the CLI. Re-tune by editing here.
# ==========================================================================

# -- architecture (E', sketches, candidate set) --
EMBEDDING_DIM             = 64      # context-vector / embedding width
RGCN_NUM_BASES       = 30      # RGCN basis decomposition
RGCN_NUM_LAYERS  = 2       # RGCN depth -> 2-hop context
SKETCH_BITS     = 8192    # Bloom membership-sketch length
NUM_NEIGHBOURS_SAMPLED           = 32      # neighbours the neighbourhood discriminator attends over
NUM_CANDIDATES          = 256     # candidates scored per triple (decode = full pool)

# -- E' warm-up (skipped when --init_context_from is given) --
RGCN_WARMUP_EPOCHS   = 10
RGCN_WARMUP_BATCH_SIZE    = 4096

# -- curriculum: pretrain both discriminators, then an alpha=0 warm-up --
NEIGHBOURHOOD_PRETRAIN_EPOCHS         = 2
PLAUSIBILITY_PRETRAIN_EPOCHS = 2
PLAUSIBILITY_ONLY_EPOCHS   = 2

# -- game optimisation --
BATCH_SIZE      = 256
GUMBEL_TEMPERATURE             = 0.5     # Gumbel-Softmax temperature (train + decode)
GENERATOR_LEARNING_RATE            = 1e-4
PLAUSIBILITY_LEARNING_RATE            = 3e-4    # plausibility discriminator
NEIGHBOURHOOD_LEARNING_RATE       = 1e-4    # neighbourhood discriminator's online updates during
                          # the game (a frozen one gets exploited by G)
LABEL_SMOOTHING = 0.1

# -- contradiction-pressure controller (alpha) --
CORROBORATION_TARGET    = 0.13    # PI set-point; MUST stay above the ~0.12-0.13
                          # structural floor or alpha saturates and the
                          # generator collapses to one-alien-fits-all picks
ALPHA_INITIAL      = 1.0     # value alpha restarts at when the warm-up ends
ALPHA_MAX       = 10.0    # anti-windup clamp
PI_PROPORTIONAL_GAIN        = 2.0     # PI proportional gain
PI_INTEGRAL_GAIN        = 0.2     # PI integral gain
CONTRADICTION_MARGIN    = 0.0     # hinge margin: alpha * relu(D_match - margin)
WRONG_ANCHOR_LOSS_WEIGHT  = 1.0     # weight of the wrong-anchor class (GAN-CLS style);
                          # this is what makes D_real anchor-conditional


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=8,
                        help="adversarial game epochs (locked recipe: 8; use 2 "
                             "for a quick smoke). Anchor-awareness peaks a few "
                             "epochs after the alpha ramp and then erodes, so "
                             "keep runs short and select across snapshots by "
                             "knockout J@10.")
    parser.add_argument("--snapshot_every", type=int, default=0,
                        help="save a full loadable checkpoint every N game "
                             "epochs once the alpha ramp starts (0 = off). "
                             "Select the reported generator across snapshots "
                             "by knockout J@10.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--init_context_from", default=None,
                        help="load a frozen E' from an existing checkpoint "
                             "instead of running the RGCN warm-up (CPU path; "
                             "vocabularies must match)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    python_rng = random.Random(args.seed)
    device = torch.device(args.device
                          or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)

    kg = load_kg(args.data)
    train_triples = list(kg["triples_train"])
    n_ent, n_rel = kg["n_ent"], kg["n_rel"]
    train_triples_tensor = torch.tensor(train_triples, dtype=torch.long)
    print(f"KG: {n_ent:,} entities, {n_rel:,} relations, "
          f"{len(train_triples):,} train triples", flush=True)

    # ---------------------------------------------------------------------
    # Phase 1: Neighbourhood Context Encoding.
    # Either reuse a frozen E' from an earlier checkpoint, or warm the RGCN
    # encoder up with a DistMult decoder on link prediction and freeze its
    # output. The decoder is a throwaway: it only exists to give the encoder
    # a training signal, and is deleted once E' is frozen.
    # ---------------------------------------------------------------------
    if args.init_context_from:
        donor = torch.load(args.init_context_from, map_location=device,
                           weights_only=False)
        assert donor["n_ent"] == n_ent and donor["dim"] == EMBEDDING_DIM, \
            "context table from --init_context_from does not match this KG"
        # The vocabulary must match too (string-keyed check on a sample).
        for entity in list(kg["ent2id"])[:50]:
            assert donor["ent2id"].get(entity) == kg["ent2id"][entity], \
                f"vocab mismatch at {entity!r}"
        context_table = donor["context_embeddings"].to(device).detach()
        torch_rng = torch.Generator().manual_seed(args.seed + 1)
        print(f"E' loaded from {args.init_context_from} "
              f"[{context_table.shape[0]}, {context_table.shape[1]}]", flush=True)
        skip_warmup = True
    else:
        skip_warmup = False
    # The encoder is constructed in BOTH branches (and discarded when E' is
    # reused) so that torch's global RNG advances identically either way —
    # keeping every later weight init reproducible per seed.
    edge_index, edge_type = NeighbourhoodContextEncoder.to_tensors(
        kg["edge_index"], kg["edge_type"], device)
    context_encoder = NeighbourhoodContextEncoder(
        n_ent, n_rel, dim=EMBEDDING_DIM, num_bases=RGCN_NUM_BASES,
        num_layers=RGCN_NUM_LAYERS).to(device)
    if skip_warmup:
        context_encoder = None
    if not skip_warmup:
        # DistMult decoder: score(h, r, t) = sum(E'[h] * w_r * E'[t]).
        # Real triples should score high, random-tail triples low — the
        # standard link-prediction warm-up that shapes E' into a meaningful
        # neighbourhood summary.
        distmult_relations = torch.nn.Parameter(
            torch.randn(n_rel, EMBEDDING_DIM, device=device) * 0.1)
        warmup_optimizer = torch.optim.Adam(
            list(context_encoder.parameters()) + [distmult_relations], lr=1e-3)
        torch_rng = torch.Generator().manual_seed(args.seed + 1)
        for epoch in range(1, RGCN_WARMUP_EPOCHS + 1):
            shuffled = torch.randperm(train_triples_tensor.shape[0],
                                      generator=torch_rng)
            total_loss = num_batches = 0
            for start in range(0, len(shuffled), RGCN_WARMUP_BATCH_SIZE):
                rows = train_triples_tensor[
                    shuffled[start:start + RGCN_WARMUP_BATCH_SIZE]].to(device)
                h, r, t = rows[:, 0], rows[:, 1], rows[:, 2]
                random_tails = torch.randint(0, n_ent, (len(rows),),
                                             generator=torch_rng).to(device)
                current_context = context_encoder(edge_index, edge_type)
                positive_scores = (current_context[h] * distmult_relations[r]
                                   * current_context[t]).sum(-1)
                negative_scores = (current_context[h] * distmult_relations[r]
                                   * current_context[random_tails]).sum(-1)
                loss = (F.binary_cross_entropy_with_logits(
                            positive_scores, torch.ones_like(positive_scores))
                        + F.binary_cross_entropy_with_logits(
                            negative_scores, torch.zeros_like(negative_scores)))
                warmup_optimizer.zero_grad(); loss.backward(); warmup_optimizer.step()
                total_loss += loss.item(); num_batches += 1
            print(f"  warmup {epoch}/{RGCN_WARMUP_EPOCHS} "
                  f"LP_loss={total_loss/max(num_batches,1):.4f}", flush=True)
        context_encoder.eval(); context_encoder.requires_grad_(False)
        with torch.no_grad():
            context_table = context_encoder(edge_index, edge_type).detach()
        del distmult_relations, warmup_optimizer
        print(f"E' frozen [{context_table.shape[0]}, {context_table.shape[1]}]",
              flush=True)

    # ---------------------------------------------------------------------
    # Train-split lookup structures (learned components see TRAIN only).
    # ---------------------------------------------------------------------
    neighbour_sets: dict[int, set] = {}        # undirected 1-hop adjacency
    true_tails: dict[tuple, set] = defaultdict(set)   # (h, r) -> known tails
    true_heads: dict[tuple, set] = defaultdict(set)   # (r, t) -> known heads
    triples_by_relation: dict[int, list] = defaultdict(list)
    for h, r, t in train_triples:
        neighbour_sets.setdefault(h, set()).add(t)
        neighbour_sets.setdefault(t, set()).add(h)
        true_tails[(h, r)].add(t)
        true_heads[(r, t)].add(h)
        triples_by_relation[r].append((h, r, t))

    print("building sketches (train split)...", flush=True)
    membership_sketches = build_membership_sketches(
        train_triples, n_ent, m=SKETCH_BITS, seed=args.seed).float()
    candidate_sampler = CandidateSampler(train_triples, n_ent, n_rel,
                                         k=NUM_CANDIDATES, seed=args.seed)
    # Type pools for the checkpoint: which entities were observed in each
    # (slot, relation) position. Corruption generation scores the FULL pool.
    pool_masks = torch.zeros(2, n_rel, n_ent, dtype=torch.bool)
    for h, r, t in train_triples:
        pool_masks[0, r, h] = True
        pool_masks[1, r, t] = True

    def sample_neighbour_batch(anchors, exclude=None):
        """Sample up to NUM_NEIGHBOURS_SAMPLED neighbours per anchor from TRAIN adjacency.

        Returns (neighbour_ids [B, NUM_NEIGHBOURS_SAMPLED], neighbour_mask [B, NUM_NEIGHBOURS_SAMPLED]); the
        mask marks which positions hold a real neighbour (rows are padded).
        """
        batch_size = len(anchors)
        neighbour_ids = torch.zeros(batch_size, NUM_NEIGHBOURS_SAMPLED, dtype=torch.long)
        neighbour_mask = torch.zeros(batch_size, NUM_NEIGHBOURS_SAMPLED, dtype=torch.bool)
        for i, anchor in enumerate(anchors):
            neighbours = neighbour_sets.get(int(anchor), ())
            neighbours = [n for n in neighbours
                          if exclude is None or n != int(exclude[i])]
            if not neighbours:
                continue
            if len(neighbours) > NUM_NEIGHBOURS_SAMPLED:
                neighbours = python_rng.sample(neighbours, NUM_NEIGHBOURS_SAMPLED)
            neighbour_ids[i, :len(neighbours)] = torch.tensor(neighbours)
            neighbour_mask[i, :len(neighbours)] = True
        return neighbour_ids, neighbour_mask

    _support_row_cache: dict[int, torch.Tensor] = {}

    def supported_entities_row(entity: int) -> torch.Tensor:
        """Cached bool [n_ent] row: which entities the graph CORROBORATES for
        this entity — its direct neighbours plus anything within two hops
        (TRAIN adjacency)."""
        row = _support_row_cache.get(entity)
        if row is None:
            neighbours = neighbour_sets.get(entity, set())
            row = torch.zeros(n_ent, dtype=torch.bool)
            if neighbours:
                index = torch.tensor(sorted(neighbours), dtype=torch.long)
                row[index] = True                      # 1-hop
                two_hop = set()
                for neighbour in neighbours:
                    two_hop |= neighbour_sets.get(neighbour, set())
                if two_hop:
                    row[torch.tensor(sorted(two_hop), dtype=torch.long)] = True
            row[entity] = False
            if len(_support_row_cache) < 20000:
                _support_row_cache[entity] = row
        return row

    def exact_support_flags(anchors, candidate_ids):
        """Exact graph support of candidates w.r.t. anchors (bool [B, K]) —
        the PI controller's measurement signal and the oracle labels for the
        neighbourhood discriminator's online updates."""
        flags = torch.zeros_like(candidate_ids, dtype=torch.bool)
        for i, anchor in enumerate(anchors.tolist()):
            flags[i] = supported_entities_row(anchor)[candidate_ids[i]]
        return flags

    # ---------------------------------------------------------------------
    # Phase 2a: pretrain the neighbourhood discriminator on pairs built purely
    # from data — (anchor, its true filler) = fits, (anchor, another anchor's
    # same-relation filler) = does not fit.
    # ---------------------------------------------------------------------
    neighbourhood_discriminator = NeighbourhoodDiscriminator(dim=EMBEDDING_DIM).to(device)
    neighbourhood_pretrain_optimizer = torch.optim.AdamW(
        neighbourhood_discriminator.parameters(), lr=1e-3)
    binary_cross_entropy = torch.nn.BCEWithLogitsLoss()
    print(f"D_match pretraining ({NEIGHBOURHOOD_PRETRAIN_EPOCHS} epochs)...", flush=True)
    for epoch in range(1, NEIGHBOURHOOD_PRETRAIN_EPOCHS + 1):
        shuffled_triples = python_rng.sample(train_triples, len(train_triples))
        total_loss = num_batches = 0
        for start in range(0, len(shuffled_triples), BATCH_SIZE):
            batch_triples = shuffled_triples[start:start + BATCH_SIZE]
            anchor_list, candidate_list, labels = [], [], []
            for h, r, t in batch_triples:
                anchor_list.append(h); candidate_list.append(t); labels.append(1.0)
                # Negative pair: a tail from ANOTHER triple of the same
                # relation, provided it is not also a neighbour of h.
                same_relation = triples_by_relation[r]
                for _ in range(6):
                    _, _, other_tail = same_relation[
                        python_rng.randrange(len(same_relation))]
                    if (other_tail != t
                            and other_tail not in neighbour_sets.get(h, set())):
                        anchor_list.append(h); candidate_list.append(other_tail)
                        labels.append(0.0)
                        break
            anchor_ids = torch.tensor(anchor_list)
            candidate_ids = torch.tensor(candidate_list)
            neighbour_ids, neighbour_mask = sample_neighbour_batch(
                anchor_ids, exclude=candidate_ids)
            logits = neighbourhood_discriminator(
                context_table[candidate_ids], context_table[neighbour_ids],
                neighbour_mask.to(device))
            loss = binary_cross_entropy(logits,
                                        torch.tensor(labels, device=device))
            neighbourhood_pretrain_optimizer.zero_grad()
            loss.backward()
            neighbourhood_pretrain_optimizer.step()
            total_loss += loss.item(); num_batches += 1
        print(f"  dmatch {epoch}/{NEIGHBOURHOOD_PRETRAIN_EPOCHS} "
              f"bce={total_loss/max(num_batches,1):.4f}", flush=True)
    # The neighbourhood discriminator is NOT frozen after pretraining: a frozen
    # one gets exploited (the generator converges onto its blind spots). It
    # keeps training during the game on the generator's own picks, labelled
    # by EXACT graph support — so every blind spot the generator finds is
    # corrected on the next batch. The oracle only supplies labels; D_match
    # remains a learned discriminator.
    neighbourhood_online_optimizer = torch.optim.AdamW(
        neighbourhood_discriminator.parameters(), lr=NEIGHBOURHOOD_LEARNING_RATE)

    # ---------------------------------------------------------------------
    # Phase 2b: build the generator and the plausibility discriminator, then
    # pretrain the plausibility discriminator. It must already be
    # anchor-conditional BEFORE the generator starts learning, or the
    # generator falls straight into the one-alien-fits-all basin.
    # ---------------------------------------------------------------------
    generator = CandidateScoringGenerator(dim=EMBEDDING_DIM, sketch_bits=SKETCH_BITS,
                                          n_rel=n_rel).to(device)
    plausibility_discriminator = PlausibilityDiscriminator(dim=EMBEDDING_DIM, n_rel=n_rel).to(device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=GENERATOR_LEARNING_RATE,
                                           betas=(0.5, 0.999))
    plausibility_optimizer = torch.optim.Adam(plausibility_discriminator.parameters(),
                                         lr=PLAUSIBILITY_LEARNING_RATE)

    for epoch in range(1, PLAUSIBILITY_PRETRAIN_EPOCHS + 1):
        shuffled = torch.randperm(train_triples_tensor.shape[0],
                                  generator=torch_rng)
        total_loss = num_batches = 0
        for start in range(0, len(shuffled), BATCH_SIZE):
            rows = train_triples_tensor[shuffled[start:start + BATCH_SIZE]]
            if len(rows) < 4:
                continue
            h, r, t = rows[:, 0], rows[:, 1], rows[:, 2]
            head_context = context_table[h.to(device)]
            tail_context = context_table[t.to(device)]
            relation_ids = r.to(device)
            random_tails = torch.randint(0, n_ent, (len(rows),),
                                         generator=torch_rng)
            score_real = plausibility_discriminator(head_context, relation_ids,
                                               tail_context)
            score_random = plausibility_discriminator(
                head_context, relation_ids, context_table[random_tails.to(device)])
            # Third class: the real tail presented with a SAME-RELATION wrong
            # head. A random wrong head would usually be type-incompatible,
            # letting D_real win on type alone; same-relation wrong heads
            # force it to judge the individual anchor.
            wrong_anchors = torch.tensor([
                triples_by_relation[int(r[i])][
                    python_rng.randrange(len(triples_by_relation[int(r[i])]))][0]
                for i in range(len(rows))])
            score_wrong_anchor = plausibility_discriminator(
                context_table[wrong_anchors.to(device)], relation_ids,
                tail_context)
            loss = (F.binary_cross_entropy_with_logits(
                        score_real,
                        torch.full_like(score_real, 1 - LABEL_SMOOTHING))
                    + F.binary_cross_entropy_with_logits(
                        score_random, torch.zeros_like(score_random))
                    + WRONG_ANCHOR_LOSS_WEIGHT * F.binary_cross_entropy_with_logits(
                        score_wrong_anchor,
                        torch.zeros_like(score_wrong_anchor)))
            plausibility_optimizer.zero_grad(); loss.backward(); plausibility_optimizer.step()
            total_loss += loss.item(); num_batches += 1
        print(f"  dreal-pre {epoch}/{PLAUSIBILITY_PRETRAIN_EPOCHS} "
              f"loss={total_loss/max(num_batches,1):.4f}", flush=True)

    def save_checkpoint(path):
        """Full candidate_v2 payload — every snapshot is independently
        loadable by kgsage.corruption_generation (same contract as the final
        save). Do not rename any key: they are the checkpoint contract."""
        torch.save({
            "arch": "candidate_v2",
            "generator_state": generator.state_dict(),
            "dmatch_state": neighbourhood_discriminator.state_dict(),
            "dreal_state": plausibility_discriminator.state_dict(),
            "context_embeddings": context_table.cpu(),
            "sketches": (membership_sketches > 0).to(torch.uint8).cpu(),
            "sketch_bits": SKETCH_BITS,
            "pool_masks": pool_masks,
            "cand_k": NUM_CANDIDATES, "dim": EMBEDDING_DIM, "tau": GUMBEL_TEMPERATURE,
            "ent2id": kg["ent2id"], "rel2id": kg["rel2id"],
            "id2ent": kg["id2ent"], "id2rel": kg["id2rel"],
            "real_triples": list(kg["triple_set_all"]),
            "n_ent": n_ent, "n_rel": n_rel,
            "train_split": "train", "alpha_final": alpha,
            "alpha_target": CORROBORATION_TARGET, "seed": args.seed,
        }, path)

    # ---------------------------------------------------------------------
    # Phase 2c: the dual-discriminator game.
    # ---------------------------------------------------------------------
    alpha = ALPHA_INITIAL
    previous_error = 0.0
    print("-" * 60, flush=True)
    print(f"Dual-discriminator: {args.epochs} epochs, K={NUM_CANDIDATES}, tau={GUMBEL_TEMPERATURE}, "
          f"alpha0={alpha} target={CORROBORATION_TARGET}", flush=True)
    for epoch in range(1, args.epochs + 1):
        in_warmup = epoch <= PLAUSIBILITY_ONLY_EPOCHS
        if in_warmup:
            alpha = 0.0                  # curriculum: learn "plausible" FIRST
        elif alpha == 0.0:
            alpha = ALPHA_INITIAL           # ramp point: hand over to the PI loop
        shuffled = torch.randperm(train_triples_tensor.shape[0],
                                  generator=torch_rng)
        epoch_metrics = defaultdict(float)
        distinct_picks = set()
        epoch_start = time.perf_counter()
        for batch_index, start in enumerate(
                range(0, len(shuffled), BATCH_SIZE)):
            rows = train_triples_tensor[shuffled[start:start + BATCH_SIZE]]
            if len(rows) < 4:
                continue
            h, r, t = rows[:, 0], rows[:, 1], rows[:, 2]
            num_rows = len(rows)
            # Alternate the corrupted slot batch by batch. The ANCHOR is the
            # entity that keeps its slot; the TRUE FILLER is the one being
            # replaced.
            slot = TAIL if batch_index % 2 == 0 else HEAD
            anchor_entities = h if slot == TAIL else t
            true_fillers = t if slot == TAIL else h

            candidate_ids, candidate_log_q = candidate_sampler.sample(
                r.numpy(), slot)
            # Ban candidates that would make the "corruption" true or
            # degenerate: every known-true filler of this query (train), the
            # current true filler, and the anchor itself (self-loop).
            banned_mask = torch.zeros(num_rows, candidate_ids.shape[1],
                                      dtype=torch.bool)
            for i in range(num_rows):
                query_key = ((int(h[i]), int(r[i])) if slot == TAIL
                             else (int(r[i]), int(t[i])))
                known_true = (true_tails if slot == TAIL
                              else true_heads).get(query_key, set())
                banned_mask[i] = torch.tensor(
                    [(int(x) in known_true) or int(x) == int(true_fillers[i])
                     or int(x) == int(anchor_entities[i])
                     for x in candidate_ids[i].tolist()])

            def generator_logits():
                logits = generator(h.to(device), r.to(device), t.to(device),
                                   context_table,
                                   membership_sketches[anchor_entities].to(device),
                                   candidate_ids.to(device),
                                   candidate_log_q.to(device), slot)
                logits = logits.masked_fill(banned_mask.to(device),
                                            float("-inf"))
                # Some (slot, relation) pools are singletons, so a whole row
                # can be banned. One all--inf row would NaN-poison every
                # weight downstream, so such rows get finite dummy logits and
                # are EXCLUDED from all losses and stats via `valid_rows`.
                valid_rows = torch.isfinite(logits).any(dim=1)
                if not bool(valid_rows.all()):
                    logits = torch.where(valid_rows.unsqueeze(1), logits,
                                         torch.zeros_like(logits))
                return logits, valid_rows

            # ---- plausibility discriminator step ----
            with torch.no_grad():
                logits, valid_rows = generator_logits()
                selection = gumbel_softmax(logits, tau=GUMBEL_TEMPERATURE, hard=True)
                generated_embedding = torch.einsum(
                    "bk,bkd->bd", selection,
                    context_table[candidate_ids.to(device)])
            if not bool(valid_rows.any()):
                continue
            anchor_context = context_table[anchor_entities.to(device)]
            score_generated = plausibility_discriminator(
                anchor_context[valid_rows], r.to(device)[valid_rows],
                generated_embedding[valid_rows])
            score_real = plausibility_discriminator(
                context_table[h.to(device)], r.to(device),
                context_table[t.to(device)]) if slot == TAIL else \
                plausibility_discriminator(
                    context_table[t.to(device)], r.to(device),
                    context_table[h.to(device)])
            plausibility_loss = (F.binary_cross_entropy_with_logits(
                                score_real,
                                torch.full_like(score_real, 1 - LABEL_SMOOTHING))
                            + F.binary_cross_entropy_with_logits(
                                score_generated,
                                torch.zeros_like(score_generated)))
            if WRONG_ANCHOR_LOSS_WEIGHT > 0:
                # Wrong-anchor class (GAN-CLS): the TRUE filler presented
                # with another anchor of the same relation, labelled fake.
                # Type and popularity are identical across the real and
                # wrong-anchor classes, so the only winning strategy is to
                # judge plausibility CONDITIONAL on the individual anchor.
                true_filler_context = (context_table[t.to(device)]
                                       if slot == TAIL
                                       else context_table[h.to(device)])
                # The wrong anchor must be sampled from the ANCHOR slot:
                # heads when the tail is corrupted, tails when the head is —
                # otherwise D_real could reject by slot type alone.
                wrong_anchor_slot = 0 if slot == TAIL else 2
                wrong_anchors = torch.tensor([
                    triples_by_relation[int(r[i])][python_rng.randrange(
                        len(triples_by_relation[int(r[i])]))][wrong_anchor_slot]
                    for i in range(num_rows)])
                score_wrong_anchor = plausibility_discriminator(
                    context_table[wrong_anchors.to(device)], r.to(device),
                    true_filler_context)
                plausibility_loss = plausibility_loss + (
                    WRONG_ANCHOR_LOSS_WEIGHT * F.binary_cross_entropy_with_logits(
                        score_wrong_anchor,
                        torch.zeros_like(score_wrong_anchor)))
            plausibility_optimizer.zero_grad()
            plausibility_loss.backward()
            plausibility_optimizer.step()

            # ---- generator step ----
            logits, valid_rows = generator_logits()
            selection = gumbel_softmax(logits, tau=GUMBEL_TEMPERATURE, hard=True)
            generated_embedding = torch.einsum(
                "bk,bkd->bd", selection, context_table[candidate_ids.to(device)])
            plausibility_of_generated = plausibility_discriminator(
                anchor_context[valid_rows], r.to(device)[valid_rows],
                generated_embedding[valid_rows])
            neighbour_ids, neighbour_mask = sample_neighbour_batch(
                anchor_entities)
            neighbourhood_fit_of_generated = neighbourhood_discriminator(
                generated_embedding[valid_rows],
                context_table[neighbour_ids.to(device)][valid_rows],
                neighbour_mask.to(device)[valid_rows])
            # The alienation term is a HINGE, not a graded reward: past the
            # margin there is no payoff for deeper alienation, so the ranking
            # WITHIN the alien set is carried by the plausibility discriminator.
            generator_loss = (-plausibility_of_generated
                              + alpha * torch.relu(neighbourhood_fit_of_generated
                                                   - CONTRADICTION_MARGIN)).mean()
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            # ---- measurement (hard picks = what would actually be emitted) ----
            with torch.no_grad():
                picked_columns = selection[valid_rows].argmax(dim=1)
                picked_entities = candidate_ids[valid_rows.cpu()].gather(
                    1, picked_columns.cpu().unsqueeze(1)).squeeze(1)
                support_matrix = exact_support_flags(
                    anchor_entities[valid_rows.cpu()],
                    candidate_ids[valid_rows.cpu()])
                picked_supported = support_matrix.gather(
                    1, picked_columns.cpu().unsqueeze(1)).squeeze(1)
                corroborated_fraction = float(picked_supported.float().mean())
                distinct_picks.update(picked_entities.tolist())

            # ---- neighbourhood discriminator online step: the generator's own
            #      picks, labelled by exact graph support ----
            neighbourhood_logits = neighbourhood_discriminator(
                context_table[picked_entities.to(device)],
                context_table[neighbour_ids.to(device)][valid_rows],
                neighbour_mask.to(device)[valid_rows])
            neighbourhood_online_loss = binary_cross_entropy(
                neighbourhood_logits, picked_supported.float().to(device))
            neighbourhood_online_optimizer.zero_grad()
            neighbourhood_online_loss.backward()
            neighbourhood_online_optimizer.step()

            # ---- PI controller with anti-windup (inactive during warm-up) ----
            if np.isfinite(corroborated_fraction) and not in_warmup:
                error = corroborated_fraction - CORROBORATION_TARGET
                alpha_updated = (alpha + PI_PROPORTIONAL_GAIN * (error - previous_error)
                                 + PI_INTEGRAL_GAIN * error)
                alpha = float(min(max(alpha_updated, 0.0), ALPHA_MAX))
                if alpha_updated == alpha:   # integrate only when unsaturated
                    previous_error = error
            with torch.no_grad():
                epoch_metrics["corr_mass"] += corroborated_fraction
                epoch_metrics["d_acc_real"] += float(
                    (torch.sigmoid(score_real) > 0.5).float().mean())
                epoch_metrics["d_acc_fake"] += float(
                    (torch.sigmoid(score_generated) < 0.5).float().mean())
                epoch_metrics["g_match"] += float(
                    neighbourhood_fit_of_generated.mean())
                epoch_metrics["dm_online"] += float(neighbourhood_online_loss)
                epoch_metrics["nb"] += 1

        num_batches = max(int(epoch_metrics["nb"]), 1)
        print(f"  epoch {epoch:3d}/{args.epochs}  "
              f"corr-pick={epoch_metrics['corr_mass']/num_batches:.3f} "
              f"alpha={alpha:.2f}  "
              f"D-acc={epoch_metrics['d_acc_real']/num_batches:.2f}"
              f"/{epoch_metrics['d_acc_fake']/num_batches:.2f} "
              f"g_match={epoch_metrics['g_match']/num_batches:+.2f}  "
              f"dm-online={epoch_metrics['dm_online']/num_batches:.3f} "
              f"distinct={len(distinct_picks)} "
              f"({time.perf_counter()-epoch_start:.0f}s)", flush=True)

        # Per-epoch snapshots start with the alpha ramp: warm-up epochs are
        # not generator candidates, the pressure epochs around the ramp are.
        if (args.snapshot_every > 0 and epoch > PLAUSIBILITY_ONLY_EPOCHS
                and epoch % args.snapshot_every == 0):
            checkpoint_stem = (args.out[:-3] if args.out.endswith(".pt")
                               else args.out)
            snapshot_path = f"{checkpoint_stem}.ep{epoch:02d}.pt"
            save_checkpoint(snapshot_path)
            print(f"  snapshot -> {snapshot_path}", flush=True)

    # ---------- final checkpoint ----------
    save_checkpoint(args.out)
    print(f"Saved KGSAGE-2 checkpoint to {args.out}", flush=True)


if __name__ == "__main__":
    main()
