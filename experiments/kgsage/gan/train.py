"""KGSAGE trainer: the dual-discriminator adversarial game (paper: Methodology).

The three phases of the method, in this one file:

  1. Neighbourhood Context Encoding — the RGCN encoder is warmed up with a
     DistMult decoder on link prediction, then its output E' (one context
     vector per entity) is FROZEN. Bloom membership sketches of every
     entity's 1-2 hop neighbourhood are built alongside.
  2. Adversarial Generator Training — the candidate-scoring generator plays
     against two discriminators over the frozen E':
       realism discriminator      "could this triple be real?"   G pushes HIGH
       consistency discriminator  "does the filler fit THIS
                                   anchor's neighbourhood?"       G pushes LOW
     Generator loss:  L_G = -D_real + alpha * relu(D_match - margin).
     alpha is set by a PI controller so that the fraction of generated picks
     the training graph actually corroborates stays at ALPHA_TARGET.
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
from kgsage.gan.realism_discriminator import RealismDiscriminator
from kgsage.gan.consistency_discriminator import ConsistencyDiscriminator
from kgsage.gan.membership_sketch import build_membership_sketches
from kgsage.gan.candidate_sampler import CandidateSampler

HEAD, TAIL = 0, 2

# ==========================================================================
# Locked configuration. Tuned once during development and fixed for every
# run; only operational args (data/out/device/seed/epochs/snapshots/E'-reuse)
# stay on the CLI. Re-tune by editing here.
# ==========================================================================

# -- architecture (E', sketches, candidate set) --
DIM             = 64      # context-vector / embedding width
NUM_BASES       = 30      # RGCN basis decomposition
ENCODER_LAYERS  = 2       # RGCN depth -> 2-hop context
SKETCH_BITS     = 8192    # Bloom membership-sketch length
N_NBR           = 32      # neighbours the consistency discriminator attends over
CAND_K          = 256     # candidates scored per triple (decode = full pool)

# -- E' warm-up (skipped when --init_context_from is given) --
WARMUP_EPOCHS   = 10
WARMUP_BATCH    = 4096

# -- curriculum: pretrain both discriminators, then an alpha=0 warm-up --
DMATCH_EPOCHS         = 2
DREAL_PRETRAIN_EPOCHS = 2
ALPHA_WARMUP_EPOCHS   = 2

# -- game optimisation --
BATCH_SIZE      = 256
TAU             = 0.5     # Gumbel-Softmax temperature (train + decode)
LR_G            = 1e-4
LR_D            = 3e-4    # realism discriminator
LR_DMATCH       = 1e-4    # consistency discriminator's online updates during
                          # the game (a frozen one gets exploited by G)
LABEL_SMOOTHING = 0.1

# -- contradiction-pressure controller (alpha) --
ALPHA_TARGET    = 0.13    # PI set-point; MUST stay above the ~0.12-0.13
                          # structural floor or alpha saturates and the
                          # generator collapses to one-alien-fits-all picks
ALPHA_INIT      = 1.0     # value alpha restarts at when the warm-up ends
ALPHA_MAX       = 10.0    # anti-windup clamp
ALPHA_KP        = 2.0     # PI proportional gain
ALPHA_KI        = 0.2     # PI integral gain
MATCH_MARGIN    = 0.0     # hinge margin: alpha * relu(D_match - margin)
DREAL_MISMATCH  = 1.0     # weight of the wrong-anchor class (GAN-CLS style);
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
        assert donor["n_ent"] == n_ent and donor["dim"] == DIM, \
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
        n_ent, n_rel, dim=DIM, num_bases=NUM_BASES,
        num_layers=ENCODER_LAYERS).to(device)
    if skip_warmup:
        context_encoder = None
    if not skip_warmup:
        # DistMult decoder: score(h, r, t) = sum(E'[h] * w_r * E'[t]).
        # Real triples should score high, random-tail triples low — the
        # standard link-prediction warm-up that shapes E' into a meaningful
        # neighbourhood summary.
        distmult_relations = torch.nn.Parameter(
            torch.randn(n_rel, DIM, device=device) * 0.1)
        warmup_optimizer = torch.optim.Adam(
            list(context_encoder.parameters()) + [distmult_relations], lr=1e-3)
        torch_rng = torch.Generator().manual_seed(args.seed + 1)
        for epoch in range(1, WARMUP_EPOCHS + 1):
            shuffled = torch.randperm(train_triples_tensor.shape[0],
                                      generator=torch_rng)
            total_loss = num_batches = 0
            for start in range(0, len(shuffled), WARMUP_BATCH):
                rows = train_triples_tensor[
                    shuffled[start:start + WARMUP_BATCH]].to(device)
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
            print(f"  warmup {epoch}/{WARMUP_EPOCHS} "
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
                                         k=CAND_K, seed=args.seed)
    # Type pools for the checkpoint: which entities were observed in each
    # (slot, relation) position. Corruption generation scores the FULL pool.
    pool_masks = torch.zeros(2, n_rel, n_ent, dtype=torch.bool)
    for h, r, t in train_triples:
        pool_masks[0, r, h] = True
        pool_masks[1, r, t] = True

    def sample_neighbour_batch(anchors, exclude=None):
        """Sample up to N_NBR neighbours per anchor from TRAIN adjacency.

        Returns (neighbour_ids [B, N_NBR], neighbour_mask [B, N_NBR]); the
        mask marks which positions hold a real neighbour (rows are padded).
        """
        batch_size = len(anchors)
        neighbour_ids = torch.zeros(batch_size, N_NBR, dtype=torch.long)
        neighbour_mask = torch.zeros(batch_size, N_NBR, dtype=torch.bool)
        for i, anchor in enumerate(anchors):
            neighbours = neighbour_sets.get(int(anchor), ())
            neighbours = [n for n in neighbours
                          if exclude is None or n != int(exclude[i])]
            if not neighbours:
                continue
            if len(neighbours) > N_NBR:
                neighbours = python_rng.sample(neighbours, N_NBR)
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
        consistency discriminator's online updates."""
        flags = torch.zeros_like(candidate_ids, dtype=torch.bool)
        for i, anchor in enumerate(anchors.tolist()):
            flags[i] = supported_entities_row(anchor)[candidate_ids[i]]
        return flags

    # ---------------------------------------------------------------------
    # Phase 2a: pretrain the consistency discriminator on pairs built purely
    # from data — (anchor, its true filler) = fits, (anchor, another anchor's
    # same-relation filler) = does not fit.
    # ---------------------------------------------------------------------
    consistency_discriminator = ConsistencyDiscriminator(dim=DIM).to(device)
    consistency_pretrain_optimizer = torch.optim.AdamW(
        consistency_discriminator.parameters(), lr=1e-3)
    binary_cross_entropy = torch.nn.BCEWithLogitsLoss()
    print(f"D_match pretraining ({DMATCH_EPOCHS} epochs)...", flush=True)
    for epoch in range(1, DMATCH_EPOCHS + 1):
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
            logits = consistency_discriminator(
                context_table[candidate_ids], context_table[neighbour_ids],
                neighbour_mask.to(device))
            loss = binary_cross_entropy(logits,
                                        torch.tensor(labels, device=device))
            consistency_pretrain_optimizer.zero_grad()
            loss.backward()
            consistency_pretrain_optimizer.step()
            total_loss += loss.item(); num_batches += 1
        print(f"  dmatch {epoch}/{DMATCH_EPOCHS} "
              f"bce={total_loss/max(num_batches,1):.4f}", flush=True)
    # The consistency discriminator is NOT frozen after pretraining: a frozen
    # one gets exploited (the generator converges onto its blind spots). It
    # keeps training during the game on the generator's own picks, labelled
    # by EXACT graph support — so every blind spot the generator finds is
    # corrected on the next batch. The oracle only supplies labels; D_match
    # remains a learned discriminator.
    consistency_online_optimizer = torch.optim.AdamW(
        consistency_discriminator.parameters(), lr=LR_DMATCH)

    # ---------------------------------------------------------------------
    # Phase 2b: build the generator and the realism discriminator, then
    # pretrain the realism discriminator. It must already be
    # anchor-conditional BEFORE the generator starts learning, or the
    # generator falls straight into the one-alien-fits-all basin.
    # ---------------------------------------------------------------------
    generator = CandidateScoringGenerator(dim=DIM, sketch_bits=SKETCH_BITS,
                                          n_rel=n_rel).to(device)
    realism_discriminator = RealismDiscriminator(dim=DIM, n_rel=n_rel).to(device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=LR_G,
                                           betas=(0.5, 0.999))
    realism_optimizer = torch.optim.Adam(realism_discriminator.parameters(),
                                         lr=LR_D)

    for epoch in range(1, DREAL_PRETRAIN_EPOCHS + 1):
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
            score_real = realism_discriminator(head_context, relation_ids,
                                               tail_context)
            score_random = realism_discriminator(
                head_context, relation_ids, context_table[random_tails.to(device)])
            # Third class: the real tail presented with a SAME-RELATION wrong
            # head. A random wrong head would usually be type-incompatible,
            # letting D_real win on type alone; same-relation wrong heads
            # force it to judge the individual anchor.
            wrong_anchors = torch.tensor([
                triples_by_relation[int(r[i])][
                    python_rng.randrange(len(triples_by_relation[int(r[i])]))][0]
                for i in range(len(rows))])
            score_wrong_anchor = realism_discriminator(
                context_table[wrong_anchors.to(device)], relation_ids,
                tail_context)
            loss = (F.binary_cross_entropy_with_logits(
                        score_real,
                        torch.full_like(score_real, 1 - LABEL_SMOOTHING))
                    + F.binary_cross_entropy_with_logits(
                        score_random, torch.zeros_like(score_random))
                    + DREAL_MISMATCH * F.binary_cross_entropy_with_logits(
                        score_wrong_anchor,
                        torch.zeros_like(score_wrong_anchor)))
            realism_optimizer.zero_grad(); loss.backward(); realism_optimizer.step()
            total_loss += loss.item(); num_batches += 1
        print(f"  dreal-pre {epoch}/{DREAL_PRETRAIN_EPOCHS} "
              f"loss={total_loss/max(num_batches,1):.4f}", flush=True)

    def save_checkpoint(path):
        """Full candidate_v2 payload — every snapshot is independently
        loadable by kgsage.corruption_generation (same contract as the final
        save). Do not rename any key: they are the checkpoint contract."""
        torch.save({
            "arch": "candidate_v2",
            "generator_state": generator.state_dict(),
            "dmatch_state": consistency_discriminator.state_dict(),
            "dreal_state": realism_discriminator.state_dict(),
            "context_embeddings": context_table.cpu(),
            "sketches": (membership_sketches > 0).to(torch.uint8).cpu(),
            "sketch_bits": SKETCH_BITS,
            "pool_masks": pool_masks,
            "cand_k": CAND_K, "dim": DIM, "tau": TAU,
            "ent2id": kg["ent2id"], "rel2id": kg["rel2id"],
            "id2ent": kg["id2ent"], "id2rel": kg["id2rel"],
            "real_triples": list(kg["triple_set_all"]),
            "n_ent": n_ent, "n_rel": n_rel,
            "train_split": "train", "alpha_final": alpha,
            "alpha_target": ALPHA_TARGET, "seed": args.seed,
        }, path)

    # ---------------------------------------------------------------------
    # Phase 2c: the dual-discriminator game.
    # ---------------------------------------------------------------------
    alpha = ALPHA_INIT
    previous_error = 0.0
    print("-" * 60, flush=True)
    print(f"Dual-discriminator: {args.epochs} epochs, K={CAND_K}, tau={TAU}, "
          f"alpha0={alpha} target={ALPHA_TARGET}", flush=True)
    for epoch in range(1, args.epochs + 1):
        in_warmup = epoch <= ALPHA_WARMUP_EPOCHS
        if in_warmup:
            alpha = 0.0                  # curriculum: learn "plausible" FIRST
        elif alpha == 0.0:
            alpha = ALPHA_INIT           # ramp point: hand over to the PI loop
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

            # ---- realism discriminator step ----
            with torch.no_grad():
                logits, valid_rows = generator_logits()
                selection = gumbel_softmax(logits, tau=TAU, hard=True)
                generated_embedding = torch.einsum(
                    "bk,bkd->bd", selection,
                    context_table[candidate_ids.to(device)])
            if not bool(valid_rows.any()):
                continue
            anchor_context = context_table[anchor_entities.to(device)]
            score_generated = realism_discriminator(
                anchor_context[valid_rows], r.to(device)[valid_rows],
                generated_embedding[valid_rows])
            score_real = realism_discriminator(
                context_table[h.to(device)], r.to(device),
                context_table[t.to(device)]) if slot == TAIL else \
                realism_discriminator(
                    context_table[t.to(device)], r.to(device),
                    context_table[h.to(device)])
            realism_loss = (F.binary_cross_entropy_with_logits(
                                score_real,
                                torch.full_like(score_real, 1 - LABEL_SMOOTHING))
                            + F.binary_cross_entropy_with_logits(
                                score_generated,
                                torch.zeros_like(score_generated)))
            if DREAL_MISMATCH > 0:
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
                score_wrong_anchor = realism_discriminator(
                    context_table[wrong_anchors.to(device)], r.to(device),
                    true_filler_context)
                realism_loss = realism_loss + (
                    DREAL_MISMATCH * F.binary_cross_entropy_with_logits(
                        score_wrong_anchor,
                        torch.zeros_like(score_wrong_anchor)))
            realism_optimizer.zero_grad()
            realism_loss.backward()
            realism_optimizer.step()

            # ---- generator step ----
            logits, valid_rows = generator_logits()
            selection = gumbel_softmax(logits, tau=TAU, hard=True)
            generated_embedding = torch.einsum(
                "bk,bkd->bd", selection, context_table[candidate_ids.to(device)])
            realism_of_generated = realism_discriminator(
                anchor_context[valid_rows], r.to(device)[valid_rows],
                generated_embedding[valid_rows])
            neighbour_ids, neighbour_mask = sample_neighbour_batch(
                anchor_entities)
            consistency_of_generated = consistency_discriminator(
                generated_embedding[valid_rows],
                context_table[neighbour_ids.to(device)][valid_rows],
                neighbour_mask.to(device)[valid_rows])
            # The alienation term is a HINGE, not a graded reward: past the
            # margin there is no payoff for deeper alienation, so the ranking
            # WITHIN the alien set is carried by the realism discriminator.
            generator_loss = (-realism_of_generated
                              + alpha * torch.relu(consistency_of_generated
                                                   - MATCH_MARGIN)).mean()
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

            # ---- consistency discriminator online step: the generator's own
            #      picks, labelled by exact graph support ----
            consistency_logits = consistency_discriminator(
                context_table[picked_entities.to(device)],
                context_table[neighbour_ids.to(device)][valid_rows],
                neighbour_mask.to(device)[valid_rows])
            consistency_online_loss = binary_cross_entropy(
                consistency_logits, picked_supported.float().to(device))
            consistency_online_optimizer.zero_grad()
            consistency_online_loss.backward()
            consistency_online_optimizer.step()

            # ---- PI controller with anti-windup (inactive during warm-up) ----
            if np.isfinite(corroborated_fraction) and not in_warmup:
                error = corroborated_fraction - ALPHA_TARGET
                alpha_updated = (alpha + ALPHA_KP * (error - previous_error)
                                 + ALPHA_KI * error)
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
                    consistency_of_generated.mean())
                epoch_metrics["dm_online"] += float(consistency_online_loss)
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
        if (args.snapshot_every > 0 and epoch > ALPHA_WARMUP_EPOCHS
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
