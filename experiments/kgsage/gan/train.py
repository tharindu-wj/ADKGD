"""KGSAGE trainer: the dual-discriminator adversarial game (paper: Methodology).

The pipeline phases of the method, in this one file:

  Phase 1 — Neighbourhood Context Encoding — the RGCN context encoder is
     warmed up against a throwaway DistMult decoder on link prediction, then
     its output E' (the context_table: one context vector per entity) is
     FROZEN. Bloom membership sketches of every entity's 1-2 hop
     neighbourhood are built alongside.
  Phase 2 — Adversarial Generator Training — the candidate-scoring generator
     plays against two discriminators over the frozen E':
       plausibility discriminator (D_real)   "could this triple be real?"
                                                                  G pushes HIGH
       neighbourhood discriminator (D_match) "does the filler fit THIS
                                   anchor's neighbourhood?"        G pushes LOW
     Generator loss:  L_G = -D_real + alpha * relu(D_match - margin).
     alpha is set by a PI controller so that the corroborated fraction — the
     share of the generator's picks that the training graph corroborates —
     stays at CORROBORATION_TARGET. Step 2a pretrains both discriminators;
     step 2b plays the game.
  Phase 3 — Corruption Generation lives in kgsage/corruption_generation.py —
     it replays the checkpoint saved here.

Everything learned (E', membership sketches, candidate pools, neighbour lists,
both discriminators) sees the TRAIN split only. The guarantee that emitted
corruptions are false against ALL splits comes from the masks applied at
corruption time, not from anything learned here.

The knobs you are most likely to re-tune are the module-level constants just
below. Fixed implementation details are NOT up there — each is defined right
above the code that uses it and marked `tuned constant:` (grep for that string
to list them). Only operational flags are on the CLI.

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
# THE KNOBS YOU ARE LIKELY TO RE-TUNE.
#
# Everything else is a fixed implementation detail, defined immediately above
# the code that uses it and marked `tuned constant:` — grep for that string to
# find them all. Nothing here is on the CLI except the operational flags
# (data / out / device / seed / epochs / snapshots / E'-reuse).
# ==========================================================================

# -- sizes --
EMBEDDING_DIM                 = 64     # width of every vector (E', relations)
RGCN_NUM_LAYERS               = 2      # RGCN depth -> 2-hop context
SKETCH_BITS                   = 8192   # Bloom membership-sketch length
NUM_NEIGHBOURS_SAMPLED        = 32     # neighbours D_match attends over
NUM_CANDIDATES                = 256    # candidates scored per triple (decode = full pool)

# -- how long each stage runs (they happen in this order) --
RGCN_WARMUP_EPOCHS            = 10     # Phase 1:  build E' (skipped with --init_context_from)
RGCN_WARMUP_BATCH_SIZE        = 4096   # Phase 1:  batch size for that warm-up
PLAUSIBILITY_PRETRAIN_EPOCHS  = 2      # Phase 2a: train D_real on its own, before the game
NEIGHBOURHOOD_PRETRAIN_EPOCHS = 2      # Phase 2a: train D_match on its own, before the game
EPOCHS_BEFORE_CONTRADICTION   = 2      # Phase 2b: opening game epochs that run at alpha = 0,
                                       #           i.e. plausibility only. The game's total
                                       #           length is the --epochs CLI flag.

# -- optimisation --
BATCH_SIZE                    = 256
GUMBEL_TEMPERATURE            = 0.5    # Gumbel-Softmax temperature (train + decode)
GENERATOR_LEARNING_RATE       = 1e-4
PLAUSIBILITY_LEARNING_RATE    = 3e-4   # D_real
NEIGHBOURHOOD_LEARNING_RATE   = 1e-4   # D_match's online updates during the game
                                       # (a frozen D_match gets exploited by G)

# -- the dial that decides how hard the corruptions are --
CORROBORATION_TARGET          = 0.13   # PI set-point: the share of the generator's picks
                                       # that the training graph corroborates. MUST stay
                                       # above the dataset's structural floor, or alpha
                                       # saturates and the generator collapses onto one
                                       # universal contradiction for every anchor.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=8,
                        help="adversarial game epochs (locked recipe: 8; use 2 "
                             "for a quick smoke). Anchor-specificity peaks a "
                             "few epochs after the alpha ramp and then erodes, so "
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
    # Phase 1 — Neighbourhood Context Encoding.
    # Either reuse a frozen E' (the context_table) from an earlier checkpoint,
    # or warm the RGCN context encoder up with a DistMult decoder on link
    # prediction and freeze its output. The decoder is a throwaway: it only
    # exists to give the encoder a training signal, and is deleted once E' is
    # frozen.
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
        # frozen payload key; "context_embeddings" holds the context table E',
        # i.e. the encoder's OUTPUT — not the entity_embeddings fed into it.
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
    # tuned constant: basis decomposition shares weights across relations so the
    # RGCN stays small on relation-rich graphs. 30 is the classic FB15K-237
    # setting; the encoder caps it at the relation count internally.
    RGCN_NUM_BASES = 30
    context_encoder = NeighbourhoodContextEncoder(
        n_ent, n_rel, dim=EMBEDDING_DIM, num_bases=RGCN_NUM_BASES,
        num_layers=RGCN_NUM_LAYERS).to(device)
    if skip_warmup:
        context_encoder = None
    if not skip_warmup:
        # Throwaway DistMult decoder: score(h, r, t) = sum(E'[h] * w_r * E'[t]).
        # The decoder is nothing but this per-relation weight matrix. Real
        # triples should score high, random-tail triples low — the standard
        # link-prediction warm-up that shapes E' into a meaningful
        # neighbourhood summary.
        distmult_decoder = torch.nn.Parameter(
            torch.randn(n_rel, EMBEDDING_DIM, device=device) * 0.1)
        warmup_optimizer = torch.optim.Adam(
            list(context_encoder.parameters()) + [distmult_decoder], lr=1e-3)
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
                # E' as it stands this step — still training; frozen below.
                context_table = context_encoder(edge_index, edge_type)
                positive_scores = (context_table[h] * distmult_decoder[r]
                                   * context_table[t]).sum(-1)
                negative_scores = (context_table[h] * distmult_decoder[r]
                                   * context_table[random_tails]).sum(-1)
                loss = (F.binary_cross_entropy_with_logits(
                            positive_scores, torch.ones_like(positive_scores))
                        + F.binary_cross_entropy_with_logits(
                            negative_scores, torch.zeros_like(negative_scores)))
                warmup_optimizer.zero_grad(); loss.backward(); warmup_optimizer.step()
                total_loss += loss.item(); num_batches += 1
            # "LP_loss=" is a frozen log token: the link-prediction loss of the
            # Phase-1 RGCN warm-up. Keep it so older logs stay comparable.
            print(f"  warmup {epoch}/{RGCN_WARMUP_EPOCHS} "
                  f"LP_loss={total_loss/max(num_batches,1):.4f}", flush=True)
        context_encoder.eval(); context_encoder.requires_grad_(False)
        with torch.no_grad():
            context_table = context_encoder(edge_index, edge_type).detach()
        del distmult_decoder, warmup_optimizer
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

    print("building membership sketches (train split)...", flush=True)
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

    _corroborated_row_cache: dict[int, torch.Tensor] = {}

    def corroborated_entities_row(entity: int) -> torch.Tensor:
        """Cached bool [n_ent] row: which entities the graph CORROBORATES for
        this entity — its direct neighbours plus anything within two hops
        (TRAIN adjacency)."""
        row = _corroborated_row_cache.get(entity)
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
            if len(_corroborated_row_cache) < 20000:
                _corroborated_row_cache[entity] = row
        return row

    def exact_corroboration_flags(anchors, candidate_ids):
        """Exact graph corroboration of candidates w.r.t. anchors (bool [B, K])
        — the PI controller's measurement signal and the oracle labels for the
        neighbourhood discriminator's online updates."""
        flags = torch.zeros_like(candidate_ids, dtype=torch.bool)
        for i, anchor in enumerate(anchors.tolist()):
            flags[i] = corroborated_entities_row(anchor)[candidate_ids[i]]
        return flags

    # ---------------------------------------------------------------------
    # Phase 2a — discriminator pretraining, part 1 of 2: the neighbourhood
    # discriminator (D_match), on pairs built purely from data —
    # (anchor, its true filler) = fits, (anchor, another anchor's
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
    # by EXACT graph corroboration — so every blind spot the generator finds
    # is corrected on the next batch. The oracle only supplies labels; D_match
    # remains a learned discriminator.
    neighbourhood_online_optimizer = torch.optim.AdamW(
        neighbourhood_discriminator.parameters(), lr=NEIGHBOURHOOD_LEARNING_RATE)

    # ---------------------------------------------------------------------
    # Phase 2a — discriminator pretraining, part 2 of 2: build the generator
    # and the plausibility discriminator (D_real), then pretrain D_real. It
    # must already be anchor-specific BEFORE the generator starts learning, or
    # the generator collapses straight onto one universal contradiction.
    # ---------------------------------------------------------------------
    generator = CandidateScoringGenerator(dim=EMBEDDING_DIM, sketch_bits=SKETCH_BITS,
                                          n_rel=n_rel).to(device)
    plausibility_discriminator = PlausibilityDiscriminator(dim=EMBEDDING_DIM, n_rel=n_rel).to(device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=GENERATOR_LEARNING_RATE,
                                           betas=(0.5, 0.999))
    plausibility_optimizer = torch.optim.Adam(plausibility_discriminator.parameters(),
                                         lr=PLAUSIBILITY_LEARNING_RATE)

    # tuned constants for EVERY D_real update — used here in Phase 2a and again
    # in the Phase 2b game further down:
    #   LABEL_SMOOTHING          real triples are labelled 0.9 instead of 1.0, so
    #                            D_real never becomes absolutely certain (a
    #                            saturated discriminator gives G no gradient).
    #   WRONG_ANCHOR_LOSS_WEIGHT weight of the GAN-CLS third class: a REAL filler
    #                            shown with the WRONG anchor, labelled fake. This
    #                            is what forces D_real to judge the anchor rather
    #                            than the filler alone.
    LABEL_SMOOTHING = 0.1
    WRONG_ANCHOR_LOSS_WEIGHT = 1.0

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
            plausibility_of_real = plausibility_discriminator(
                head_context, relation_ids, tail_context)
            plausibility_of_random = plausibility_discriminator(
                head_context, relation_ids, context_table[random_tails.to(device)])
            # Third class: the real tail presented with a SAME-RELATION wrong
            # head. A random wrong head would usually be type-incompatible,
            # letting D_real win on type alone; same-relation wrong heads
            # force it to judge the individual anchor.
            wrong_anchors = torch.tensor([
                triples_by_relation[int(r[i])][
                    python_rng.randrange(len(triples_by_relation[int(r[i])]))][0]
                for i in range(len(rows))])
            plausibility_of_wrong_anchor = plausibility_discriminator(
                context_table[wrong_anchors.to(device)], relation_ids,
                tail_context)
            loss = (F.binary_cross_entropy_with_logits(
                        plausibility_of_real,
                        torch.full_like(plausibility_of_real, 1 - LABEL_SMOOTHING))
                    + F.binary_cross_entropy_with_logits(
                        plausibility_of_random,
                        torch.zeros_like(plausibility_of_random))
                    + WRONG_ANCHOR_LOSS_WEIGHT * F.binary_cross_entropy_with_logits(
                        plausibility_of_wrong_anchor,
                        torch.zeros_like(plausibility_of_wrong_anchor)))
            plausibility_optimizer.zero_grad(); loss.backward(); plausibility_optimizer.step()
            total_loss += loss.item(); num_batches += 1
        # "dreal-pre" is a frozen log prefix — keep it so old logs stay
        # comparable; it marks the D_real pretraining epochs.
        print(f"  dreal-pre {epoch}/{PLAUSIBILITY_PRETRAIN_EPOCHS} "
              f"loss={total_loss/max(num_batches,1):.4f}", flush=True)

    def save_checkpoint(path):
        """Full candidate_v2 payload — every snapshot is independently
        loadable by kgsage.corruption_generation (same contract as the final
        save). Do not rename any key: they are the checkpoint contract."""
        torch.save({
            # frozen arch string; "candidate_v2" names the dual-discriminator
            # architecture and is compared as a literal by the loaders.
            "arch": "candidate_v2",
            "generator_state": generator.state_dict(),
            # frozen key; dmatch = D_match, the neighbourhood discriminator
            "dmatch_state": neighbourhood_discriminator.state_dict(),
            # frozen key; dreal = D_real, the plausibility discriminator
            "dreal_state": plausibility_discriminator.state_dict(),
            "context_embeddings": context_table.cpu(),
            # frozen key; "sketches" are the Bloom membership sketches
            "sketches": (membership_sketches > 0).to(torch.uint8).cpu(),
            "sketch_bits": SKETCH_BITS,
            "pool_masks": pool_masks,
            "cand_k": NUM_CANDIDATES, "dim": EMBEDDING_DIM, "tau": GUMBEL_TEMPERATURE,
            "ent2id": kg["ent2id"], "rel2id": kg["rel2id"],
            "id2ent": kg["id2ent"], "id2rel": kg["id2rel"],
            "real_triples": list(kg["triple_set_all"]),
            "n_ent": n_ent, "n_rel": n_rel,
            # frozen keys; alpha_final = the alpha value this run ended on,
            # alpha_target = the PI set-point CORROBORATION_TARGET, which is a
            # target CORROBORATED FRACTION, not a target alpha.
            "train_split": "train", "alpha_final": alpha,
            "alpha_target": CORROBORATION_TARGET, "seed": args.seed,
        }, path)

    # ---------------------------------------------------------------------
    # Phase 2b — the dual-discriminator game.
    # ---------------------------------------------------------------------
    # tuned constants for the contradiction-pressure controller. `alpha` is the
    # weight on the neighbourhood penalty in L_G; a PI controller nudges it each
    # batch so the corroborated fraction tracks CORROBORATION_TARGET.
    ALPHA_INITIAL = 1.0           # what alpha restarts at when pressure switches on
    ALPHA_MAX = 10.0              # anti-windup clamp: stops alpha running away when
                                  # the target is unreachable for this dataset
    PI_PROPORTIONAL_GAIN = 2.0    # reacts to the CURRENT error
    PI_INTEGRAL_GAIN = 0.2        # reacts to the ACCUMULATED error
    CONTRADICTION_MARGIN = 0.0    # hinge: alpha * relu(D_match - margin), so there
                                  # is no reward for contradicting past the margin

    alpha = ALPHA_INITIAL
    previous_error = 0.0
    print("-" * 60, flush=True)
    print(f"Dual-discriminator: {args.epochs} epochs, K={NUM_CANDIDATES}, tau={GUMBEL_TEMPERATURE}, "
          f"alpha0={alpha} target={CORROBORATION_TARGET}", flush=True)
    for epoch in range(1, args.epochs + 1):
        # The opening alpha=0 epochs are the PLAUSIBILITY-ONLY PHASE (never
        # "warm-up" — that word belongs to the Phase-1 RGCN warm-up).
        in_plausibility_only_phase = epoch <= EPOCHS_BEFORE_CONTRADICTION
        if in_plausibility_only_phase:
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
            # D_real's own reading of the generated triple, on picks detached
            # from G. The generator step below re-reads fresh picks WITH
            # gradients as plausibility_of_generated_for_g — two different
            # tensors, so they carry two different names.
            plausibility_of_generated_for_d = plausibility_discriminator(
                anchor_context[valid_rows], r.to(device)[valid_rows],
                generated_embedding[valid_rows])
            plausibility_of_real = plausibility_discriminator(
                context_table[h.to(device)], r.to(device),
                context_table[t.to(device)]) if slot == TAIL else \
                plausibility_discriminator(
                    context_table[t.to(device)], r.to(device),
                    context_table[h.to(device)])
            plausibility_loss = (F.binary_cross_entropy_with_logits(
                                plausibility_of_real,
                                torch.full_like(plausibility_of_real,
                                                1 - LABEL_SMOOTHING))
                            + F.binary_cross_entropy_with_logits(
                                plausibility_of_generated_for_d,
                                torch.zeros_like(plausibility_of_generated_for_d)))
            if WRONG_ANCHOR_LOSS_WEIGHT > 0:
                # Wrong-anchor class (GAN-CLS): the TRUE filler presented
                # with another anchor of the same relation, labelled fake.
                # Type and popularity are identical across the real and
                # wrong-anchor classes, so the only winning strategy is to
                # judge plausibility for the SPECIFIC anchor in front of it.
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
                plausibility_of_wrong_anchor = plausibility_discriminator(
                    context_table[wrong_anchors.to(device)], r.to(device),
                    true_filler_context)
                plausibility_loss = plausibility_loss + (
                    WRONG_ANCHOR_LOSS_WEIGHT * F.binary_cross_entropy_with_logits(
                        plausibility_of_wrong_anchor,
                        torch.zeros_like(plausibility_of_wrong_anchor)))
            plausibility_optimizer.zero_grad()
            plausibility_loss.backward()
            plausibility_optimizer.step()

            # ---- generator step ----
            logits, valid_rows = generator_logits()
            selection = gumbel_softmax(logits, tau=GUMBEL_TEMPERATURE, hard=True)
            generated_embedding = torch.einsum(
                "bk,bkd->bd", selection, context_table[candidate_ids.to(device)])
            plausibility_of_generated_for_g = plausibility_discriminator(
                anchor_context[valid_rows], r.to(device)[valid_rows],
                generated_embedding[valid_rows])
            neighbour_ids, neighbour_mask = sample_neighbour_batch(
                anchor_entities)
            neighbourhood_fit_of_generated = neighbourhood_discriminator(
                generated_embedding[valid_rows],
                context_table[neighbour_ids.to(device)][valid_rows],
                neighbour_mask.to(device)[valid_rows])
            # The contradiction term is a HINGE, not a graded reward: past the
            # margin there is no payoff for a deeper contradiction, so the
            # ranking WITHIN the contradicting candidates is carried by the
            # plausibility discriminator.
            generator_loss = (-plausibility_of_generated_for_g
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
                corroboration_matrix = exact_corroboration_flags(
                    anchor_entities[valid_rows.cpu()],
                    candidate_ids[valid_rows.cpu()])
                picked_corroborated = corroboration_matrix.gather(
                    1, picked_columns.cpu().unsqueeze(1)).squeeze(1)
                corroborated_fraction = float(picked_corroborated.float().mean())
                distinct_picks.update(picked_entities.tolist())

            # ---- neighbourhood discriminator online step: the generator's own
            #      picks, labelled by exact graph corroboration ----
            neighbourhood_logits = neighbourhood_discriminator(
                context_table[picked_entities.to(device)],
                context_table[neighbour_ids.to(device)][valid_rows],
                neighbour_mask.to(device)[valid_rows])
            neighbourhood_online_loss = binary_cross_entropy(
                neighbourhood_logits, picked_corroborated.float().to(device))
            neighbourhood_online_optimizer.zero_grad()
            neighbourhood_online_loss.backward()
            neighbourhood_online_optimizer.step()

            # ---- PI controller with anti-windup (proportional + integral;
            #      inactive during the plausibility-only phase) ----
            if (np.isfinite(corroborated_fraction)
                    and not in_plausibility_only_phase):
                error = corroborated_fraction - CORROBORATION_TARGET
                alpha_updated = (alpha + PI_PROPORTIONAL_GAIN * (error - previous_error)
                                 + PI_INTEGRAL_GAIN * error)
                alpha = float(min(max(alpha_updated, 0.0), ALPHA_MAX))
                if alpha_updated == alpha:   # integrate only when unsaturated
                    previous_error = error
            with torch.no_grad():
                epoch_metrics["corroborated_fraction"] += corroborated_fraction
                epoch_metrics["d_acc_real"] += float(
                    (torch.sigmoid(plausibility_of_real) > 0.5).float().mean())
                epoch_metrics["d_acc_fake"] += float(
                    (torch.sigmoid(plausibility_of_generated_for_d)
                     < 0.5).float().mean())
                epoch_metrics["g_match"] += float(
                    neighbourhood_fit_of_generated.mean())
                epoch_metrics["dm_online"] += float(neighbourhood_online_loss)
                epoch_metrics["nb"] += 1

        num_batches = max(int(epoch_metrics["nb"]), 1)
        # Frozen log tokens: every training log already collected is parsed
        # against them, so the emitted text must stay byte-identical. Decoder:
        #   corr-pick=  the corroborated fraction — the share of the
        #               generator's picks the TRAIN graph corroborates, and the
        #               PI controller's measurement. Here "corr" means
        #               CORROBORATED; in the eval CSVs corr_* means CORRUPTED.
        #   alpha=      current weight on the neighbourhood penalty
        #   D-acc=      D_real's accuracy, real/generated
        #   g_match=    D_match's mean fit score for the generator's picks
        #               (more negative = more contradictory)
        #   dm-online=  BCE of D_match's online update on those same picks
        #   distinct=   how many distinct entities got picked this epoch
        print(f"  epoch {epoch:3d}/{args.epochs}  "
              f"corr-pick={epoch_metrics['corroborated_fraction']/num_batches:.3f} "
              f"alpha={alpha:.2f}  "
              f"D-acc={epoch_metrics['d_acc_real']/num_batches:.2f}"
              f"/{epoch_metrics['d_acc_fake']/num_batches:.2f} "
              f"g_match={epoch_metrics['g_match']/num_batches:+.2f}  "
              f"dm-online={epoch_metrics['dm_online']/num_batches:.3f} "
              f"distinct={len(distinct_picks)} "
              f"({time.perf_counter()-epoch_start:.0f}s)", flush=True)

        # Per-epoch snapshots start with the alpha ramp: plausibility-only
        # epochs are not generator candidates, the contradiction-pressure
        # epochs around the ramp are.
        if (args.snapshot_every > 0 and epoch > EPOCHS_BEFORE_CONTRADICTION
                and epoch % args.snapshot_every == 0):
            checkpoint_stem = (args.out[:-3] if args.out.endswith(".pt")
                               else args.out)
            snapshot_path = f"{checkpoint_stem}.ep{epoch:02d}.pt"
            save_checkpoint(snapshot_path)
            print(f"  snapshot -> {snapshot_path}", flush=True)

    # ---------- final checkpoint ----------
    save_checkpoint(args.out)
    print(f"Saved KGSAGE checkpoint to {args.out}", flush=True)


if __name__ == "__main__":
    main()
