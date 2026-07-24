"""Corruption Generation (paper: Methodology, final phase).

Loads a trained KGSAGE checkpoint and produces one corruption (negative
triple) per input triple. This is the public generation API of the package —
a downstream detector calls it (through a bridge such as
kgsage_bridge.bridge) every time it needs a batch of negatives. Everything
stays in-process; no intermediate files. PyG is never needed here: the
checkpoint carries the frozen context table E' and the membership sketches.

The pipeline, one corruption per real triple:

  STEP 1  Translate the caller's integer ids -> strings -> the generator's
          integer ids (the caller and the generator may number the same
          entity differently; strings are the shared language).
  STEP 2  Pick the slot to corrupt: head or tail, 50/50. The relation slot
          is never corrupted — with head and tail fixed there is rarely a
          coherent alternative relation, so relation corruptions come out
          type-incoherent (confirmed in downstream-detector logs).
  STEP 3  Score the relation's FULL type pool with the generator,
          conditioned on the frozen E' rows of the triple and the anchor's
          membership sketch. (The anchor = the entity that keeps its slot.)
  STEP 4  Mask out everything that must never be emitted: the original
          value, every KNOWN-TRUE filler of the query across ALL splits
          (this is the falseness guarantee), the anchor itself (self-loop),
          everything outside the relation's type pool, and — optionally —
          every candidate the anchor's neighbourhood corroborates (the
          inverted-support mask, see support_ban_row).
  STEP 5  Sample the replacement: argmax over the masked scores plus Gumbel
          noise, drawn from a torch.Generator seeded from the caller's
          numpy rng — reproducible per seed.
  STEP 6  Check the candidate (no self-loop, not a real fact); up to
          `max_resample` redraws. If every redraw fails, the ORIGINAL triple
          is emitted as a null corruption, counted in `used_original` and
          listed in `null_indices` — a null is a real fact and callers must
          drop or replace it before training on it. There is NO hidden
          random fallback.
  STEP 7  Translate the corrupted triple back to the caller's integer ids.
"""
import numpy as np
import torch


def load_checkpoint(ckpt_path, device=None):
    """Reconstruct the trained generator and its lookup tables from a .pt file.

    Only candidate_v2 payloads (CandidateScoringGenerator + sketches) are
    loadable. Older checkpoints remain useful only as --init_context_from E'
    donors for the trainer, which reads their tensors directly and never
    calls this function.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    if payload.get("arch") == "candidate_v2":
        return _load_candidate_v2_payload(payload, ckpt_path, device)

    raise ValueError(
        f"Checkpoint {ckpt_path!r} is not a candidate_v2 payload "
        f"(arch={payload.get('arch')!r}). Legacy checkpoints are no longer "
        "loadable; use one of the canonical generator_*.pt artifacts or "
        "retrain with `python -m kgsage.gan.train`."
    )


def _load_candidate_v2_payload(saved, ckpt_path, device):
    """Build the runtime payload dict from a candidate_v2 checkpoint.

    The keys of the returned dict are a contract — knockout_eval, the CLI
    scripts and the bridge all read them. Do not rename them.
    """
    from kgsage.gan.generator import CandidateScoringGenerator

    generator = CandidateScoringGenerator(
        dim=saved["dim"], sketch_bits=saved["sketch_bits"],
        n_rel=saved["n_rel"]).to(device)
    generator.load_state_dict(saved["generator_state"])
    generator.eval()

    entity_context = saved["context_embeddings"].to(device)

    # Known-true fillers per query, across ALL splits — the falseness bans.
    real_triple_set = set(tuple(t) for t in saved["real_triples"])
    true_tails, true_heads = {}, {}
    for h, r, t in real_triple_set:
        true_tails.setdefault((h, r), []).append(t)
        true_heads.setdefault((r, t), []).append(h)

    pool_masks = saved.get("pool_masks")
    if pool_masks is None:
        # Early smoke checkpoints predate pool storage: derive pools from the
        # all-splits triples (slightly more permissive than train-only pools).
        print(f"[v2] {ckpt_path}: no pool_masks in payload -- deriving from "
              "real_triples (all splits)", flush=True)
        pool_masks = torch.zeros(2, saved["n_rel"], saved["n_ent"],
                                 dtype=torch.bool)
        for h, r, t in real_triple_set:
            pool_masks[0, r, h] = True
            pool_masks[1, r, t] = True

    return {
        "arch": "candidate_v2",
        "generator": generator,
        "device": device,
        "entity_context": entity_context,
        "sketches": saved["sketches"].float(),
        "ent2id": saved["ent2id"], "rel2id": saved["rel2id"],
        "id2ent": saved["id2ent"], "id2rel": saved["id2rel"],
        "real_triple_set": real_triple_set,
        "true_tails": true_tails, "true_heads": true_heads,
        "pool_masks": pool_masks,
        "n_ent": saved["n_ent"], "n_rel": saved["n_rel"],
        "tau": saved.get("tau", 0.5),
    }


def support_ban_row(payload, anchor, support_max=0):
    """Bool [n_ent] row: candidates the anchor's neighbourhood SUPPORTS.

    A candidate x counts as supported when the anchor's surroundings
    corroborate it:
      - x is a direct (1-hop) neighbour of the anchor, or
      - x shares more than `support_max` neighbours with the anchor.

    The INVERTED-SUPPORT MASK bans exactly these, so an emitted corruption is
    contradicted by the anchor's neighbourhood BY CONSTRUCTION ("no one
    around this entity points at the replacement") instead of relying on the
    learned scores alone. support_max=0 is the strict reading: one shared
    neighbour already counts as corroboration.

    The undirected adjacency is built once per payload from the checkpoint's
    all-splits triples and cached; per-anchor rows are cached too.
    """
    import scipy.sparse as sp

    cache = payload.setdefault("_support_cache", {})
    cache_key = (anchor, support_max)
    if cache_key in cache:
        return cache[cache_key]

    adjacency = payload.get("_support_adj")
    if adjacency is None:
        n_ent = payload["n_ent"]
        rows, cols = [], []
        for h, _, t in payload["real_triple_set"]:
            rows.append(h); cols.append(t)
            rows.append(t); cols.append(h)
        adjacency = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.int32), (rows, cols)),
            shape=(n_ent, n_ent))
        adjacency.data[:] = 1              # collapse parallel edges to 0/1
        payload["_support_adj"] = adjacency

    anchor_neighbours = adjacency.getrow(anchor)     # 1 x n: N(anchor)
    shared_counts = anchor_neighbours @ adjacency    # 1 x n: |N(anchor) ∩ N(x)|
    ban_numpy = ((anchor_neighbours.toarray()[0] > 0)
                 | (shared_counts.toarray()[0] > support_max))
    ban = torch.from_numpy(ban_numpy).to(payload["device"])
    if len(cache) < 20000:
        cache[cache_key] = ban
    return ban


def _sample_replacement_index(logits, clean_index, torch_rng,
                              banned=None, pool_row=None, self_row=None,
                              support_ban=None):
    """STEPS 4-5 for one row: apply the masks, then Gumbel-sample an index.

    Masks applied (each optional beyond the original value):
      clean_index : the true value — forces the slot to actually change.
      banned      : every known-true filler of this query (all splits).
      self_row    : the triple's other entity (self-loop ban).
      pool_row    : bool [n_ent] type pool — bans everything outside the
                    relation's observed slot fillers.
      support_ban : bool [n_ent] inverted-support mask — bans every candidate
                    the anchor's neighbourhood corroborates (see
                    support_ban_row), making the pick neighbourhood-
                    contradicting by construction.

    Sampling adds Gumbel noise at temperature 0.5 (mostly argmax) drawn from
    the caller's seeded torch.Generator, so generation is reproducible.

    If a row has nothing left to sample, masks are relaxed in order: the
    support ban is lifted first, then the type pool. The correctness bans
    (true value, known-true, self-loop) are NEVER lifted.

    Returns (index, lifted_support): index is -1 if nothing is sampleable;
    lifted_support flags that the support mask had to be dropped.
    """
    def correctness_masked():
        masked = logits.clone()
        masked[clean_index] = float("-inf")
        if banned:
            masked[banned] = float("-inf")
        if self_row is not None:
            masked[self_row] = float("-inf")
        return masked

    lifted_support = False
    masked = correctness_masked()
    if pool_row is not None:
        masked[~pool_row] = float("-inf")
    if support_ban is not None:
        masked[support_ban] = float("-inf")
        if torch.isinf(masked).all():          # degenerate: lift support first
            lifted_support = True
            masked = correctness_masked()
            if pool_row is not None:
                masked[~pool_row] = float("-inf")
    if pool_row is not None and torch.isinf(masked).all():
        masked = correctness_masked()          # degenerate pool: lift pool ban
        if support_ban is not None and not lifted_support:
            masked[support_ban] = float("-inf")
            if torch.isinf(masked).all():
                lifted_support = True
                masked = correctness_masked()
    if torch.isinf(masked).all():
        return -1, lifted_support

    uniform_noise = torch.empty_like(masked)
    uniform_noise.uniform_(generator=torch_rng).clamp_(1e-10, 1.0 - 1e-10)
    gumbel_noise = -torch.log(-torch.log(uniform_noise))
    return int((masked + gumbel_noise * 0.5).argmax().item()), lifted_support


def generate_negatives(triples, payload, id_maps, rng=None,
                       batch_size=256, max_resample=8, support_max=None):
    """Generate one corruption per input triple. Main entry point.

    triples      : list of (h, r, t) in the CALLER's integer id space.
    payload      : the dict returned by load_checkpoint().
    id_maps      : dict with 'id2ent', 'id2rel', 'ent2id', 'rel2id' for the
                   caller's vocabulary (translation goes through strings).
    rng          : numpy random Generator, seeded by the caller.
    max_resample : redraws before a row degrades to a null corruption.
    support_max  : None = off. An integer >= 0 turns on the INVERTED-SUPPORT
                   MASK: candidates the anchor's neighbourhood corroborates
                   (direct neighbours, or more than support_max shared
                   neighbours) become unsampleable, so every emitted
                   corruption contradicts the neighbourhood by construction.
                   Degenerate rows lift this mask first (counted in stats).

    Returns (negatives, stats). stats['null_indices'] lists the positions
    whose emitted 'negative' is the original triple — callers training on
    these negatives must drop or replace those rows.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if payload.get("arch") == "candidate_v2":
        return _generate_negatives_candidate_v2(triples, payload, id_maps,
                                                rng, max_resample, support_max)

    raise ValueError(
        "generate_negatives requires a candidate_v2 payload; legacy "
        "checkpoints are no longer supported."
    )


def _generate_negatives_candidate_v2(triples, payload, id_maps, rng,
                                     max_resample, support_max):
    """The candidate_v2 decode: score the relation's FULL type pool per row,
    scatter those scores into an n_ent-wide vector, then run the shared mask
    ladder (_sample_replacement_index)."""
    generator = payload["generator"]
    device = payload["device"]
    entity_context = payload["entity_context"]
    sketches = payload["sketches"]
    ent2id_gen, rel2id_gen = payload["ent2id"], payload["rel2id"]
    id2ent_gen, id2rel_gen = payload["id2ent"], payload["id2rel"]
    real_triple_set = payload["real_triple_set"]
    true_tails = payload.get("true_tails", {})
    true_heads = payload.get("true_heads", {})
    pool_masks = payload["pool_masks"]
    n_ent = payload["n_ent"]

    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(int(rng.integers(0, 2**31 - 1)))

    # STEP 1: caller ids -> strings -> generator ids.
    generator_triples = []
    for h_caller, r_caller, t_caller in triples:
        generator_triples.append(
            (ent2id_gen[id_maps["id2ent"][h_caller]],
             rel2id_gen[id_maps["id2rel"][r_caller]],
             ent2id_gen[id_maps["id2ent"][t_caller]]))

    negatives = []
    stats = {"processed": 0, "used_original": 0, "null_indices": [],
             "resampled": 0, "type_valid": 0, "slot_h": 0, "slot_r": 0,
             "slot_t": 0, "support_lifted": 0}

    for row_index, (h, r, t) in enumerate(generator_triples):
        # STEP 2: pick the corrupted slot, 50/50 head or tail.
        slot = 2 if rng.random() < 0.5 else 0
        if slot == 0:                                  # corrupt the HEAD
            clean_value, self_row, anchor = h, t, t
            banned = true_heads.get((r, t))
            pool_row = pool_masks[0, r]
        else:                                          # corrupt the TAIL
            clean_value, self_row, anchor = t, h, h
            banned = true_tails.get((h, r))
            pool_row = pool_masks[1, r]

        # STEP 3: score the relation's full type pool with the generator.
        pool_ids = torch.nonzero(pool_row).flatten()
        if len(pool_ids) == 0:
            pool_ids = torch.arange(n_ent)
        with torch.no_grad():
            head_id = torch.tensor([h], device=device)
            relation_id = torch.tensor([r], device=device)
            tail_id = torch.tensor([t], device=device)
            pool_scores = generator(
                head_id, relation_id, tail_id, entity_context,
                sketches[[anchor]].to(device),
                pool_ids.unsqueeze(0).to(device),
                torch.zeros(1, len(pool_ids), device=device), slot)[0]
        # Scatter pool scores into a full-vocabulary vector (everything else
        # -inf) so the mask ladder applies uniformly.
        full_scores = torch.full((n_ent,), float("-inf"), device=device)
        full_scores[pool_ids.to(device)] = pool_scores

        support_ban = (support_ban_row(payload, anchor, support_max)
                       if support_max is not None else None)

        # STEPS 4-6: mask, sample, check; bounded redraws.
        neg_h, neg_r, neg_t = h, r, t
        emitted = False
        row_lifted = False
        for _ in range(max_resample):
            new_index, lifted = _sample_replacement_index(
                full_scores, clean_value, torch_rng, banned=banned,
                pool_row=pool_row.to(device), self_row=self_row,
                support_ban=support_ban)
            row_lifted = row_lifted or lifted
            if new_index < 0:
                break
            candidate = ((new_index, r, t) if slot == 0
                         else (h, r, new_index))
            if (candidate[0] != candidate[2]
                    and candidate not in real_triple_set):
                neg_h, neg_r, neg_t = candidate
                emitted = True
                if bool(pool_row[new_index]):
                    stats["type_valid"] += 1
                break
            stats["resampled"] += 1

        if not emitted:
            stats["used_original"] += 1
            stats["null_indices"].append(row_index)
        if row_lifted:
            stats["support_lifted"] += 1
        stats["slot_h" if slot == 0 else "slot_t"] += 1

        # STEP 7: generator ids -> strings -> caller ids.
        negatives.append((id_maps["ent2id"][id2ent_gen[neg_h]],
                          id_maps["rel2id"][id2rel_gen[neg_r]],
                          id_maps["ent2id"][id2ent_gen[neg_t]]))
        stats["processed"] += 1

    return negatives, stats


def render_stats(stats):
    """Human-readable one-line summary of one batch of generation."""
    total = stats["slot_h"] + stats["slot_r"] + stats["slot_t"]
    if total > 0:
        slot_pct = (
            f"head={stats['slot_h']}/{total}({stats['slot_h']/total:.1%}) "
            f"rel={stats['slot_r']}/{total}({stats['slot_r']/total:.1%}) "
            f"tail={stats['slot_t']}/{total}({stats['slot_t']/total:.1%})"
        )
    else:
        slot_pct = "no slots"
    processed = stats["processed"]
    used = stats["used_original"]
    fail_pct = (f"{used:,}/{processed:,}({used / processed:.1%})"
                if processed else "n/a")
    extras = ""
    if "type_valid" in stats and processed:
        extras = (f"  type_valid={stats['type_valid']:,}/{processed:,}"
                  f"({stats['type_valid'] / processed:.1%})"
                  f"  resampled={stats.get('resampled', 0):,}")
    return (
        f"processed={processed:,}  "
        f"used_original(gen_failed)={fail_pct}{extras}  "
        f"slot_distribution: {slot_pct}"
    )
