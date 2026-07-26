"""Phase 3 — Corruption Generation (the paper's final pipeline phase).

Loads a trained KGSAGE checkpoint and produces one corruption per input
triple. This is the public generation API of the package — a downstream
detector calls it (through a bridge such as kgsage_bridge.bridge) every time
it needs a batch of negatives. Everything stays in-process; no intermediate
files. PyG is never needed here: the checkpoint carries the frozen context
table E' and the membership sketches.

Vocabulary note: inside kgsage/ the emitted false triple is always a
CORRUPTION. The bridge is what re-labels it a "negative" (training) or an
"anomaly" (evaluation) for the detector — that is a deliberate seam, not two
names for one thing drifting apart. See experiments/docs/KGSAGE_glossary.md.

The pipeline, one corruption per real triple:

  STEP 1  Translate the caller's integer ids -> strings -> the generator's
          integer ids (the caller and the generator may number the same
          entity differently; strings are the shared language).
  STEP 2  Pick the slot to corrupt: head or tail, 50/50. The entity that
          keeps its slot is the ANCHOR; the value being replaced is the
          TRUE_FILLER. The relation slot is never corrupted — with head and
          tail fixed there is rarely a coherent alternative relation, so
          relation corruptions come out type-incoherent (confirmed in
          downstream-detector logs).
  STEP 3  Score every CANDIDATE in the relation's FULL type pool with the
          generator, conditioned on the frozen E' rows of the triple and the
          anchor's membership sketch.
  STEP 4  Mask out every candidate that must never be picked: the
          true_filler, every KNOWN-TRUE filler of the query across ALL
          splits (this is the falseness guarantee), the other_entity of the
          triple (self-loop), everything outside the relation's type pool,
          and — optionally — every candidate the anchor's neighbourhood
          corroborates (the corroboration mask, see
          corroborated_entities_mask).
  STEP 5  Choose the PICKED_CANDIDATE: argmax over the masked scores plus
          Gumbel noise, drawn from a torch.Generator seeded from the
          caller's numpy rng — reproducible per seed.
  STEP 6  Check the picked_candidate (no self-loop, not a real fact); up to
          `max_resample` redraws. If every redraw fails, the ORIGINAL triple
          is emitted as a null corruption, counted in `used_original` and
          listed in `null_indices` — a null is a real fact and callers must
          drop or replace it before training on it. There is NO hidden
          random fallback.
  STEP 7  Translate the corruption back to the caller's integer ids.
"""
import numpy as np
import torch


def load_checkpoint(ckpt_path, device=None):
    """Reconstruct the trained generator and its lookup tables from a .pt file.

    Only candidate_v2 payloads (CandidateScoringGenerator + membership
    sketches) are loadable. Older checkpoints remain useful only as
    --init_context_from E' donors for the trainer, which reads their tensors
    directly and never calls this function.

    `load_checkpoint` is a frozen public name: the bridge re-exports it as
    load_gan for repo-root dataset.py.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    # frozen arch literal; "candidate_v2" = the dual-discriminator architecture
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

    `saved` holds the ON-DISK payload keys, which are frozen: renaming any of
    them would make every archived .pt unloadable. The runtime dict this
    returns uses glossary vocabulary instead.
    """
    from kgsage.gan.generator import CandidateScoringGenerator

    generator = CandidateScoringGenerator(
        dim=saved["dim"], sketch_bits=saved["sketch_bits"],
        n_rel=saved["n_rel"]).to(device)
    # frozen payload key; generator_state = the trained G's state_dict
    generator.load_state_dict(saved["generator_state"])
    generator.eval()

    # frozen payload key; context_embeddings = the frozen context table E'
    # (the RGCN encoder's OUTPUT, not the entity_embeddings that fed into it)
    context_table = saved["context_embeddings"].to(device)

    # Known-true fillers per query, across ALL splits — the falseness bans.
    real_triple_set = set(tuple(t) for t in saved["real_triples"])
    true_tails, true_heads = {}, {}
    for h, r, t in real_triple_set:
        true_tails.setdefault((h, r), []).append(t)
        true_heads.setdefault((r, t), []).append(h)

    # frozen payload key; pool_masks = the per-relation type pools
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
        "context_table": context_table,
        # frozen payload key; sketches = the Bloom membership sketches
        "sketches": saved["sketches"].float(),
        "ent2id": saved["ent2id"], "rel2id": saved["rel2id"],
        "id2ent": saved["id2ent"], "id2rel": saved["id2rel"],
        "real_triple_set": real_triple_set,
        "true_tails": true_tails, "true_heads": true_heads,
        "pool_masks": pool_masks,
        "n_ent": saved["n_ent"], "n_rel": saved["n_rel"],
        "tau": saved.get("tau", 0.5),
    }


def corroborated_entities_mask(payload, anchor, support_max=0):
    """Bool [n_ent] row: candidates the anchor's neighbourhood CORROBORATES.

    A candidate x counts as corroborated when the anchor's surroundings vouch
    for it:
      - x is a direct (1-hop) neighbour of the anchor, or
      - x shares more than `support_max` neighbours with the anchor.

    `support_max` keeps its old spelling because it is a frozen public kwarg
    of generate_negatives; read it as "the corroboration tolerance".

    The CORROBORATION MASK bans exactly these, so an emitted corruption is
    contradicted by the anchor's neighbourhood BY CONSTRUCTION ("no one
    around this entity points at the picked_candidate") instead of relying on
    the learned scores alone. support_max=0 is the strict reading: one shared
    neighbour already counts as corroboration.

    The undirected adjacency is built once per payload from the checkpoint's
    all-splits triples and cached; per-anchor rows are cached too.
    """
    import scipy.sparse as sp

    cache = payload.setdefault("_corroboration_cache", {})
    cache_key = (anchor, support_max)
    if cache_key in cache:
        return cache[cache_key]

    adjacency = payload.get("_corroboration_adj")
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
        payload["_corroboration_adj"] = adjacency

    anchor_neighbours = adjacency.getrow(anchor)     # 1 x n: N(anchor)
    shared_counts = anchor_neighbours @ adjacency    # 1 x n: |N(anchor) ∩ N(x)|
    mask_numpy = ((anchor_neighbours.toarray()[0] > 0)
                  | (shared_counts.toarray()[0] > support_max))
    mask = torch.from_numpy(mask_numpy).to(payload["device"])
    if len(cache) < 20000:
        cache[cache_key] = mask
    return mask


def _pick_candidate_index(logits, true_filler_index, torch_rng,
                          banned=None, pool_row=None, other_entity=None,
                          corroboration_mask=None):
    """STEPS 4-5 for one row: apply the masks, then Gumbel-pick a candidate.

    Masks applied (each optional beyond the true_filler):
      true_filler_index  : the original value — forces the slot to change.
      banned             : every known-true filler of this query (all splits).
      other_entity       : the triple's other entity (self-loop ban).
      pool_row           : bool [n_ent] type pool — bans every candidate
                           outside the relation's observed slot fillers.
      corroboration_mask : bool [n_ent] — bans every candidate the anchor's
                           neighbourhood corroborates (see
                           corroborated_entities_mask), making the pick
                           neighbourhood-contradicting by construction.

    Picking adds Gumbel noise at temperature 0.5 (mostly argmax) drawn from
    the caller's seeded torch.Generator, so generation is reproducible.

    If a row has nothing left to pick, masks are relaxed in order: the
    corroboration mask is lifted first, then the type pool. The correctness
    bans (true_filler, known-true, self-loop) are NEVER lifted.

    Returns (index, lifted_corroboration): index is -1 if nothing is
    pickable; lifted_corroboration flags that the corroboration mask had to
    be dropped.
    """
    def correctness_masked():
        masked = logits.clone()
        masked[true_filler_index] = float("-inf")
        if banned:
            masked[banned] = float("-inf")
        if other_entity is not None:
            masked[other_entity] = float("-inf")
        return masked

    lifted_corroboration = False
    masked = correctness_masked()
    if pool_row is not None:
        masked[~pool_row] = float("-inf")
    if corroboration_mask is not None:
        masked[corroboration_mask] = float("-inf")
        if torch.isinf(masked).all():     # degenerate: lift corroboration first
            lifted_corroboration = True
            masked = correctness_masked()
            if pool_row is not None:
                masked[~pool_row] = float("-inf")
    if pool_row is not None and torch.isinf(masked).all():
        masked = correctness_masked()          # degenerate pool: lift pool ban
        if corroboration_mask is not None and not lifted_corroboration:
            masked[corroboration_mask] = float("-inf")
            if torch.isinf(masked).all():
                lifted_corroboration = True
                masked = correctness_masked()
    if torch.isinf(masked).all():
        return -1, lifted_corroboration

    uniform_noise = torch.empty_like(masked)
    uniform_noise.uniform_(generator=torch_rng).clamp_(1e-10, 1.0 - 1e-10)
    gumbel_noise = -torch.log(-torch.log(uniform_noise))
    return (int((masked + gumbel_noise * 0.5).argmax().item()),
            lifted_corroboration)


def generate_negatives(triples, payload, id_maps, rng=None,
                       batch_size=256, max_resample=8, support_max=None):
    """Generate one corruption per input triple. Main entry point.

    The name `generate_negatives` is frozen — repo-root dataset.py reaches it
    through kgsage_bridge. Inside kgsage/ the emitted object is a CORRUPTION;
    the bridge is what presents it to the detector as a "negative".

    triples      : list of (h, r, t) in the CALLER's integer id space.
    payload      : the dict returned by load_checkpoint().
    id_maps      : dict with 'id2ent', 'id2rel', 'ent2id', 'rel2id' for the
                   caller's vocabulary (translation goes through strings).
    rng          : numpy random Generator, seeded by the caller.
    max_resample : redraws before a row degrades to a null corruption.
    support_max  : frozen public kwarg name for the corroboration tolerance.
                   None = off. An integer >= 0 turns on the CORROBORATION
                   MASK: candidates the anchor's neighbourhood corroborates
                   (direct neighbours, or more than support_max shared
                   neighbours) become unpickable, so every emitted corruption
                   contradicts the neighbourhood by construction. Degenerate
                   rows lift this mask first (counted in stats).

    Returns (corruptions, stats). stats['null_indices'] lists the positions
    whose emitted corruption is the original triple — callers training on
    these must drop or replace those rows.
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
    """The candidate_v2 decode: score every candidate in the relation's FULL
    type pool per row, scatter those scores into an n_ent-wide vector, then
    run the shared mask ladder (_pick_candidate_index)."""
    generator = payload["generator"]
    device = payload["device"]
    context_table = payload["context_table"]
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

    corruptions = []
    # used_original / null_indices are frozen stats keys; repo-root dataset.py
    # reads them through the bridge to find and replace null corruptions.
    stats = {"processed": 0, "used_original": 0, "null_indices": [],
             "resampled": 0, "type_valid": 0, "slot_h": 0, "slot_r": 0,
             "slot_t": 0, "corroboration_lifted": 0}

    for row_index, (h, r, t) in enumerate(generator_triples):
        # STEP 2: pick the corrupted slot, 50/50 head or tail. The anchor is
        # whichever entity keeps its slot; the true_filler is what we replace.
        slot = 2 if rng.random() < 0.5 else 0
        if slot == 0:                                  # corrupt the HEAD
            true_filler, other_entity, anchor = h, t, t
            banned = true_heads.get((r, t))
            pool_row = pool_masks[0, r]
        else:                                          # corrupt the TAIL
            true_filler, other_entity, anchor = t, h, h
            banned = true_tails.get((h, r))
            pool_row = pool_masks[1, r]

        # STEP 3: score every candidate in the relation's type pool.
        pool_ids = torch.nonzero(pool_row).flatten()
        if len(pool_ids) == 0:
            pool_ids = torch.arange(n_ent)
        with torch.no_grad():
            head_id = torch.tensor([h], device=device)
            relation_id = torch.tensor([r], device=device)
            tail_id = torch.tensor([t], device=device)
            pool_scores = generator(
                head_id, relation_id, tail_id, context_table,
                sketches[[anchor]].to(device),
                pool_ids.unsqueeze(0).to(device),
                torch.zeros(1, len(pool_ids), device=device), slot)[0]
        # Scatter candidate scores into a full-vocabulary vector (everything
        # else -inf) so the mask ladder applies uniformly.
        full_scores = torch.full((n_ent,), float("-inf"), device=device)
        full_scores[pool_ids.to(device)] = pool_scores

        corroboration_mask = (
            corroborated_entities_mask(payload, anchor, support_max)
            if support_max is not None else None)

        # STEPS 4-6: mask, pick, check; bounded redraws.
        # emit_* is the triple actually returned; it stays == (h, r, t) if
        # every redraw fails, which is what makes the row a null corruption.
        # (Deliberately NOT named corr_*: that prefix is reserved for the
        # frozen eval-CSV columns and the corr-pick= training-log token.)
        emit_h, emit_r, emit_t = h, r, t
        emitted = False
        row_lifted = False
        for _ in range(max_resample):
            picked_index, lifted = _pick_candidate_index(
                full_scores, true_filler, torch_rng, banned=banned,
                pool_row=pool_row.to(device), other_entity=other_entity,
                corroboration_mask=corroboration_mask)
            row_lifted = row_lifted or lifted
            if picked_index < 0:
                break
            proposed_corruption = ((picked_index, r, t) if slot == 0
                                   else (h, r, picked_index))
            if (proposed_corruption[0] != proposed_corruption[2]
                    and proposed_corruption not in real_triple_set):
                emit_h, emit_r, emit_t = proposed_corruption
                emitted = True
                if bool(pool_row[picked_index]):
                    stats["type_valid"] += 1
                break
            stats["resampled"] += 1

        if not emitted:
            stats["used_original"] += 1
            stats["null_indices"].append(row_index)
        if row_lifted:
            stats["corroboration_lifted"] += 1
        stats["slot_h" if slot == 0 else "slot_t"] += 1

        # STEP 7: generator ids -> strings -> caller ids.
        corruptions.append((id_maps["ent2id"][id2ent_gen[emit_h]],
                            id_maps["rel2id"][id2rel_gen[emit_r]],
                            id_maps["ent2id"][id2ent_gen[emit_t]]))
        stats["processed"] += 1

    return corruptions, stats


def render_stats(stats):
    """Human-readable one-line summary of one batch of generation.

    Frozen public name AND frozen output shape: repo-root dataset.py prints
    this line, so the field names below stay as they are for log
    comparability with runs already collected.
    """
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
