"""Use a trained KGSAGE GAN checkpoint to produce one negative per input triple.

This is the public generation API for the KGSAGE package. A downstream
detector calls it (via a bridge such as kgsage_bridge.bridge) every time it
builds a training batch of negatives (e.g. ADKGD's `--neg_source gan`).
Everything stays in-process — no intermediate file.

The 7-step pipeline (one negative per real triple):

  STEP 1: Translate the caller's integer IDs -> strings -> GAN integer IDs.
          (the caller and the GAN may number the same entity differently; strings
           are the lingua franca that keeps both worlds aligned.)
  STEP 2: Run the generator forward to get 3 logit vectors, CONDITIONED on the
          cached context table E' (loaded from the checkpoint — no PyG needed).
          (one over entities for the new head, one over relations for the new
           rel, one over entities for the new tail. These are PROBABILITIES,
           not picks yet.)
  STEP 3: GAN-CHOSEN slot. Score how confidently the generator can corrupt each
          slot (softmax mass on its best WRONG value). The RELATION slot is
          SKIPPED — a fixed head+tail rarely admits a coherent alternative
          relation, so relation corruptions are the weak, type-incoherent ones
          (confirmed empirically in downstream-detector logs). Among the two ENTITY slots the GAN picks
          head vs tail, sampled proportional to the score — so the GAN, not a
          coin flip, decides WHERE to corrupt. (Deliberately no longer matches
          the random baseline's uniform 3-slot distribution.)
  STEP 4: Mask the chosen slot's logits: the original index, every KNOWN-TRUE
          filler of the query across all splits (1-N safe), the self-loop
          entity, and -- on checkpoints that carry `pool_masks` --
          everything outside the relation's train-split type pool. Then sample
          via argmax + Gumbel noise (seeded torch.Generator derived from the
          caller's numpy rng, so generation is reproducible per seed).
  STEP 5: Build the candidate triple by gluing the new value into the slot.
          Up to `max_resample` (default 8) redraws if the residual validity
          check fails.
  STEP 6: A candidate surviving the checks is emitted. If every redraw FAILS
          (degenerate row), use the ORIGINAL triple (a null corruption), count
          it in `used_original` AND record its position in `null_indices` so
          callers can drop/replace it -- a null is a real fact and must never
          be trained on as a negative. There is NO hidden random fallback.
  STEP 7: Translate GAN integer IDs back to the caller's integer IDs via strings.
"""
import numpy as np
import torch


def load_checkpoint(ckpt_path, device=None):
    """Reconstruct the trained generator + helpers from a .pt file.

    Only candidate_v2 payloads (CandidateScoringGenerator + sketches) are
    loadable; the decode scores the FULL type pool per row. Legacy v1
    checkpoints cannot be decoded any more, but they remain valid as
    --init_context_from E' donors for the trainer (which reads the payload
    tensors directly and never calls this function).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    if payload.get("arch") == "candidate_v2":
        return _load_checkpoint_v2(payload, ckpt_path, device)

    raise ValueError(
        f"Checkpoint {ckpt_path!r} is not a candidate_v2 payload "
        f"(arch={payload.get('arch')!r}). Legacy v1 checkpoints are no longer "
        "loadable; use one of the canonical generator_*.pt artifacts or "
        "retrain with `python -m kgsage.gan.train`."
    )


def _load_checkpoint_v2(payload, ckpt_path, device):
    """candidate_v2 loader: the one and only payload contract."""
    from kgsage.gan.generator import CandidateScoringGenerator

    G = CandidateScoringGenerator(
        dim=payload["dim"], sketch_bits=payload["sketch_bits"],
        n_rel=payload["n_rel"]).to(device)
    G.load_state_dict(payload["generator_state"])
    G.eval()

    entity_context = payload["context_embeddings"].to(device)
    real_triple_set = set(tuple(t) for t in payload["real_triples"])
    true_tails, true_heads = {}, {}
    for h, r, t in real_triple_set:
        true_tails.setdefault((h, r), []).append(t)
        true_heads.setdefault((r, t), []).append(h)

    pool_masks = payload.get("pool_masks")
    if pool_masks is None:
        # early smoke checkpoints predate pool storage: derive from the
        # all-splits triples (slightly more permissive than train-only pools)
        print(f"[v2] {ckpt_path}: no pool_masks in payload -- deriving from "
              "real_triples (all splits)", flush=True)
        pool_masks = torch.zeros(2, payload["n_rel"], payload["n_ent"],
                                 dtype=torch.bool)
        for h, r, t in real_triple_set:
            pool_masks[0, r, h] = True
            pool_masks[1, r, t] = True

    return {
        "arch": "candidate_v2",
        "generator": G,
        "device": device,
        "entity_context": entity_context,
        "sketches": payload["sketches"].float(),
        "ent2id": payload["ent2id"], "rel2id": payload["rel2id"],
        "id2ent": payload["id2ent"], "id2rel": payload["id2rel"],
        "real_triple_set": real_triple_set,
        "true_tails": true_tails, "true_heads": true_heads,
        "pool_masks": pool_masks,
        "n_ent": payload["n_ent"], "n_rel": payload["n_rel"],
        "tau": payload.get("tau", 0.5),
    }


def support_ban_row(payload, anchor, support_max=0):
    """Bool [n_ent] torch row: candidates SUPPORTED by `anchor`'s neighbourhood.

    A candidate x is supported when the anchor's surroundings corroborate it:
      - x is a direct (1-hop) neighbour of the anchor, or
      - x shares more than `support_max` neighbours with the anchor
        (equivalently: >support_max two-hop paths anchor -> m -> x).

    The INVERTED-SUPPORT MASK bans exactly these, so an emitted corruption is
    contradicted by the anchor's neighbourhood BY CONSTRUCTION ("no one around
    this entity points at the replacement") instead of relying on the learned
    logits to be context-aware. support_max=0 is the strict reading: a single
    shared neighbour already counts as corroboration.

    The undirected adjacency is built once per payload from the checkpoint's
    all-splits real_triple_set and cached; per-anchor rows are also cached.
    """
    import scipy.sparse as sp

    cache = payload.setdefault("_support_cache", {})
    key = (anchor, support_max)
    if key in cache:
        return cache[key]

    adj = payload.get("_support_adj")
    if adj is None:
        n = payload["n_ent"]
        rows, cols = [], []
        for h, _, t in payload["real_triple_set"]:
            rows.append(h); cols.append(t)
            rows.append(t); cols.append(h)
        adj = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n, n))
        adj.data[:] = 1                       # collapse parallel edges to 0/1
        payload["_support_adj"] = adj

    nbr = adj.getrow(anchor)                  # 1 x n: N(anchor)
    shared = nbr @ adj                        # 1 x n: |N(anchor) ∩ N(x)|
    ban_np = (nbr.toarray()[0] > 0) | (shared.toarray()[0] > support_max)
    ban = torch.from_numpy(ban_np).to(payload["device"])
    if len(cache) < 20000:
        cache[key] = ban
    return ban


def _pick_new_index_with_noise(logits, clean_index, torch_gen,
                               banned=None, pool_row=None, self_row=None,
                               support_ban=None):
    """Implements STEP 4: mask, then Gumbel-sample a new index.

    Masks applied (each optional beyond the original value):
      clean_index : the true value -- forces the slot to move
      banned      : every known-true filler of this query (all splits, 1-N safe)
      self_row    : the triple's other entity (self-loop ban)
      pool_row    : bool [n_ent] type pool (when present) -- -inf
                    outside the relation's observed slot fillers
      support_ban : bool [n_ent] inverted-support mask (when present) -- -inf
                    on every candidate the anchor's neighbourhood corroborates
                    (see support_ban_row), so the pick is neighbourhood-
                    contradicting by construction
    Sampling adds Gumbel noise at temperature 0.5 (mostly-argmax) drawn from
    the caller's seeded torch.Generator -- reproducible per seed and per
    subprocess, unlike the old global-RNG draw.

    Degenerate rows relax masks in order: lift support_ban first, then the
    type pool (matching the old behaviour) -- the correctness bans (true value,
    known-true, self-loop) are never lifted.

    Returns (index, lifted_support) where index is -1 if nothing is sampleable
    and lifted_support flags that the support mask had to be dropped.
    """
    def _base():
        m = logits.clone()
        m[clean_index] = float("-inf")
        if banned:
            m[banned] = float("-inf")
        if self_row is not None:
            m[self_row] = float("-inf")
        return m

    lifted_support = False
    masked = _base()
    if pool_row is not None:
        masked[~pool_row] = float("-inf")
    if support_ban is not None:
        masked[support_ban] = float("-inf")
        if torch.isinf(masked).all():          # degenerate: lift support first
            lifted_support = True
            masked = _base()
            if pool_row is not None:
                masked[~pool_row] = float("-inf")
    if pool_row is not None and torch.isinf(masked).all():
        masked = _base()                       # degenerate pool: lift pool ban
        if support_ban is not None and not lifted_support:
            masked[support_ban] = float("-inf")
            if torch.isinf(masked).all():
                lifted_support = True
                masked = _base()
    if torch.isinf(masked).all():
        return -1, lifted_support
    u = torch.empty_like(masked)
    u.uniform_(generator=torch_gen).clamp_(1e-10, 1.0 - 1e-10)
    gumbel = -torch.log(-torch.log(u))
    return int((masked + gumbel * 0.5).argmax().item()), lifted_support


def generate_negatives(triples, payload, id_maps, rng=None,
                       batch_size=256, max_resample=8, support_max=None):
    """Generate one negative per input triple. Main entry point.

    triples       : list of (h, r, t) in the caller's integer ID space
    payload       : the dict returned by load_checkpoint()
    id_maps       : dict with 'id2ent', 'id2rel', 'ent2id', 'rel2id' from
                    the caller's id vocabulary (round-trip via strings)
    rng           : numpy random.Generator (per-caller seeded for reproducibility)
    max_resample  : bounded redraws before a row degrades to a null corruption
    support_max   : None = off (original behaviour). Integer >= 0 turns on the
                    INVERTED-SUPPORT MASK: candidates corroborated by the
                    anchor entity's neighbourhood (direct neighbours, or more
                    than support_max shared neighbours) are unsampleable, so
                    every emitted corruption is neighbourhood-contradicting by
                    construction. The anchor is the entity that KEEPS its slot
                    (head for a tail corruption and vice versa). Degenerate
                    rows lift this mask first (counted in stats).

    Returns: (negatives_list, stats_dict). stats['null_indices'] lists the
    positions whose emitted 'negative' is the original triple -- callers
    training on these negatives must drop or replace those rows.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if payload.get("arch") == "candidate_v2":
        return _generate_negatives_v2(triples, payload, id_maps, rng,
                                      max_resample, support_max)

    raise ValueError(
        "generate_negatives requires a candidate_v2 payload; legacy v1 "
        "checkpoints are no longer supported."
    )


def _generate_negatives_v2(triples, payload, id_maps, rng, max_resample,
                           support_max):
    """candidate_v2 decode: score the FULL type pool per row, then run the
    shared mask ladder (_pick_new_index_with_noise -- type pool,
    known-true, self, optional inverted-support) by scattering pool scores
    into an n_ent-sized vector. Same return contract as the v1 path."""
    G = payload["generator"]
    device = payload["device"]
    entity_context = payload["entity_context"]
    sketches = payload["sketches"]
    ent2id_gan, rel2id_gan = payload["ent2id"], payload["rel2id"]
    id2ent_gan, id2rel_gan = payload["id2ent"], payload["id2rel"]
    real_triple_set = payload["real_triple_set"]
    true_tails = payload.get("true_tails", {})
    true_heads = payload.get("true_heads", {})
    pool_masks = payload["pool_masks"]
    n_ent = payload["n_ent"]

    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(int(rng.integers(0, 2**31 - 1)))

    gan_triples = []
    for h_ext, r_ext, t_ext in triples:
        gan_triples.append((ent2id_gan[id_maps["id2ent"][h_ext]],
                            rel2id_gan[id_maps["id2rel"][r_ext]],
                            ent2id_gan[id_maps["id2ent"][t_ext]]))

    out = []
    stats = {"processed": 0, "used_original": 0, "null_indices": [],
             "resampled": 0, "type_valid": 0, "slot_h": 0, "slot_r": 0,
             "slot_t": 0, "support_lifted": 0}

    for i, (h_gan, r_gan, t_gan) in enumerate(gan_triples):
        slot = 2 if rng.random() < 0.5 else 0          # 50/50 head-tail
        if slot == 0:
            clean_value, self_row, anchor = h_gan, t_gan, t_gan
            banned = true_heads.get((r_gan, t_gan))
            pool_row = pool_masks[0, r_gan]
        else:
            clean_value, self_row, anchor = t_gan, h_gan, h_gan
            banned = true_tails.get((h_gan, r_gan))
            pool_row = pool_masks[1, r_gan]

        pool_ids = torch.nonzero(pool_row).flatten()
        if len(pool_ids) == 0:
            pool_ids = torch.arange(n_ent)
        with torch.no_grad():
            hi = torch.tensor([h_gan], device=device)
            ri = torch.tensor([r_gan], device=device)
            ti = torch.tensor([t_gan], device=device)
            lg = G(hi, ri, ti, entity_context,
                   sketches[[anchor]].to(device),
                   pool_ids.unsqueeze(0).to(device),
                   torch.zeros(1, len(pool_ids), device=device), slot)[0]
        # scatter pool scores into full-vocab space; everything else -inf, so
        # the shared mask ladder applies verbatim (incl. inverted support).
        full = torch.full((n_ent,), float("-inf"), device=device)
        full[pool_ids.to(device)] = lg

        sup_ban = (support_ban_row(payload, anchor, support_max)
                   if support_max is not None else None)

        neg_h, neg_r, neg_t = h_gan, r_gan, t_gan
        emitted = False
        row_lifted = False
        for _ in range(max_resample):
            new_idx, lifted = _pick_new_index_with_noise(
                full, clean_value, torch_gen, banned=banned,
                pool_row=pool_row.to(device), self_row=self_row,
                support_ban=sup_ban)
            row_lifted = row_lifted or lifted
            if new_idx < 0:
                break
            candidate = ((new_idx, r_gan, t_gan) if slot == 0
                         else (h_gan, r_gan, new_idx))
            if candidate[0] != candidate[2] and candidate not in real_triple_set:
                neg_h, neg_r, neg_t = candidate
                emitted = True
                if bool(pool_row[new_idx]):
                    stats["type_valid"] += 1
                break
            stats["resampled"] += 1

        if not emitted:
            stats["used_original"] += 1
            stats["null_indices"].append(i)
        if row_lifted:
            stats["support_lifted"] += 1
        stats["slot_h" if slot == 0 else "slot_t"] += 1
        out.append((id_maps["ent2id"][id2ent_gan[neg_h]],
                    id_maps["rel2id"][id2rel_gan[neg_r]],
                    id_maps["ent2id"][id2ent_gan[neg_t]]))
        stats["processed"] += 1

    return out, stats


def render_stats(stats):
    """Human-readable summary of one batch of generation."""
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
    fail_pct = f"{used:,}/{processed:,}({used / processed:.1%})" if processed else "n/a"
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
