"""Use a trained KGSAGE GAN checkpoint to produce one negative per input triple.

This is the public generation API for the KGSAGE package. ADKGD calls it
(via kgsage_bridge.bridge) every time it builds a training batch with
`--neg_source gan`. Everything stays in-process — no intermediate file.

The 8-step pipeline (one negative per real triple):

  STEP 1: Translate ADKGD integer IDs -> strings -> GAN integer IDs.
          (ADKGD and the GAN may number the same entity differently; strings
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
          (confirmed in ADKGD logs). Among the two ENTITY slots the GAN picks
          head vs tail, sampled proportional to the score — so the GAN, not a
          coin flip, decides WHERE to corrupt. (Deliberately no longer matches
          the random baseline's uniform 3-slot distribution.)
  STEP 4: Mask the chosen slot's logits: the original index, every KNOWN-TRUE
          filler of the query across all splits (1-N safe), the self-loop
          entity, and -- on A-ii checkpoints that carry `pool_masks` --
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
  STEP 7: Translate GAN integer IDs back to ADKGD integer IDs via strings.
"""
import numpy as np
import torch

from kgsage.gan.models import KGSAGEGenerator


def load_checkpoint(ckpt_path, device=None):
    """Reconstruct the trained generator + helpers from a .pt file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    # B1a checkpoints cache the RGCN context table E' so inference can condition
    # the generator without ever running the encoder (or importing PyG).
    if "context_embeddings" not in payload:
        raise KeyError(
            f"Checkpoint {ckpt_path!r} has no 'context_embeddings' (E'). It looks "
            "like an old pre-B1a checkpoint. Retrain with `python -m "
            "kgsage.cli.train_gan ...` — the current pipeline caches E' automatically."
        )

    gen_kwargs = dict(
        n_ent=payload["n_ent"],
        n_rel=payload["n_rel"],
        dim=payload["dim"],
        z_dim=payload["z_dim"],
    )
    if "hidden" in payload:
        gen_kwargs["hidden"] = payload["hidden"]
    G = KGSAGEGenerator(**gen_kwargs).to(device)
    G.load_state_dict(payload["generator_state"])
    G.eval()

    # Cached context table E' [n_ent, dim]. Move to `device` ONCE here; the
    # generator conditions on it every forward. This is the "PyG-free" firewall.
    entity_context = payload["context_embeddings"].to(device)

    # The set of real triples (in the GAN's ID space) for collision filtering.
    real_triple_set = set(tuple(t) for t in payload["real_triples"])

    # A5: known-true filler bans per query direction (1-N safe masking at
    # decode time -- collisions become structurally impossible, not retried).
    true_tails, true_heads = {}, {}
    for h, r, t in real_triple_set:
        true_tails.setdefault((h, r), []).append(t)
        true_heads.setdefault((r, t), []).append(h)

    return {
        "generator": G,
        "device": device,
        "entity_context": entity_context,
        "ent2id": payload["ent2id"],
        "rel2id": payload["rel2id"],
        "id2ent": payload["id2ent"],
        "id2rel": payload["id2rel"],
        "real_triple_set": real_triple_set,
        "true_tails": true_tails,
        "true_heads": true_heads,
        # A-ii checkpoints carry the train-split type pools; legacy ones don't.
        "pool_masks": payload.get("pool_masks"),
        "n_ent": payload["n_ent"],
        "n_rel": payload["n_rel"],
        "z_dim": payload["z_dim"],
    }


def _pick_new_index_with_noise(logits, clean_index, torch_gen,
                               banned=None, pool_row=None, self_row=None):
    """Implements STEP 4: mask, then Gumbel-sample a new index.

    Masks applied (each optional beyond the original value):
      clean_index : the true value -- forces the slot to move
      banned      : every known-true filler of this query (all splits, 1-N safe)
      self_row    : the triple's other entity (self-loop ban)
      pool_row    : bool [n_ent] type pool (A-ii checkpoints only) -- -inf
                    outside the relation's observed slot fillers
    Sampling adds Gumbel noise at temperature 0.5 (mostly-argmax) drawn from
    the caller's seeded torch.Generator -- reproducible per seed and per
    subprocess, unlike the old global-RNG draw.

    Returns -1 if masking left no candidate (caller falls back to null).
    """
    masked = logits.clone()
    masked[clean_index] = float("-inf")
    if banned:
        masked[banned] = float("-inf")
    if self_row is not None:
        masked[self_row] = float("-inf")
    if pool_row is not None:
        masked[~pool_row] = float("-inf")
        if torch.isinf(masked).all():          # degenerate pool: lift pool ban
            masked = logits.clone()
            masked[clean_index] = float("-inf")
            if banned:
                masked[banned] = float("-inf")
            if self_row is not None:
                masked[self_row] = float("-inf")
    if torch.isinf(masked).all():
        return -1
    u = torch.empty_like(masked)
    u.uniform_(generator=torch_gen).clamp_(1e-10, 1.0 - 1e-10)
    gumbel = -torch.log(-torch.log(u))
    return int((masked + gumbel * 0.5).argmax().item())


def _corruptibility_scores(head_logits, rel_logits, tail_logits, h_in, r_in, t_in):
    """Implements STEP 3's per-slot score: the generator's confidence in its best
    WRONG value for each slot.

    For each triple and slot, this is the softmax mass the generator puts on its
    top NON-true value. High = the generator confidently prefers a wrong value
    (a good slot to corrupt); low = it believes the true value, or is unsure
    (e.g. no coherent relation exists) — so that slot is rarely chosen. Trained
    on random relation targets, the relation head stays comparatively flat, which
    is exactly what steers corruption away from the weak relation slot.

    Returns a [n, 3] tensor, columns = (head, relation, tail).
    """
    def best_wrong(logits, true_idx):
        probs = torch.softmax(logits, dim=1)
        probs.scatter_(1, true_idx.unsqueeze(1), 0.0)   # drop the true value's mass
        return probs.max(dim=1).values                  # top remaining (wrong) value
    return torch.stack([
        best_wrong(head_logits, h_in),
        best_wrong(rel_logits, r_in),
        best_wrong(tail_logits, t_in),
    ], dim=1)


def generate_negatives(adkgd_triples, payload, adkgd_maps, rng=None,
                       batch_size=256, max_resample=8):
    """Generate one negative per input triple. Main entry point.

    adkgd_triples : list of (h, r, t) in ADKGD's integer ID space
    payload       : the dict returned by load_checkpoint()
    adkgd_maps    : dict with 'id2ent', 'id2rel', 'ent2id', 'rel2id' from
                    ADKGD's Reader (round-trip via strings)
    rng           : numpy random.Generator (per-Reader seeded for reproducibility)
    max_resample  : bounded redraws before a row degrades to a null corruption

    Returns: (negatives_list, stats_dict). stats['null_indices'] lists the
    positions whose emitted 'negative' is the original triple -- callers
    training on these negatives must drop or replace those rows.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    # ALL torch randomness (z + Gumbel) flows from this generator, which is
    # derived from the caller's numpy rng -- the whole call is reproducible
    # per seed with no dependence on global torch RNG state.
    torch_gen = torch.Generator()
    torch_gen.manual_seed(int(rng.integers(0, 2**31 - 1)))

    G = payload["generator"]
    device = payload["device"]
    entity_context = payload["entity_context"]  # cached E' [n_ent, dim]
    ent2id_gan = payload["ent2id"]
    rel2id_gan = payload["rel2id"]
    id2ent_gan = payload["id2ent"]
    id2rel_gan = payload["id2rel"]
    real_triple_set = payload["real_triple_set"]
    true_tails = payload.get("true_tails", {})
    true_heads = payload.get("true_heads", {})
    pool_masks = payload.get("pool_masks")      # [2, n_rel, n_ent] bool or None
    z_dim = payload["z_dim"]

    # STEP 1: Translate ADKGD IDs -> strings -> GAN IDs (once, up front).
    gan_triples = []
    for h_adk, r_adk, t_adk in adkgd_triples:
        h_s = adkgd_maps["id2ent"][h_adk]
        r_s = adkgd_maps["id2rel"][r_adk]
        t_s = adkgd_maps["id2ent"][t_adk]
        gan_triples.append((ent2id_gan[h_s], rel2id_gan[r_s], ent2id_gan[t_s]))

    out = []
    stats = {
        "processed": 0,
        "used_original": 0,   # generation failed -> kept original (see null_indices)
        "null_indices": [],   # positions of null corruptions in the output
        "resampled": 0,       # extra draws consumed by the bounded retry loop
        "type_valid": 0,      # emitted negatives inside the relation's type pool
        "slot_h": 0,
        "slot_r": 0,
        "slot_t": 0,
    }

    for batch_start in range(0, len(gan_triples), batch_size):
        batch = gan_triples[batch_start:batch_start + batch_size]
        n = len(batch)

        # STEP 2: Generator forward pass on the whole batch.
        h_in = torch.tensor([row[0] for row in batch], dtype=torch.long, device=device)
        r_in = torch.tensor([row[1] for row in batch], dtype=torch.long, device=device)
        t_in = torch.tensor([row[2] for row in batch], dtype=torch.long, device=device)

        with torch.no_grad():
            z = torch.empty(n, z_dim, device=device).normal_(generator=torch_gen)
            # Condition on the cached context table E' (STEP 2).
            head_logits, rel_logits, tail_logits = G(h_in, r_in, t_in, z, entity_context)

        # STEP 3: GAN-chosen slot. Score how confidently the generator can corrupt
        # each slot (its best wrong value; see _corruptibility_scores), then choose
        # WHERE to corrupt. The relation slot is skipped: a fixed head+tail rarely
        # admits a coherent alternative relation, so those are the weak, type-
        # incoherent corruptions. The GAN picks which ENTITY slot (head or tail),
        # weighted by its confidence there.
        with torch.no_grad():
            scores = _corruptibility_scores(
                head_logits, rel_logits, tail_logits, h_in, r_in, t_in).cpu().numpy()
        entity_scores = scores[:, [0, 2]] + 1e-9        # columns: head, tail
        p_tail = entity_scores[:, 1] / entity_scores.sum(axis=1)
        slots = np.where(rng.random(n) < p_tail, 2, 0)  # 2 = tail, 0 = head (never 1 = rel)

        for i in range(n):
            h_gan, r_gan, t_gan = batch[i]
            slot = int(slots[i])

            # STEP 4: slot logits + the full mask set for this query.
            if slot == 0:      # head
                logits_for_slot = head_logits[i]
                clean_value, self_row = h_gan, t_gan
                banned = true_heads.get((r_gan, t_gan))
                pool_row = pool_masks[0, r_gan] if pool_masks is not None else None
            else:              # tail (relation slot is never chosen; see STEP 3)
                logits_for_slot = tail_logits[i]
                clean_value, self_row = t_gan, h_gan
                banned = true_tails.get((h_gan, r_gan))
                pool_row = pool_masks[1, r_gan] if pool_masks is not None else None

            # STEP 5: masked Gumbel pick with a bounded resample loop. With the
            # known-true + self masks a collision is structurally impossible;
            # the residual check guards edge cases (e.g. lifted degenerate pool).
            neg_h, neg_r, neg_t = h_gan, r_gan, t_gan   # null default
            emitted = False
            for attempt in range(max_resample):
                new_idx = _pick_new_index_with_noise(
                    logits_for_slot, clean_value, torch_gen,
                    banned=banned, pool_row=pool_row, self_row=self_row)
                if new_idx < 0:
                    break                                  # nothing sampleable
                candidate = ((new_idx, r_gan, t_gan) if slot == 0
                             else (h_gan, r_gan, new_idx))
                if candidate[0] != candidate[2] and candidate not in real_triple_set:
                    neg_h, neg_r, neg_t = candidate
                    emitted = True
                    if pool_row is not None and bool(pool_row[new_idx]):
                        stats["type_valid"] += 1
                    break
                stats["resampled"] += 1

            # STEP 6: every redraw failed -> null corruption, flagged for the
            # caller (a null is a real fact; it must never train as a negative).
            if not emitted:
                stats["used_original"] += 1
                stats["null_indices"].append(batch_start + i)

            if slot == 0:
                stats["slot_h"] += 1
            elif slot == 1:
                stats["slot_r"] += 1
            else:
                stats["slot_t"] += 1

            # STEP 7: Translate GAN IDs -> strings -> ADKGD IDs.
            out.append((
                adkgd_maps["ent2id"][id2ent_gan[neg_h]],
                adkgd_maps["rel2id"][id2rel_gan[neg_r]],
                adkgd_maps["ent2id"][id2ent_gan[neg_t]],
            ))
        stats["processed"] += n

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
