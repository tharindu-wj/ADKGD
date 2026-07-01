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
  STEP 4: Mask the original index in the chosen slot to -inf, then sample
          the new value via argmax + Gumbel noise.
          (Mask = "force the slot to move". Gumbel noise = "vary across calls"
           so we don't return the same negative every time.)
  STEP 5: Build the candidate triple by gluing the new value into the slot we
          picked. SINGLE shot — no retry loop.
  STEP 6: Keep the candidate only if it is a VALID negative (not a self-loop and
          not an existing real triple). If it FAILS, use the ORIGINAL triple (a
          null corruption) and count it in `used_original`. There is NO random
          fallback: we measure the generator's real failure rate rather than
          papering over it, so training reflects the model, not a safety net.
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

    return {
        "generator": G,
        "device": device,
        "entity_context": entity_context,
        "ent2id": payload["ent2id"],
        "rel2id": payload["rel2id"],
        "id2ent": payload["id2ent"],
        "id2rel": payload["id2rel"],
        "real_triple_set": real_triple_set,
        "n_ent": payload["n_ent"],
        "n_rel": payload["n_rel"],
        "z_dim": payload["z_dim"],
    }


def _pick_new_index_with_noise(logits, clean_index, rng):
    """Implements STEP 4: mask the original + Gumbel-sample a new index.

    Two things happen here:
      (a) MASK: set logits[clean_index] = -inf so argmax can never pick the
          original value. This is what forces the slot to actually move.
      (b) NOISE: add Gumbel noise before argmax. Same input -> different
          output across calls.
    """
    masked = logits.clone()
    masked[clean_index] = float("-inf")

    # Add Gumbel-distributed noise (Gumbel-Max trick). Temperature 0.5 =
    # mostly argmax but with some randomness.
    u = torch.rand_like(masked).clamp_(1e-10, 1.0 - 1e-10)
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
                       batch_size=256):
    """Generate one negative per input triple. Main entry point.

    adkgd_triples : list of (h, r, t) in ADKGD's integer ID space
    payload       : the dict returned by load_checkpoint()
    adkgd_maps    : dict with 'id2ent', 'id2rel', 'ent2id', 'rel2id' from
                    ADKGD's Reader (round-trip via strings)
    rng           : numpy random.Generator (per-Reader seeded for reproducibility)

    Returns: (negatives_list, stats_dict).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    G = payload["generator"]
    device = payload["device"]
    entity_context = payload["entity_context"]  # cached E' [n_ent, dim]
    ent2id_gan = payload["ent2id"]
    rel2id_gan = payload["rel2id"]
    id2ent_gan = payload["id2ent"]
    id2rel_gan = payload["id2rel"]
    real_triple_set = payload["real_triple_set"]
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
        "used_original": 0,   # generation failed (self-loop/collision) -> kept original
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
            z = torch.randn(n, z_dim, device=device)
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

            # STEP 4: select the logits + clean value for the chosen slot.
            if slot == 0:
                logits_for_slot = head_logits[i]
                clean_value = h_gan
            elif slot == 1:
                logits_for_slot = rel_logits[i]
                clean_value = r_gan
            else:
                logits_for_slot = tail_logits[i]
                clean_value = t_gan

            # STEP 5: single-shot pick (mask the true value, Gumbel-argmax). No
            # retry loop, no random fallback.
            new_idx = _pick_new_index_with_noise(logits_for_slot, clean_value, rng)
            if slot == 0:
                candidate = (new_idx, r_gan, t_gan)
            elif slot == 1:
                candidate = (h_gan, new_idx, t_gan)
            else:
                candidate = (h_gan, r_gan, new_idx)

            # STEP 6: keep it only if it is a valid negative (not a self-loop and
            # not an existing real triple). If it FAILS, use the ORIGINAL triple
            # (a null corruption) and record it, so the generator's real failure
            # rate is visible instead of hidden by a random fallback.
            if candidate[0] != candidate[2] and candidate not in real_triple_set:
                neg_h, neg_r, neg_t = candidate
            else:
                neg_h, neg_r, neg_t = h_gan, r_gan, t_gan
                stats["used_original"] += 1

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
    return (
        f"processed={processed:,}  "
        f"used_original(gen_failed)={fail_pct}  "
        f"slot_distribution: {slot_pct}"
    )
