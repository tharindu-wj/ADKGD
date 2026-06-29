"""Use a trained KGSAGE GAN checkpoint to produce one negative per input triple.

This is the public generation API for the KGSAGE package. ADKGD calls it
(via kgsage_bridge.bridge) every time it builds a training batch with
`--neg_source gan`. Everything stays in-process — no intermediate file.

The function `generate_negatives` is also exposed as `generate_contradictions`
for forward-compatibility with the Phase 2 pair-aware generator, which will
replace the simple per-slot corruption below with a role-swap partner head.

The 8-step pipeline (one negative per real triple):

  STEP 1: Translate ADKGD integer IDs -> strings -> GAN integer IDs.
          (ADKGD and the GAN may number the same entity differently; strings
           are the lingua franca that keeps both worlds aligned.)
  STEP 2: Run the generator forward to get 3 logit vectors.
          (one over entities for the new head, one over relations for the new
           rel, one over entities for the new tail. These are PROBABILITIES,
           not picks yet.)
  STEP 3: Pick which slot to corrupt uniformly at random {head, rel, tail}.
          (Matches the baseline's `random.randint(0, 2)` distribution exactly.)
  STEP 4: Mask the original index in the chosen slot to -inf, then sample
          the new value via argmax + Gumbel noise.
          (Mask = "force the slot to move". Gumbel noise = "vary across calls"
           so we don't return the same negative every time.)
  STEP 5: Build the candidate triple by gluing the new value into the slot
          we picked, keeping the other 2 slots from the original.
  STEP 6: Validate the candidate. If it's a self-loop (h == t) or already in
          the real graph, RETRY up to `max_retries` times with fresh Gumbel
          noise on the same logits.
  STEP 7: If all retries exhaust, fall back to uniform-random replacement
          for that one slot (last-resort safety net).
  STEP 8: Translate GAN integer IDs back to ADKGD integer IDs via strings.
"""
from collections import defaultdict

import numpy as np
import torch

from kgsage.gan.models import Generator, KGSAGEGenerator


def load_checkpoint(ckpt_path, device=None):
    """Reconstruct the trained Generator + helpers from a .pt file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    # `hidden` was not saved by older checkpoints; fall back to the Generator default.
    gen_kwargs = dict(
        n_ent=payload["n_ent"],
        n_rel=payload["n_rel"],
        dim=payload["dim"],
        z_dim=payload["z_dim"],
    )
    if "hidden" in payload:
        gen_kwargs["hidden"] = payload["hidden"]
    G = Generator(**gen_kwargs).to(device)
    G.load_state_dict(payload["generator_state"])
    G.eval()

    # The set of real triples (in the GAN's ID space) for collision filtering.
    real_triple_set = set(tuple(t) for t in payload["real_triples"])

    return {
        "generator": G,
        "device": device,
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
          output across calls. Without this, every call would return the
          same negative, which gives ADKGD's contrastive loss zero variety.
    """
    # (a) MASK
    masked = logits.clone()
    masked[clean_index] = float("-inf")

    # (b) NOISE: add Gumbel-distributed noise (proven equivalent to sampling
    # from softmax(logits) when followed by argmax — the "Gumbel-Softmax"
    # trick). Temperature 0.5 = mostly argmax but with some randomness.
    u = torch.rand_like(masked).clamp_(1e-10, 1.0 - 1e-10)
    gumbel = -torch.log(-torch.log(u))
    return int((masked + gumbel * 0.5).argmax().item())


def _uniform_fallback(h, r, t, slot, n_ent, n_rel, real_triple_set, rng, max_tries=200):
    """Implements STEP 7: last-resort uniform-random replacement.

    Only called when the GAN's retry loop in STEP 6 exhausts its budget
    (every Gumbel-sampled candidate kept colliding with the real graph).
    We give up on the GAN's distribution and just pick uniformly at random.
    This guarantees ADKGD always gets a valid negative back.
    """
    for _ in range(max_tries):
        if slot == 0:
            candidate = (int(rng.integers(n_ent)), r, t)
        elif slot == 1:
            candidate = (h, int(rng.integers(n_rel)), t)
        else:
            candidate = (h, r, int(rng.integers(n_ent)))
        if candidate[0] != candidate[2] and candidate not in real_triple_set:
            return candidate
    # Shouldn't happen on a sane KG — return whatever we last tried.
    return candidate


def generate_negatives(adkgd_triples, payload, adkgd_maps, rng=None,
                       batch_size=256, max_retries=10):
    """Generate one negative per input triple. Main entry point.

    adkgd_triples : list of (h, r, t) in ADKGD's integer ID space
    payload       : the dict returned by load_checkpoint()
    adkgd_maps    : dict with 'id2ent', 'id2rel', 'ent2id', 'rel2id' from
                    ADKGD's Reader (those are different mappings than the
                    GAN's — we round-trip via strings)
    rng           : numpy random.Generator (per-Reader seeded for reproducibility)

    Returns: (negatives_list, stats_dict).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    G = payload["generator"]
    device = payload["device"]
    ent2id_gan = payload["ent2id"]
    rel2id_gan = payload["rel2id"]
    id2ent_gan = payload["id2ent"]
    id2rel_gan = payload["id2rel"]
    real_triple_set = payload["real_triple_set"]
    n_ent = payload["n_ent"]
    n_rel = payload["n_rel"]
    z_dim = payload["z_dim"]

    # ------------------------------------------------------------------
    # STEP 1: Translate ADKGD IDs -> strings -> GAN IDs (once, up front).
    # ------------------------------------------------------------------
    # ADKGD and the GAN both built `ent2id`/`rel2id` from the SAME files,
    # but in "first-seen" order, so the same string can get a different int
    # in each repo. We use the string as the bridge between the two worlds.
    gan_triples = []
    for h_adk, r_adk, t_adk in adkgd_triples:
        # ADKGD int  ->  string
        h_s = adkgd_maps["id2ent"][h_adk]
        r_s = adkgd_maps["id2rel"][r_adk]
        t_s = adkgd_maps["id2ent"][t_adk]
        # string  ->  GAN int
        gan_triples.append((ent2id_gan[h_s], rel2id_gan[r_s], ent2id_gan[t_s]))

    out = []
    stats = {
        "processed": 0,
        "retries": 0,
        "uniform_fallbacks": 0,
        "slot_h": 0,
        "slot_r": 0,
        "slot_t": 0,
    }

    # Outer loop: process in batches so STEP 2 (generator forward) is one
    # efficient matmul per batch instead of one tiny matmul per triple.
    for batch_start in range(0, len(gan_triples), batch_size):
        batch = gan_triples[batch_start:batch_start + batch_size]
        n = len(batch)

        # --------------------------------------------------------------
        # STEP 2: Generator forward pass on the whole batch.
        # --------------------------------------------------------------
        # Output: three logit tensors, each (n, vocab_size). Think of these
        # as "the GAN's score for every possible entity/relation in each slot".
        # They are PROBABILITIES, not picks. Picking happens in STEP 4.
        h_in = torch.tensor([row[0] for row in batch], dtype=torch.long, device=device)
        r_in = torch.tensor([row[1] for row in batch], dtype=torch.long, device=device)
        t_in = torch.tensor([row[2] for row in batch], dtype=torch.long, device=device)

        with torch.no_grad():
            z = torch.randn(n, z_dim, device=device)
            head_logits, rel_logits, tail_logits = G(h_in, r_in, t_in, z)

        # --------------------------------------------------------------
        # STEP 3: Pick which slot to corrupt - uniform random per triple.
        # --------------------------------------------------------------
        # 0 = head, 1 = relation, 2 = tail. Matches ADKGD baseline exactly.
        slots = rng.integers(3, size=n)

        # Inner loop: one triple at a time. Steps 4-8 happen per item.
        for i in range(n):
            h_gan, r_gan, t_gan = batch[i]
            slot = int(slots[i])

            # ----------------------------------------------------------
            # STEP 4: Select the right logits + clean value for that slot.
            # (The actual mask + Gumbel sample happens inside the retry
            # loop below, calling _pick_new_index_with_noise().)
            # ----------------------------------------------------------
            if slot == 0:
                logits_for_slot = head_logits[i]
                clean_value = h_gan
            elif slot == 1:
                logits_for_slot = rel_logits[i]
                clean_value = r_gan
            else:
                logits_for_slot = tail_logits[i]
                clean_value = t_gan

            # ----------------------------------------------------------
            # STEPS 5 + 6: Build candidate, validate, retry on collision.
            # ----------------------------------------------------------
            # STEP 5 = assemble (new_value + 2 original slots).
            # STEP 6 = check (self-loop? real triple?). Retry with fresh
            #          Gumbel noise if the candidate is invalid.
            neg_h, neg_r, neg_t = h_gan, r_gan, t_gan
            num_tries = 0
            for _ in range(max_retries):
                # Pick a new value (STEP 4: mask + Gumbel + argmax).
                new_idx = _pick_new_index_with_noise(logits_for_slot, clean_value, rng)

                # STEP 5: glue new_idx into the chosen slot, keep the other two.
                if slot == 0:
                    candidate = (new_idx, r_gan, t_gan)
                elif slot == 1:
                    candidate = (h_gan, new_idx, t_gan)
                else:
                    candidate = (h_gan, r_gan, new_idx)
                num_tries += 1

                # STEP 6: validate. `not self-loop` AND `not in real graph`.
                if candidate[0] != candidate[2] and candidate not in real_triple_set:
                    neg_h, neg_r, neg_t = candidate
                    break  # accepted - move on to the next triple
            else:
                # ------------------------------------------------------
                # STEP 7: Uniform fallback (retries exhausted).
                # ------------------------------------------------------
                # The GAN's distribution kept proposing real triples or self-
                # loops. Give up on the model and pick uniformly at random.
                # Rare on a sane KG; if you see this number rise, the GAN's
                # output distribution has collapsed (mode collapse).
                neg_h, neg_r, neg_t = _uniform_fallback(
                    h_gan, r_gan, t_gan, slot, n_ent, n_rel, real_triple_set, rng,
                )
                stats["uniform_fallbacks"] += 1
            stats["retries"] += num_tries - 1  # extra rolls beyond the first

            # Bookkeeping for the stats line.
            if slot == 0:
                stats["slot_h"] += 1
            elif slot == 1:
                stats["slot_r"] += 1
            else:
                stats["slot_t"] += 1

            # ----------------------------------------------------------
            # STEP 8: Translate GAN IDs -> strings -> ADKGD IDs.
            # ----------------------------------------------------------
            # Same string-bridge trick as STEP 1, in reverse, so the value
            # we return matches the vocab ADKGD's Reader expects.
            out.append((
                adkgd_maps["ent2id"][id2ent_gan[neg_h]],   # GAN int -> string -> ADKGD int
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
    return (
        f"processed={stats['processed']:,}  "
        f"retries={stats['retries']:,}  "
        f"uniform_fallbacks={stats['uniform_fallbacks']:,}  "
        f"slot_distribution: {slot_pct}"
    )


# ===========================================================================
# Phase 2 — the PAIR-AWARE role-swap contradiction generator.
#
# Unlike generate_negatives (single-slot corruption), this loads a checkpoint
# from gan/train_kgsage.py and, for each anchor (h, r, t), samples a partner
# relation r' and emits the role-swap contradiction (t, r', h). It keeps the
# (t, r', h) absent-from-graph check that makes the pair a contradiction and
# auto-rejects symmetric relations (whose reverse IS in the graph).
# ===========================================================================
def load_kgsage_checkpoint(ckpt_path, device=None):
    """Reconstruct a trained KGSAGEGenerator from a gan/train_kgsage.py .pt."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_ent, n_rel, dim = payload["n_ent"], payload["n_rel"], payload["dim"]
    hidden = payload.get("hidden", 256)
    # Rebuild from empty tensors of the right shape; load_state_dict fills them.
    G = KGSAGEGenerator(
        torch.empty(n_ent, dim), torch.empty(n_rel, dim), hidden=hidden,
    ).to(device)
    G.load_state_dict(payload["generator_state"])
    G.eval()
    return {
        "generator": G,
        "device": device,
        "ent2id": payload["ent2id"],
        "rel2id": payload["rel2id"],
        "id2ent": payload["id2ent"],
        "id2rel": payload["id2rel"],
        "real_triple_set": set(tuple(t) for t in payload["real_triples"]),
        "n_ent": n_ent,
        "n_rel": n_rel,
        "dim": dim,
    }


def _gumbel_argmax(logits, tau):
    """Sample an index from `logits` via Gumbel-argmax at temperature `tau`."""
    u = torch.rand_like(logits).clamp_(1e-10, 1.0 - 1e-10)
    gumbel = -torch.log(-torch.log(u))
    return int((logits + tau * gumbel).argmax().item())


def _uniform_partner(t, h, n_rel, real_set, rng, max_tries=200):
    """Fallback: a uniform r' such that (t, r', h) is absent from the graph."""
    for _ in range(max_tries):
        rp = int(rng.integers(n_rel))
        if (t, rp, h) not in real_set:
            return (t, rp, h)
    return (t, int(rng.integers(n_rel)), h)


def generate_kgsage_partners(anchor_triples, payload, adkgd_maps=None, rng=None,
                             batch_size=256, tau=0.5, max_retries=10):
    """For each anchor (h, r, t), emit the role-swap contradiction (t, r', h).

    anchor_triples : list of (h, r, t).
    payload        : dict from load_kgsage_checkpoint().
    adkgd_maps     : optional {id2ent, id2rel, ent2id, rel2id} for the ADKGD id
                     round-trip (string-bridged, exactly like generate_negatives).
                     If None, triples are taken in the GAN's own id space — the
                     mode the smoke test and standalone generation use.

    Returns (partners_list, stats_dict). Each partner is a (t, r', h) triple.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    G = payload["generator"]
    device = payload["device"]
    real_set = payload["real_triple_set"]
    n_rel = payload["n_rel"]

    # ADKGD ids -> strings -> GAN ids (only when wired into ADKGD).
    if adkgd_maps is not None:
        ent2id_gan, rel2id_gan = payload["ent2id"], payload["rel2id"]
        gan_triples = [
            (ent2id_gan[adkgd_maps["id2ent"][h]],
             rel2id_gan[adkgd_maps["id2rel"][r]],
             ent2id_gan[adkgd_maps["id2ent"][t]])
            for (h, r, t) in anchor_triples
        ]
    else:
        gan_triples = list(anchor_triples)

    out = []
    stats = {"processed": 0, "generated": 0, "fallbacks": 0,
             "skipped_selfloop": 0, "self_swap": 0, "rel_counts": defaultdict(int)}

    for bstart in range(0, len(gan_triples), batch_size):
        batch = gan_triples[bstart:bstart + batch_size]
        h_in = torch.tensor([b[0] for b in batch], dtype=torch.long, device=device)
        r_in = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        t_in = torch.tensor([b[2] for b in batch], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = G(h_in, r_in, t_in)   # (n, n_rel)

        for i, (h, r, t) in enumerate(batch):
            if h == t:
                stats["skipped_selfloop"] += 1
                continue
            partner = None
            for _ in range(max_retries):
                rp = _gumbel_argmax(logits[i], tau)
                cand = (t, rp, h)
                if cand not in real_set:   # absent => contradiction (symmetric auto-rejected)
                    partner = cand
                    break
            if partner is None:
                partner = _uniform_partner(t, h, n_rel, real_set, rng)
                stats["fallbacks"] += 1

            if partner[1] == r:
                stats["self_swap"] += 1
            stats["rel_counts"][partner[1]] += 1
            stats["generated"] += 1

            if adkgd_maps is not None:
                id2ent_gan, id2rel_gan = payload["id2ent"], payload["id2rel"]
                out.append((
                    adkgd_maps["ent2id"][id2ent_gan[partner[0]]],
                    adkgd_maps["rel2id"][id2rel_gan[partner[1]]],
                    adkgd_maps["ent2id"][id2ent_gan[partner[2]]],
                ))
            else:
                out.append(partner)
        stats["processed"] += len(batch)

    return out, stats


def render_partner_stats(stats):
    """Human-readable summary of one batch of partner generation."""
    distinct = len(stats["rel_counts"])
    return (
        f"processed={stats['processed']:,}  generated={stats['generated']:,}  "
        f"fallbacks={stats['fallbacks']:,}  self_swap={stats['self_swap']:,}  "
        f"skipped_selfloop={stats['skipped_selfloop']:,}  "
        f"distinct_partner_rels={distinct}"
    )


# Public API alias — the SIMPLE single-slot generator remains the default the
# ADKGD bridge calls, because the current production checkpoints (and the
# kgsage_bridge contract) are built for it. Flip this to generate_kgsage_partners
# ONLY after a Phase 2 checkpoint exists AND kgsage_bridge.bridge is updated to
# load it via load_kgsage_checkpoint (the two checkpoint formats differ).
generate_contradictions = generate_negatives

# Direct handle on the Phase 2 generator (no alias indirection).
generate_partners = generate_kgsage_partners

