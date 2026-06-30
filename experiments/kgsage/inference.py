"""Use a trained KGSAGE GAN checkpoint to produce one negative per input triple.

This is the public generation API for the KGSAGE package. ADKGD calls it
(via kgsage_bridge.bridge) every time it builds a training batch with
`--neg_source gan`. Everything stays in-process — no intermediate file.

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
import numpy as np
import torch

from kgsage.gan.models import KGSAGEGenerator


def load_checkpoint(ckpt_path, device=None):
    """Reconstruct the trained generator + helpers from a .pt file."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

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
          output across calls.
    """
    masked = logits.clone()
    masked[clean_index] = float("-inf")

    # Add Gumbel-distributed noise (Gumbel-Max trick). Temperature 0.5 =
    # mostly argmax but with some randomness.
    u = torch.rand_like(masked).clamp_(1e-10, 1.0 - 1e-10)
    gumbel = -torch.log(-torch.log(u))
    return int((masked + gumbel * 0.5).argmax().item())


def _uniform_fallback(h, r, t, slot, n_ent, n_rel, real_triple_set, rng, max_tries=200):
    """Implements STEP 7: last-resort uniform-random replacement.

    Only called when the GAN's retry loop in STEP 6 exhausts its budget
    (every Gumbel-sampled candidate kept colliding with the real graph).
    Guarantees ADKGD always gets a valid negative back.
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
    return candidate


def generate_negatives(adkgd_triples, payload, adkgd_maps, rng=None,
                       batch_size=256, max_retries=10):
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
    ent2id_gan = payload["ent2id"]
    rel2id_gan = payload["rel2id"]
    id2ent_gan = payload["id2ent"]
    id2rel_gan = payload["id2rel"]
    real_triple_set = payload["real_triple_set"]
    n_ent = payload["n_ent"]
    n_rel = payload["n_rel"]
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
        "retries": 0,
        "uniform_fallbacks": 0,
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
            head_logits, rel_logits, tail_logits = G(h_in, r_in, t_in, z)

        # STEP 3: Pick which slot to corrupt - uniform random per triple.
        slots = rng.integers(3, size=n)

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

            # STEPS 5 + 6: build candidate, validate, retry on collision.
            neg_h, neg_r, neg_t = h_gan, r_gan, t_gan
            num_tries = 0
            for _ in range(max_retries):
                new_idx = _pick_new_index_with_noise(logits_for_slot, clean_value, rng)
                if slot == 0:
                    candidate = (new_idx, r_gan, t_gan)
                elif slot == 1:
                    candidate = (h_gan, new_idx, t_gan)
                else:
                    candidate = (h_gan, r_gan, new_idx)
                num_tries += 1
                if candidate[0] != candidate[2] and candidate not in real_triple_set:
                    neg_h, neg_r, neg_t = candidate
                    break
            else:
                # STEP 7: Uniform fallback (retries exhausted).
                neg_h, neg_r, neg_t = _uniform_fallback(
                    h_gan, r_gan, t_gan, slot, n_ent, n_rel, real_triple_set, rng,
                )
                stats["uniform_fallbacks"] += 1
            stats["retries"] += num_tries - 1

            if slot == 0:
                stats["slot_h"] += 1
            elif slot == 1:
                stats["slot_r"] += 1
            else:
                stats["slot_t"] += 1

            # STEP 8: Translate GAN IDs -> strings -> ADKGD IDs.
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
    return (
        f"processed={stats['processed']:,}  "
        f"retries={stats['retries']:,}  "
        f"uniform_fallbacks={stats['uniform_fallbacks']:,}  "
        f"slot_distribution: {slot_pct}"
    )
