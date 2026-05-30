"""Use a trained GAN checkpoint to produce one negative per input triple.

This is what ADKGD calls every time it builds a training batch with
`--neg_source gan`. Everything stays in-process — no intermediate file.

How it works:
  1. Load the trained generator + its baked-in vocab + the set of real triples.
  2. For each input triple (h, r, t):
       a. Pick a slot uniformly at random ({head, relation, tail}).
       b. Run the generator with random noise.
       c. Mask the original index in that slot to -inf so the slot must move.
       d. Pick the new index via argmax + Gumbel noise (for stochasticity).
       e. If the resulting triple already exists in the graph, retry up to
          `max_retries` times.
       f. If retries exhaust, fall back to uniform random for that one slot.
  3. Translate back from the GAN's integer IDs to ADKGD's integer IDs
     via strings (the two repos may number the same entity differently).
"""
import os
import sys

import numpy as np
import torch

# Import sibling modules.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Generator  # noqa: E402


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
    """Sample a new index from `logits`, forbidding `clean_index` itself.

    We mask the clean position to -inf (so argmax can't pick it) and add
    Gumbel noise so the same triple produces different negatives across
    calls. Pure argmax would be deterministic — not what we want for
    contrastive training.
    """
    masked = logits.clone()
    masked[clean_index] = float("-inf")
    # Gumbel noise (small temperature, mostly argmax but with some variety).
    u = torch.rand_like(masked).clamp_(1e-10, 1.0 - 1e-10)
    gumbel = -torch.log(-torch.log(u))
    return int((masked + gumbel * 0.5).argmax().item())


def _uniform_fallback(h, r, t, slot, n_ent, n_rel, real_triple_set, rng, max_tries=200):
    """Last-resort: pick a uniform random replacement in the chosen slot."""
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

    # Step 1: ADKGD integer IDs -> strings -> GAN integer IDs.
    # We do this once up front so the inner loop only deals with integers.
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

    # Step 2: process in batches so the generator's forward pass is efficient.
    for batch_start in range(0, len(gan_triples), batch_size):
        batch = gan_triples[batch_start:batch_start + batch_size]
        n = len(batch)

        h_in = torch.tensor([row[0] for row in batch], dtype=torch.long, device=device)
        r_in = torch.tensor([row[1] for row in batch], dtype=torch.long, device=device)
        t_in = torch.tensor([row[2] for row in batch], dtype=torch.long, device=device)

        with torch.no_grad():
            z = torch.randn(n, z_dim, device=device)
            head_logits, rel_logits, tail_logits = G(h_in, r_in, t_in, z)

        # One uniform random slot pick per positive.
        slots = rng.integers(3, size=n)

        for i in range(n):
            h_gan, r_gan, t_gan = batch[i]
            slot = int(slots[i])

            # Pick which logits to sample from based on the slot.
            if slot == 0:
                logits_for_slot = head_logits[i]
                clean_value = h_gan
            elif slot == 1:
                logits_for_slot = rel_logits[i]
                clean_value = r_gan
            else:
                logits_for_slot = tail_logits[i]
                clean_value = t_gan

            # Retry loop: re-sample if we hit a self-loop or a real triple.
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
                # All retries exhausted — uniform random for this one slot.
                neg_h, neg_r, neg_t = _uniform_fallback(
                    h_gan, r_gan, t_gan, slot, n_ent, n_rel, real_triple_set, rng,
                )
                stats["uniform_fallbacks"] += 1
            stats["retries"] += num_tries - 1  # extra rolls beyond the first

            if slot == 0:
                stats["slot_h"] += 1
            elif slot == 1:
                stats["slot_r"] += 1
            else:
                stats["slot_t"] += 1

            # Step 3: GAN integer IDs -> strings -> ADKGD integer IDs.
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
