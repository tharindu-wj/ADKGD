"""Generate role-swap contradiction negatives from a trained KGSAGE checkpoint.

This is the public generation API for the KGSAGE package. ADKGD calls it (via
kgsage_bridge.bridge) every time it builds a training batch with
`--neg_source gan`. Everything stays in-process -- no intermediate file.

For each anchor (h, r, t), the generator samples a partner relation r' and emits
the role-swap contradiction (t, r', h), keeping the absent-from-graph check that
makes the pair a contradiction (and auto-rejects symmetric relations, whose
reverse IS in the graph). IDs are string-bridged between ADKGD and the GAN.
"""
from collections import defaultdict

import numpy as np
import torch

from kgsage.gan.models import KGSAGEGenerator


# For each anchor (h, r, t) this loads a checkpoint from gan/train.py and samples
# a partner relation r', emitting the role-swap contradiction (t, r', h). The
# (t, r', h) absent-from-graph check is what makes the pair a contradiction and
# auto-rejects symmetric relations (whose reverse IS in the graph).
def load_kgsage_checkpoint(ckpt_path, device=None):
    """Reconstruct a trained KGSAGEGenerator from a gan/train.py .pt."""
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
                             batch_size=256, tau=0.5, max_retries=10,
                             pad_selfloops=False):
    """For each anchor (h, r, t), emit the role-swap contradiction (t, r', h).

    anchor_triples : list of (h, r, t).
    payload        : dict from load_kgsage_checkpoint().
    adkgd_maps     : optional {id2ent, id2rel, ent2id, rel2id} for the ADKGD id
                     round-trip (string-bridged between ADKGD and GAN id spaces).
                     If None, triples are taken in the GAN's own id space — the
                     mode the smoke test and standalone generation use.
    pad_selfloops  : if False (default), self-loop anchors (h == t) are SKIPPED,
                     so the output may be shorter than the input. If True, each
                     self-loop anchor gets a uniform-random fallback negative
                     instead, keeping the output 1:1 and in input order — which
                     is what the ADKGD bridge needs (one negative per positive).

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
                if not pad_selfloops:
                    continue
                # ADKGD wants exactly one negative per positive; emit a uniform
                # fallback for self-loop anchors so the output stays 1:1.
                partner = _uniform_partner(t, h, n_rel, real_set, rng)
            else:
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


# Public API name for the standalone generator.
generate_partners = generate_kgsage_partners

