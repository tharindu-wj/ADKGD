"""KGSAGE <-> ADKGD bridge.

The single integration layer between the KGSAGE package and ADKGD's training
pipeline. ADKGD's `dataset.py` imports the three-function API from here:

  - load_gan(checkpoint_path)        -> payload dict (model + vocab + reals)
  - generate(triples, payload, ...)  -> list of negative triples + stats
  - render_stats(stats)              -> human-readable log line

This is the only module in the repo that knows about BOTH the standalone
KGSAGE package (`kgsage.*`) and ADKGD's vocabulary/ID conventions.

Why this lives outside `experiments/kgsage/`:
  We want `kgsage/` to be a self-contained generation library that doesn't
  import or assume ADKGD. The bridge here is application glue, not library
  code, so it stays out of the package.

WHY NOT JUST CALL kgsage DIRECTLY FROM dataset.py:
  ADKGD's dataset.py has hard-coded sys.path manipulation around an import.
  Keeping that import pointed at a small bridge (with a stable contract) lets
  the KGSAGE internals evolve — including the future Phase 2 pair-aware
  generator — without touching dataset.py again.
"""
import os
import sys

import numpy as np

# Put `experiments/` on sys.path so `from kgsage.inference import ...` resolves
# when dataset.py imports this bridge. dataset.py already adds
# `experiments/kgsage_bridge` to sys.path; we add `experiments/` here so the
# absolute `kgsage.*` imports below work without further fuss.
_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

# Two generator families live behind this bridge, auto-selected by checkpoint:
#   * SIMPLE GAN (Phase 1 placeholder) -> single-slot corruption negatives.
#   * PAIR-AWARE KGSAGE GAN (Phase 2)  -> role-swap contradiction partners.
# dataset.py is unchanged: it always calls load_gan / generate / render_stats,
# and the bridge picks the right implementation from the checkpoint's "kind".
from kgsage.inference import (  # noqa: E402
    load_checkpoint, generate_negatives, render_stats as _render_simple,
    load_kgsage_checkpoint, generate_kgsage_partners, render_partner_stats,
)

__all__ = ["load_gan", "generate", "render_stats"]


def load_gan(ckpt_path, device=None):
    """Load a trained KGSAGE GAN checkpoint once; returns a payload to reuse.

    Auto-detects the checkpoint family from its "kind" field:
      "kgsage_pairgan" -> the Phase 2 pair-aware role-swap generator
      anything else    -> the simple single-slot-corruption generator

    The returned payload carries a private "_kind" tag so generate()/render_stats()
    dispatch to the matching implementation. Common keys either way:
      generator, device, ent2id, rel2id, id2ent, id2rel, real_triple_set,
      n_ent, n_rel.
    """
    import torch  # local import: keep module import cheap for non-torch callers

    head = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kind = head.get("kind") if isinstance(head, dict) else None

    if kind == "kgsage_pairgan":
        payload = load_kgsage_checkpoint(ckpt_path, device=device)
        payload["_kind"] = "kgsage_pairgan"
    else:
        payload = load_checkpoint(ckpt_path, device=device)
        payload["_kind"] = "simple"
    return payload


def generate(adkgd_triples, *,
             payload,
             adkgd_id2ent, adkgd_id2rel,
             adkgd_ent2id, adkgd_rel2id,
             rng=None):
    """Produce one ADKGD-ID negative per input ADKGD-ID positive.

    Simple checkpoint  -> single-slot corruption of each triple.
    KGSAGE checkpoint  -> the role-swap contradiction partner (t, r', h);
                          self-loops are padded so the output stays 1:1.

    Returns (negatives, stats_dict).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    adkgd_maps = {
        "id2ent": adkgd_id2ent,
        "id2rel": adkgd_id2rel,
        "ent2id": adkgd_ent2id,
        "rel2id": adkgd_rel2id,
    }
    if payload.get("_kind") == "kgsage_pairgan":
        return generate_kgsage_partners(
            list(adkgd_triples), payload, adkgd_maps, rng=rng, pad_selfloops=True)
    return generate_negatives(list(adkgd_triples), payload, adkgd_maps, rng=rng)


def render_stats(stats):
    """Human-readable one-line summary, dispatched by stats shape.

    The pair-aware generator emits a different stats dict (with 'rel_counts' /
    'self_swap') than the simple one ('slot_h' / 'slot_r' / 'slot_t'); pick the
    matching formatter so dataset.py's `print('[GAN] ' + render_stats(stats))`
    works for both.
    """
    if "rel_counts" in stats or "self_swap" in stats:
        return "kgsage role-swap | " + render_partner_stats(stats)
    return _render_simple(stats)
