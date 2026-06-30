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

# The KGSAGE pair-aware generator is the only generator. dataset.py calls
# load_gan / generate / render_stats; all three are KGSAGE role-swap here.
from kgsage.inference import (  # noqa: E402
    load_kgsage_checkpoint, generate_kgsage_partners, render_partner_stats,
)

__all__ = ["load_gan", "generate", "render_stats"]


def load_gan(ckpt_path, device=None):
    """Load a trained KGSAGE GAN checkpoint once; returns a payload to reuse.

    Returns a payload with: generator, device, ent2id, rel2id, id2ent, id2rel,
    real_triple_set, n_ent, n_rel. Raises if the checkpoint is not a KGSAGE
    pair-aware checkpoint (e.g. an old simple-GAN .pt), so a wrong --gan_path
    fails loudly instead of mis-loading.
    """
    import torch  # local import: keep module import cheap for non-torch callers

    head = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not (isinstance(head, dict) and head.get("kind") == "kgsage_pairgan"):
        raise ValueError(
            f"{ckpt_path} is not a KGSAGE pair-aware checkpoint "
            "(kind != 'kgsage_pairgan'). Train one with "
            "`python -m kgsage.cli.train_gan`.")
    return load_kgsage_checkpoint(ckpt_path, device=device)


def generate(adkgd_triples, *,
             payload,
             adkgd_id2ent, adkgd_id2rel,
             adkgd_ent2id, adkgd_rel2id,
             rng=None):
    """Produce one ADKGD-ID role-swap contradiction (t, r', h) per input
    positive. Self-loop anchors are padded so the output stays 1:1 with the
    input (the count ADKGD's bp_triples + bn_triples construction expects).

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
    return generate_kgsage_partners(
        list(adkgd_triples), payload, adkgd_maps, rng=rng, pad_selfloops=True)


def render_stats(stats):
    """Human-readable one-line summary of a generation batch."""
    return "kgsage role-swap | " + render_partner_stats(stats)
