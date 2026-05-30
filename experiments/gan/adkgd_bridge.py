"""Bridge between ADKGD's Reader and the simple GAN.

ADKGD imports this module from dataset.py. We re-export the two functions
Reader.get_data() needs:
  - load_gan(checkpoint_path)        -> payload dict (generator + vocab + reals)
  - generate(triples, payload, ...)  -> list of negative triples + stats

Why this file exists: ADKGD lives at the repo root and doesn't know about the
GAN's folder structure. This bridge is the one place that knows about both.
"""
import os
import sys

import numpy as np

# Put this folder on sys.path so the sibling files (model.py, generate.py)
# resolve when ADKGD imports us.
_GAN_DIR = os.path.dirname(os.path.abspath(__file__))
if _GAN_DIR not in sys.path:
    sys.path.insert(0, _GAN_DIR)

# Re-exported for dataset.py — `render_stats` formats the per-batch stats line.
from generate import load_checkpoint, generate_negatives, render_stats  # noqa: E402,F401

__all__ = ["load_gan", "generate", "render_stats"]


def load_gan(ckpt_path, device=None):
    """Load a trained GAN checkpoint once; returns a payload to reuse."""
    return load_checkpoint(ckpt_path, device=device)


def generate(adkgd_triples, *,
             payload,
             adkgd_id2ent, adkgd_id2rel,
             adkgd_ent2id, adkgd_rel2id,
             rng=None):
    """Produce one ADKGD-ID negative per input ADKGD-ID positive.

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
    return generate_negatives(list(adkgd_triples), payload, adkgd_maps, rng=rng)
