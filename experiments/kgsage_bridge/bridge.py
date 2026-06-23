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

# Re-export render_stats unchanged - it's a string formatter that already
# matches dataset.py's expectation.
from kgsage.inference import load_checkpoint, generate_negatives, render_stats  # noqa: E402,F401

__all__ = ["load_gan", "generate", "render_stats"]


def load_gan(ckpt_path, device=None):
    """Load a trained KGSAGE GAN checkpoint once; returns a payload to reuse.

    Returns a dict with keys:
      generator        - the trained Generator (torch.nn.Module)
      device           - torch.device the model is on
      ent2id, rel2id   - GAN's string -> int vocab maps
      id2ent, id2rel   - inverse maps
      real_triple_set  - set of (h, r, t) tuples (for collision filtering)
      n_ent, n_rel     - vocabulary sizes
      z_dim            - noise dimension
    """
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
