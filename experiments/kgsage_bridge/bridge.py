"""KGSAGE <-> ADKGD bridge.

The single integration layer between the KGSAGE package and ADKGD's training
pipeline. ADKGD's `dataset.py` imports the three-function API from here:

  Detector-side negatives (from a KGSAGE generator checkpoint):
  - load_gan(checkpoint_path)        -> payload dict (model + vocab + reals)
  - generate(triples, payload, ...)  -> list of negative triples + stats
  - render_stats(stats)              -> human-readable log line

All three names are FROZEN. They read as GAN/negative vocabulary because they
are ADKGD's side of the contract: `--neg_source gan` / `--test_anomaly_source
gan` and `--gan_path` are the detector's flags, and dataset.py (repo root)
imports these functions by name.

THE VOCABULARY SEAM: inside `kgsage/` the object produced is a CORRUPTION.
Once it crosses this file it becomes a NEGATIVE (ADKGD training) or an
ANOMALY (ADKGD evaluation). Same object, different owner, different word —
a deliberate boundary rather than drift. Translating the vocabulary is
exactly what this module is for, so do not "fix" the naming on either side.

This is the only module in the repo that knows about BOTH the standalone
KGSAGE package (`kgsage.*`) and ADKGD's vocabulary/ID conventions.

Why this lives outside `experiments/kgsage/`:
  We want `kgsage/` to be a self-contained generation library that doesn't
  import or assume ADKGD. The bridge here is application glue, not library code.
"""
import os
import sys

import numpy as np

# Put `experiments/` on sys.path so `from kgsage.corruption_generation import
# ...` resolves when dataset.py imports this bridge.
_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

# Re-export render_stats unchanged - it's a string formatter that already
# matches dataset.py's expectation.
from kgsage.corruption_generation import load_checkpoint, generate_negatives, render_stats  # noqa: E402,F401

__all__ = ["load_gan", "generate", "render_stats"]


def load_gan(ckpt_path, device=None):
    """Load a trained KGSAGE checkpoint once; returns a payload to reuse.

    Frozen name (`--gan_path` is the matching detector flag); it is a thin
    alias for kgsage.corruption_generation.load_checkpoint.

    Returns a dict with keys:
      generator        - the trained CandidateScoringGenerator G (nn.Module)
      device           - torch.device the model is on
      context_table    - the frozen context table E', [n_ent, dim]
      sketches         - the Bloom membership sketches
      ent2id, rel2id   - the generator's string -> int vocab maps
      id2ent, id2rel   - inverse maps
      real_triple_set  - set of (h, r, t) tuples (for collision filtering)
      pool_masks       - per-relation type pools
      n_ent, n_rel     - vocabulary sizes
    """
    return load_checkpoint(ckpt_path, device=device)


def generate(adkgd_triples, *,
             payload,
             adkgd_id2ent, adkgd_id2rel,
             adkgd_ent2id, adkgd_rel2id,
             rng=None):
    """Produce one ADKGD-ID negative per input ADKGD-ID positive.

    Each one is a single-entity-slot corruption (head or tail — never the
    relation) of the input triple, decoded from the trained generator. This
    is where the vocabulary crosses over: kgsage calls the returned triples
    corruptions, ADKGD calls them negatives.

    Returns (negatives, stats). stats["null_indices"] flags rows whose
    "negative" is in fact the unchanged original triple; ADKGD's dataset.py
    replaces those in the training role and filters them in the eval role.
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
