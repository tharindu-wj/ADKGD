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
  import or assume ADKGD. The bridge here is application glue, not library code.
"""
import os
import sys

import numpy as np

# Put `experiments/` on sys.path so `from kgsage.inference import ...` resolves
# when dataset.py imports this bridge.
_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

# Re-export render_stats unchanged - it's a string formatter that already
# matches dataset.py's expectation.
from kgsage.inference import load_checkpoint, generate_negatives, render_stats  # noqa: E402,F401

__all__ = ["load_gan", "generate", "render_stats",
           "load_lp", "generate_band", "render_band_stats"]


def load_gan(ckpt_path, device=None):
    """Load a trained KGSAGE GAN checkpoint once; returns a payload to reuse.

    Returns a dict with keys:
      generator        - the trained KGSAGEGenerator (torch.nn.Module)
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

    Each negative is a single-slot corruption (head, relation, or tail) of the
    input triple, decoded from the trained generator. Returns (negatives, stats).
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


# ---------------------------------------------------------------------------
# Option B: frozen-LP band sampler as a negative / test-anomaly source
# ---------------------------------------------------------------------------

def load_lp(lp_ckpt_path, lp_ids_dir, adkgd_data_dir,
            band_k=10, band_temp=0.5):
    """Build the close-but-false band sampler once; returns a payload to reuse.

    lp_ckpt_path   - LibKGE ComplEx checkpoint (see kgsage/cli/fetch_lp.py)
    lp_ids_dir     - the LibKGE archive dir the checkpoint's ids were assigned
                     from (train/valid/test.txt) -- NOT ADKGD's data dir
    adkgd_data_dir - ADKGD's data/<dataset>/ dir: type pools and the all-splits
                     falseness masks are built from THESE files, so they match
                     the graph ADKGD actually trains/evaluates on
    """
    from kgsage.lp_scorer import ComplExScorer
    from kgsage.band_sampler import BandSampler

    scorer = ComplExScorer.from_libkge(lp_ckpt_path, lp_ids_dir)
    sampler = BandSampler(scorer, adkgd_data_dir,
                          band_k=band_k, band_temp=band_temp)
    return {"sampler": sampler, "lp_ckpt": str(lp_ckpt_path)}


def generate_band(adkgd_triples, *, payload,
                  adkgd_id2ent, adkgd_id2rel,
                  adkgd_ent2id, adkgd_rel2id,
                  rng=None):
    """One ADKGD-ID close-but-false negative per input ADKGD-ID positive.

    Unlike the GAN path there are NO null corruptions: the sampler's fallback
    ladder always produces a genuine single-slot corruption (counted in stats).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    sampler = payload["sampler"]
    str_triples = [(adkgd_id2ent[h], adkgd_id2rel[r], adkgd_id2ent[t])
                   for h, r, t in adkgd_triples]
    neg_str, stats = sampler.corrupt_strings(str_triples, rng=rng)
    negatives = [(adkgd_ent2id[h], adkgd_rel2id[r], adkgd_ent2id[t])
                 for h, r, t in neg_str]
    return negatives, stats


def render_band_stats(stats):
    from kgsage.band_sampler import render_stats as _rs
    return _rs(stats)
