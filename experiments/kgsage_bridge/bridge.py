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
KGSAGE package (`kgsage.*`) and ADKGD's vocabulary/ID conventions. Nothing else
here may import `kgsage.*` — that rule is what keeps the two repos separable,
and Phase 5 of the decoupling check greps for violations.

KGSAGE IS AN OPTIONAL DEPENDENCY. It lives in its own repository and is
installed with `pip install -e /path/to/kgsage`. ADKGD's own baseline
(`--neg_source random`) must keep working when it is absent, so this module is
imported lazily by dataset.py and never at ADKGD start-up.
"""
import importlib.util

import numpy as np

# A directory named `kgsage` anywhere on sys.path shadows the installed package,
# even an editable one. setuptools registers its editable finder AFTER the stock
# PathFinder in sys.meta_path, so PathFinder resolves the bare directory as an
# implicit namespace package and wins. The symptom is a baffling
# "cannot import name X from 'kgsage' (unknown location)"; name the cause here.
_spec = importlib.util.find_spec("kgsage")
if _spec is not None and _spec.origin is None:
    _shadow = "\n    ".join(_spec.submodule_search_locations or [])
    raise ImportError(
        "A directory named 'kgsage' is shadowing the installed KGSAGE package:\n"
        "    " + _shadow + "\n"
        "It has no __init__.py, so Python treats it as a namespace package and\n"
        "it outranks the pip-installed one. Delete it -- this repo no longer\n"
        "vendors KGSAGE."
    )

# Re-export render_stats unchanged - it's a string formatter that already
# matches dataset.py's expectation.
try:
    from kgsage.corruption_generation import (  # noqa: F401
        load_checkpoint, generate_negatives, render_stats)
except ImportError as _exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "KGSAGE is not installed.\n"
        "\n"
        "It is a SEPARATE package with its own repository, required only for\n"
        "  --neg_source gan   or   --test_anomaly_source gan\n"
        "\n"
        "Install it (editable, so the checkout stays authoritative):\n"
        "    pip install -e /path/to/kgsage\n"
        "\n"
        "ADKGD's own baseline does not need it:\n"
        "    --neg_source random --test_anomaly_source random\n"
    ) from _exc

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
