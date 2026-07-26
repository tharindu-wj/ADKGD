"""KGSAGE — Knowledge Graph Semantic Anomaly Generator.

A standalone-ready Python package for generating knowledge-graph CORRUPTIONS
— false triples built from real ones — used to train and evaluate per-triple
anomaly detectors such as ADKGD. Inside this package the emitted object is
always a "corruption"; only the bridge re-labels it a detector-side
"negative" (training) or "anomaly" (evaluation).

The dual-discriminator architecture takes a real triple and produces a
false-but-plausible one: exactly one ENTITY slot (head or tail) is corrupted;
the relation is never corrupted (see kgsage/corruption_generation.py,
STEP 2). The adversarial training stack lives in `kgsage.gan` — read that
folder name as "the adversarial training stack": it holds the neighbourhood
context encoder, the membership sketches, the candidate sampler and BOTH
discriminators, not just the GAN loop. Downstream-detector integration (e.g.
the ADKGD bridge that calls KGSAGE from a detector's training pipeline) lives
in `experiments/kgsage_bridge/`, keeping `kgsage/` detector-agnostic.

Public API (stable across versions; suitable for the future pip release):
  load_kg(path)                - load a KG from a TSV directory
  resolve_dataset(name_or_path)- look up known dataset defaults
  KNOWN_DATASETS               - dict of pre-configured datasets
  CandidateScoringGenerator    - the generator G (scores candidates; conditioned
                                 on the context table E' + membership sketches)
  generate_negatives           - Phase 3: one corruption per input triple
                                 (frozen name; the bridge re-exports it)
  load_checkpoint              - load a trained checkpoint for generation

The torch-backed symbols are lazy-loaded: `import kgsage` works without torch
installed (so dataset registry lookups + path resolution still run on machines
that only have the standard library).

See README.md for the run order and dataset extension story.
"""
__version__ = "0.1.0"

# Eager — pure-Python, no torch.
from kgsage.data.loaders import load_kg
from kgsage.data.datasets import resolve_dataset, KNOWN_DATASETS

# Lazy — defer heavy imports (torch) until a model symbol or generation helper
# is actually accessed. Lets `import kgsage` succeed without torch installed.
_LAZY_ATTRS = {
    # Phase 2 — the generator G (needs torch)
    "CandidateScoringGenerator": "kgsage.gan.generator",
    "gumbel_softmax":       "kgsage.gan.generator",
    # Phase 1 — the RGCN encoder that builds the context table E'
    # (needs torch + torch_geometric)
    "NeighbourhoodContextEncoder": "kgsage.gan.neighbourhood_context_encoder",
    # Phase 3 — Corruption Generation (needs torch + the trained models)
    "generate_negatives":   "kgsage.corruption_generation",
    "load_checkpoint":      "kgsage.corruption_generation",
}


def __getattr__(name):
    """Module-level lazy attribute lookup (PEP 562, Python 3.7+).

    Only triggers when something tries to access `kgsage.<name>` for a name
    that wasn't eagerly imported above.
    """
    if name in _LAZY_ATTRS:
        import importlib
        module = importlib.import_module(_LAZY_ATTRS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'kgsage' has no attribute {name!r}")


__all__ = [
    "__version__",
    # Data
    "load_kg",
    "resolve_dataset",
    "KNOWN_DATASETS",
    # Generator - lazy-loaded; requires torch at access time
    "CandidateScoringGenerator",
    "gumbel_softmax",
    # Context encoder - lazy-loaded; requires torch + torch_geometric at access time
    "NeighbourhoodContextEncoder",
    # Corruption generation - lazy-loaded; requires torch at access time
    "generate_negatives",
    "load_checkpoint",
]
