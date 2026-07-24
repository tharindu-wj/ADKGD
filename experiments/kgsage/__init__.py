"""KGSAGE — Knowledge Graph Semantic Anomaly Generator.

A standalone-ready Python package for generating synthetic knowledge-graph
anomalies (single-slot-corruption negatives) to train and evaluate per-triple
anomaly detectors such as ADKGD.

A conditional GAN takes a real triple and produces a fake-but-plausible one:
exactly one ENTITY slot (head or tail) is corrupted; the relation is never
corrupted (see kgsage/corruption_generation.py, STEP 2). The training stack
lives in `kgsage.gan`. Downstream-detector integration (e.g. the ADKGD bridge
that calls KGSAGE from a detector's training pipeline) lives in
`experiments/kgsage_bridge/`, keeping `kgsage/` detector-agnostic.

Public API (stable across versions; suitable for the future pip release):
  load_kg(path)                - load a KG from a TSV directory
  resolve_dataset(name_or_path)- look up known dataset defaults
  KNOWN_DATASETS               - dict of pre-configured datasets
  CandidateScoringGenerator    - the generator (candidate scoring; conditioned on E' + sketches)
  generate_negatives           - corruption generation: one negative per input triple
  load_checkpoint              - load a trained checkpoint for generation

The GAN symbols are lazy-loaded: `import kgsage` works without torch installed
(so dataset registry lookups + path resolution still run on machines that only
have the standard library).

See README.md for the run order and dataset extension story.
"""
__version__ = "0.1.0"

# Eager — pure-Python, no torch.
from kgsage.data.loaders import load_kg
from kgsage.data.datasets import resolve_dataset, KNOWN_DATASETS

# Lazy — defer heavy imports (torch) until a model symbol or generation helper
# is actually accessed. Lets `import kgsage` succeed without torch installed.
_LAZY_ATTRS = {
    # GAN (needs torch)
    "CandidateScoringGenerator": "kgsage.gan.generator",
    "gumbel_softmax":       "kgsage.gan.generator",
    # RGCN context encoder (needs torch + torch_geometric)
    "NeighbourhoodContextEncoder": "kgsage.gan.neighbourhood_context_encoder",
    # Corruption generation (needs torch + the GAN models)
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
    # GAN - lazy-loaded; requires torch at access time
    "CandidateScoringGenerator",
    "gumbel_softmax",
    # RGCN encoder - lazy-loaded; requires torch + torch_geometric at access time
    "NeighbourhoodContextEncoder",
    "generate_negatives",
    "load_checkpoint",
]
