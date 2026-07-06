"""KGSAGE — Knowledge Graph Semantic Anomaly Generator.

A standalone-ready Python package for generating synthetic knowledge-graph
anomalies (single-slot-corruption negatives) to train and evaluate per-triple
anomaly detectors such as ADKGD.

A conditional GAN consumes a real triple + noise and produces a fake-but-
plausible triple (one ENTITY slot — head or tail — corrupted; the relation
head exists but is never chosen at generation time, see inference STEP 3). Lives in
`kgsage.gan`. Downstream-detector integration (e.g. the ADKGD bridge that
calls KGSAGE from a detector's training pipeline) lives in
`experiments/kgsage_bridge/`, keeping `kgsage/` detector-agnostic.

Public API (stable across versions; suitable for the future pip release):
  load_kg(path)                - load a KG from a TSV directory
  resolve_dataset(name_or_path)- look up known dataset defaults
  KNOWN_DATASETS               - dict of pre-configured datasets
  KGSAGEGenerator              - GAN generator (3-head; conditioned on cached RGCN E')
  generate_negatives           - inference API: one negative per input triple
  load_checkpoint              - load a trained GAN checkpoint for inference

The GAN symbols are lazy-loaded: `import kgsage` works without torch installed
(so dataset registry lookups + path resolution still run on machines that only
have the standard library).

See README.md for the run order and dataset extension story.
"""
__version__ = "0.1.0"

# Eager — pure-Python, no torch.
from kgsage.data.loaders import load_kg
from kgsage.data.datasets import resolve_dataset, KNOWN_DATASETS

# Lazy — defer heavy imports (torch) until a model symbol or inference helper
# is actually accessed. Lets `import kgsage` succeed without torch installed.
_LAZY_ATTRS = {
    # GAN (needs torch)
    "KGSAGEGenerator":      "kgsage.gan.models",
    "gumbel_softmax":       "kgsage.gan.models",
    # RGCN context encoder (needs torch + torch_geometric)
    "KGSAGEEncoder":        "kgsage.gan.encoder",
    # Inference (needs torch + the GAN models)
    "generate_negatives":   "kgsage.inference",
    "load_checkpoint":      "kgsage.inference",
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
    "KGSAGEGenerator",
    "gumbel_softmax",
    # RGCN encoder - lazy-loaded; requires torch + torch_geometric at access time
    "KGSAGEEncoder",
    "generate_negatives",
    "load_checkpoint",
]
