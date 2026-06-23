"""KGSAGE — Knowledge Graph Semantic Anomaly Generator.

A standalone-ready Python package for generating role-swap contradiction
anomalies in knowledge graphs, designed to extend per-triple anomaly detectors
(like ADKGD) to multi-triple anomaly categories.

Two phases:
  Phase 1 - encoder pretraining (RGCN + DistMult).  In `kgsage.encoder`.
  Phase 2 - adversarial Generator + Discriminator.  In `kgsage.gan`.
            Current implementation is the simple 3-layer MLP GAN; Phase 2
            of the thesis upgrades it to a pair-aware contradiction generator
            conditioned on the Phase 1 encoder embeddings.

The package is structurally standalone — nothing here imports from outside
the `kgsage.*` namespace. ADKGD integration (the bridge that calls KGSAGE
from ADKGD's training pipeline) lives in `experiments/kgsage_bridge/`,
keeping `kgsage/` ADKGD-agnostic.

Public API (stable across versions; suitable for the future pip release):
  load_kg(path)                - load a KG from a TSV directory
  resolve_dataset(name_or_path)- look up known dataset defaults
  KNOWN_DATASETS               - dict of pre-configured datasets
  KGSAGEEncoder                - RGCN encoder (Phase 1) - requires torch_geometric
  KGSAGEDistMultDecoder        - DistMult decoder (Phase 1)
  KGSAGELinkPredictor          - combined encoder + decoder (Phase 1)
  Generator                    - GAN Generator (Phase 2 - simple MLP placeholder)
  Discriminator                - GAN Discriminator (Phase 2)
  generate_contradictions      - inference API: produce negatives for an anchor batch

The encoder + GAN symbols are lazy-loaded: `import kgsage` works without
torch installed (so dataset registry lookups + path resolution still run on
machines that only have the standard library).

See README.md for the run order, dataset extension story, and migration notes.
See ../docs/THESIS_PLAN_pairgan_contradictions.md for the full thesis plan.
"""
__version__ = "0.1.0"

# Eager — pure-Python, no torch / no PyG.
from kgsage.data.loaders import load_kg
from kgsage.data.datasets import resolve_dataset, KNOWN_DATASETS

# Lazy — defer heavy imports (torch, torch_geometric) until a model symbol
# or inference helper is actually accessed. Lets `import kgsage` succeed on
# machines that don't have those installed.
_LAZY_ATTRS = {
    # Phase 1 encoder (needs torch + torch_geometric)
    "KGSAGEEncoder":           "kgsage.encoder.models",
    "KGSAGEDistMultDecoder":   "kgsage.encoder.models",
    "KGSAGELinkPredictor":     "kgsage.encoder.models",
    # Phase 2 GAN (needs torch)
    "Generator":               "kgsage.gan.models",
    "Discriminator":           "kgsage.gan.models",
    # Inference (needs torch + the GAN models)
    "generate_contradictions": "kgsage.inference",
    "load_checkpoint":         "kgsage.inference",
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
    # Encoder (Phase 1) - lazy-loaded; requires torch_geometric at access time
    "KGSAGEEncoder",
    "KGSAGEDistMultDecoder",
    "KGSAGELinkPredictor",
    # GAN (Phase 2) - lazy-loaded; requires torch at access time
    "Generator",
    "Discriminator",
    "generate_contradictions",
    "load_checkpoint",
]
