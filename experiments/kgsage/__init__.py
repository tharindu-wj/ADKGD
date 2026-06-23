"""KGSAGE — Knowledge Graph Semantic Anomaly Generator.

A standalone-ready Python package for generating role-swap contradiction
anomalies in knowledge graphs, designed to extend per-triple anomaly detectors
(like ADKGD) to multi-triple anomaly categories.

Two phases:
  Phase 1 — encoder pretraining (RGCN + DistMult). Implemented in `kgsage.encoder`.
  Phase 2 — pair-aware adversarial generator + discriminator. Will live in
            `kgsage.gan`. Loads the Phase 1 checkpoint as initialisation.

The package is structurally standalone — nothing here imports from outside
the `kgsage.*` namespace. ADKGD integration (the bridge that calls KGSAGE
from ADKGD's training pipeline) lives in `experiments/kgsage_bridge/`,
keeping `kgsage/` ADKGD-agnostic.

Public API (stable across versions; suitable for the future pip release):
  load_kg(path)                — load a KG from a TSV directory
  resolve_dataset(name_or_path)— look up known dataset defaults
  KNOWN_DATASETS               — dict of pre-configured datasets
  KGSAGEEncoder                — RGCN encoder (Phase 1) — requires torch_geometric
  KGSAGEDistMultDecoder        — DistMult decoder (Phase 1)
  KGSAGELinkPredictor          — combined encoder + decoder (Phase 1)

The encoder symbols are lazy-loaded: `import kgsage` works without
torch_geometric installed (so data-layer operations like resolve_dataset()
and audit_dataset still run on dev machines). torch_geometric is only
required when an encoder symbol is actually accessed.

See README.md for the run order, dataset extension story, and migration notes.
See ../docs/THESIS_PLAN_pairgan_contradictions.md for the full thesis plan.
"""
__version__ = "0.1.0"

# Eager — pure-Python / torch-only, no heavy deps
from kgsage.data.loaders import load_kg
from kgsage.data.datasets import resolve_dataset, KNOWN_DATASETS

# Lazy — defer torch_geometric import until an encoder symbol is actually
# accessed. Lets `import kgsage` succeed on machines that have torch but
# not torch_geometric (typical dev setup; PyG is HPC-only here).
_LAZY_ATTRS = {
    "KGSAGEEncoder":         "kgsage.encoder.models",
    "KGSAGEDistMultDecoder": "kgsage.encoder.models",
    "KGSAGELinkPredictor":   "kgsage.encoder.models",
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
    # Encoder (Phase 1) — lazy-loaded; requires torch_geometric at access time
    "KGSAGEEncoder",
    "KGSAGEDistMultDecoder",
    "KGSAGELinkPredictor",
]
