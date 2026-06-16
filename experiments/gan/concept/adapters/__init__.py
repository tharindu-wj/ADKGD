"""Family adapter registry for the concept module.

A family adapter knows how to extract entity types from any KG belonging
to a specific knowledge-base family. Adapters are registered here and
instantiated via get_adapter(family).

Currently supported:
  "freebase"  -> FreebaseAdapter   (FB15K, FB15K-237, FB13, ...)

Planned (added when extending to new datasets):
  "wordnet"   -> WordNetAdapter    (WN18, WN18RR, ...)
  "yago"      -> YagoAdapter       (YAGO 3-10, YAGO 4, YAGO 4.5, ...)

Usage:
  from experiments.gan.concept.adapters import get_adapter
  adapter = get_adapter("freebase")
  entity_to_types = adapter.extract_types("data/FB15K-237/train.txt")
"""
from .base import BaseKBAdapter
from .freebase import FreebaseAdapter


# Registered adapters. Add new families here when their adapter file lands.
_REGISTRY = {
    "freebase": FreebaseAdapter,
}


def get_adapter(family):
    """Return an adapter instance for the named KB family.

    Args:
      family: lowercase family name, e.g. "freebase".

    Raises:
      ValueError if the family isn't registered.
    """
    if family not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown KB family: {family!r}. "
            f"Currently supported: {available}"
        )
    return _REGISTRY[family]()


__all__ = ["BaseKBAdapter", "FreebaseAdapter", "get_adapter"]
