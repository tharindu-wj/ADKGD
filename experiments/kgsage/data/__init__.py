"""KGSAGE data layer — KG loading and the known-dataset registry.

Public API surfaces:
  load_kg              — load a KG from a TSV directory
  resolve_dataset      — look up known dataset defaults (or treat input as a path)
  KNOWN_DATASETS       — registry of pre-configured datasets

This is the layer every KGSAGE phase starts from: Phase 1 builds the context
table E' from the edge list load_kg returns, and Phase 3 (corruption
generation) reuses the same vocabulary maps.

Add a new dataset to KNOWN_DATASETS in datasets.py; everything else just works.
yago_to_tsv.py is a standalone converter, not part of this import surface.
"""
from kgsage.data.loaders import load_kg
from kgsage.data.datasets import resolve_dataset, KNOWN_DATASETS

__all__ = ["load_kg", "resolve_dataset", "KNOWN_DATASETS"]
