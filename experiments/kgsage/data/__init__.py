"""KGSAGE data layer — KG loading, dataset registry, dataset-level audit.

Public API surfaces:
  load_kg              — load a KG from a TSV directory
  resolve_dataset      — look up known dataset defaults (or treat input as a path)
  KNOWN_DATASETS       — registry of pre-configured datasets

Add a new dataset to KNOWN_DATASETS in datasets.py; everything else just works.
"""
from kgsage.data.loaders import load_kg
from kgsage.data.datasets import resolve_dataset, KNOWN_DATASETS

__all__ = ["load_kg", "resolve_dataset", "KNOWN_DATASETS"]
