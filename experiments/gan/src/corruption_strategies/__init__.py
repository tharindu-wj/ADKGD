"""TRIC-style corruption for multi-relational KG adjacency tensors.

    corrupted = apply_tric_corruption(adj_nxnxr, num_corruptions, rng, node_types=..., op_weights=...)

Use functools.partial to pre-bind node_types / op_weights before passing to
build_kg_paired_dataset.
"""

from .tric import apply_tric_corruption

__all__ = ["apply_tric_corruption"]
