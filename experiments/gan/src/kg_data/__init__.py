"""Unified KG loading for dummy and FB15k-237-format datasets.

Typical use::

    from kg_data import load_kg, load_kg_union
    kg = load_kg('data/dummy_kg')
    union = load_kg_union('data/FB15K')

    # Now use:
    kg.entity_id_to_row        # str -> int
    kg.relation_id_to_channel  # str -> int
    kg.triple_set_idx          # O(1) "does this triple exist?" check
    kg.node_labels             # int -> str (display names)
    kg.node_types              # int -> str (entity types; may be 'unknown')
    kg.relation_names          # list[str], indexed by channel
"""

from .loader import KnowledgeGraph, load_kg, load_kg_union

__all__ = [
    "KnowledgeGraph",
    "load_kg",
    "load_kg_union",
]
