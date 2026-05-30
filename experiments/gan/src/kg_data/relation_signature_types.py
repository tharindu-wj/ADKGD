"""Derive pseudo-type labels for entities from their relation signatures.

FB15k-237 ships with no entity-type metadata. Without types,
`corruption_strategies.tric.apply_tric_corruption` silently disables its
`swap_subject_same_type` and `swap_object_same_type` operations - and those
are exactly the operations that produce the "type-coherent but factually
wrong" negatives we want the GAN to learn from.

This module fills that gap WITHOUT relying on an external Freebase type
ontology. The standard trick: an entity's "type" is well-approximated by
the bag of relations it participates in, distinguishing head-position from
tail-position usage. Two entities that play the same syntactic role across
the graph (e.g. both appear as `(?, /people/person/profession, ?)` heads
and `(?, /people/person/place_of_birth, ?)` heads) almost certainly share
a semantic type.

Pipeline:
  1. Build a sparse (num_entities, 2*num_relations) matrix where
     column `2*r`     counts how many times entity e appears as HEAD of relation r,
     column `2*r + 1` counts how many times entity e appears as TAIL of relation r.
  2. TF-IDF normalise the matrix - so a relation that nearly every entity
     participates in (e.g. `/common/topic/name` if it existed) loses weight,
     while rare-but-discriminative relations gain weight.
  3. Truncated SVD reduces to `svd_dim` components for tractable clustering.
  4. MiniBatchKMeans clusters into `num_clusters` groups. Each cluster id
     becomes a pseudo-type string `"type_037"`.

The output is a `Dict[entity_id_string, type_string]` plus a helper that
writes the file in the exact 3-column format
`src.kg_data.loader.load_kg` already reads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

from .loader import KnowledgeGraph


def derive_pseudo_types(
    kg: KnowledgeGraph,
    *,
    num_clusters: int = 100,
    svd_dim: int = 32,
    random_state: int = 0,
) -> Dict[str, str]:
    """Cluster entities by relation signature into `num_clusters` pseudo-types.

    Parameters
    ----------
    kg
        A `KnowledgeGraph` from `load_kg_union` (we want the signature to
        reflect ALL three splits, not just training).
    num_clusters
        Target number of pseudo-types. 100 is a reasonable default for
        FB15k-237 (~14k entities, 237 relations). For the dummy KG you'd
        want a much smaller number; the helper clamps to `num_entities`.
    svd_dim
        Dimensionality of the SVD projection before clustering. 32 is a
        good speed/quality trade-off.
    random_state
        Seed for SVD and KMeans (both are randomised).

    Returns
    -------
    Dict[entity_id_string, type_string]
        Maps each entity string from `kg.entity_id_to_row` to a
        pseudo-type label like `"type_042"`.
    """
    num_entities  = len(kg.entity_id_to_row)
    num_relations = len(kg.relation_id_to_channel)
    if num_entities == 0:
        return {}

    # Clamp k so it never exceeds the number of entities (would crash KMeans).
    effective_num_clusters = min(num_clusters, num_entities)

    # ── Step 1: sparse (entity, head-or-tail-of-relation) count matrix ──
    # We use COO during construction (cheap appends) then convert to CSR.
    row_indices: list = []
    col_indices: list = []
    data_values: list = []
    for head_index, relation_index, tail_index in kg.triples_idx:
        # head-of-relation slot
        row_indices.append(head_index)
        col_indices.append(2 * relation_index)
        data_values.append(1)
        # tail-of-relation slot
        row_indices.append(tail_index)
        col_indices.append(2 * relation_index + 1)
        data_values.append(1)
    count_matrix = coo_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(num_entities, 2 * num_relations),
        dtype=np.float32,
    ).tocsr()

    # ── Step 2: TF-IDF reweight ────────────────────────────────────────
    tfidf = TfidfTransformer(sublinear_tf=True)
    weighted_matrix: csr_matrix = tfidf.fit_transform(count_matrix)

    # ── Step 3: truncated SVD to svd_dim components ────────────────────
    # SVD also needs n_components < min(num_entities, num_features) - 1.
    max_components = min(weighted_matrix.shape) - 1
    effective_svd_dim = max(2, min(svd_dim, max_components))
    if effective_svd_dim < 2:
        # Degenerate case (tiny KG): skip SVD, work with the sparse matrix
        # densified - it's small enough to fit.
        dense_features = weighted_matrix.toarray()
    else:
        svd = TruncatedSVD(n_components=effective_svd_dim, random_state=random_state)
        dense_features = svd.fit_transform(weighted_matrix)

    # ── Step 4: cluster ───────────────────────────────────────────────
    kmeans = MiniBatchKMeans(
        n_clusters=effective_num_clusters,
        random_state=random_state,
        n_init="auto",
        batch_size=min(1024, num_entities),
    )
    cluster_assignments: np.ndarray = kmeans.fit_predict(dense_features)

    # Build the entity_id_string -> "type_NNN" map.
    # Pad cluster id with zeros so files sort sensibly.
    pad_width = max(2, len(str(effective_num_clusters - 1)))
    row_to_entity_id = {
        row_index: entity_id
        for entity_id, row_index in kg.entity_id_to_row.items()
    }
    pseudo_types: Dict[str, str] = {}
    for row_index, cluster_id in enumerate(cluster_assignments):
        pseudo_types[row_to_entity_id[row_index]] = f"type_{int(cluster_id):0{pad_width}d}"
    return pseudo_types


def write_entity_metadata(
    path: str | Path,
    pseudo_types: Dict[str, str],
    display_names: Optional[Dict[str, str]] = None,
) -> None:
    """Write the 3-column metadata file the existing loader reads.

    Format: `entity_id <TAB> display_name <TAB> type`, UTF-8, LF endings.
    If `display_names` is missing or doesn't include a given entity, the
    entity_id itself is used as the display name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for entity_id, type_string in pseudo_types.items():
            display_name = (display_names or {}).get(entity_id, entity_id)
            output_file.write(f"{entity_id}\t{display_name}\t{type_string}\n")
