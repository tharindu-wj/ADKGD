"""TRIC-style triple corruption (Senaratne et al., ESWC 2023).

This module turns a "clean" knowledge graph (a list of triples) into a
"corrupted" one by applying a few random edits. The corruptions follow the
TRIC taxonomy, where Table 1 of the paper lists 12 ways a triple can be
distorted. We implement the seven that apply to a binary directed multi-
relational graph (no literals).

A triple is a 3-tuple of integer indices: (head_index, relation_index,
tail_index). Each index is a row in an entity vocabulary or a relation
vocabulary - see kg_data.loader.load_kg for how those vocabularies are built.

The module returns BOTH:
  * the corrupted triple list (the new KG), and
  * a list of "edit records": (clean_triple_or_None, corrupted_triple_or_None)
    documenting each successful corruption step. The dataset builder uses
    those records to build (clean, corrupted) training pairs without having
    to re-diff the two KGs.

TRIC operations supported here:
    remove                  Delete a random existing triple.        (TRIC 1-4)
    swap_subject_same_type  (h, r, t) -> (h', r, t),
                            type(h') == type(h).                    (TRIC 1)
    swap_object_same_type   (h, r, t) -> (h, r, t'),
                            type(t') == type(t).                    (TRIC 2)
    change_relation         (h, r, t) -> (h, r', t).                (TRIC 3)
    swap_subject_diff_type  (h, r, t) -> (h', r, t),
                            type(h') != type(h).                    (TRIC 5)
    swap_object_diff_type   (h, r, t) -> (h, r, t'),
                            type(t') != type(t).                    (TRIC 6)
    add_spurious            Add a random non-existing triple.       (TRIC 9-10)
"""
from typing import Dict, List, Optional, Tuple

import numpy as np


# A triple is just a 3-tuple of integer indices.
# (head_index, relation_index, tail_index)
Triple = Tuple[int, int, int]

# An edit record documents one corruption step:
#   (clean_triple_before_edit, corrupted_triple_after_edit)
# For a "remove" op the second element is None (the triple is gone).
# For an "add_spurious" op the first element is None (no clean triple to start with).
# For substitutions (swap_*, change_relation) both are populated.
EditRecord = Tuple[Optional[Triple], Optional[Triple]]


# Names of every TRIC operation we support. Used as keys in op_weights.
_ALL_OPERATIONS = [
    "remove",
    "swap_subject_same_type",
    "swap_object_same_type",
    "change_relation",
    "swap_subject_diff_type",
    "swap_object_diff_type",
    "add_spurious",
]

# Operations that need node-type information (the "same_type" / "diff_type" swaps).
# These are silently dropped if the caller doesn't provide node_types.
_TYPE_AWARE_OPERATIONS = {
    "swap_subject_same_type",
    "swap_object_same_type",
    "swap_subject_diff_type",
    "swap_object_diff_type",
}

# Default mixing weights when node_types are available.
# Higher weight = operation gets picked more often.
_DEFAULT_OPERATION_WEIGHTS_WITH_TYPES = {
    "remove":                  2,
    "swap_subject_same_type":  2,
    "swap_object_same_type":   2,
    "change_relation":         1,
    "swap_subject_diff_type":  1,
    "swap_object_diff_type":   1,
    "add_spurious":            1,
}

# Default mixing weights when node_types are NOT available.
_DEFAULT_OPERATION_WEIGHTS_NO_TYPES = {
    "remove":          3,
    "change_relation": 2,
    "add_spurious":    1,
}


def apply_tric_corruption(
    clean_triples: List[Triple],
    num_corruption_steps: int,
    random_generator: np.random.Generator,
    *,
    num_entities: int,
    num_relations: int,
    node_types: Optional[Dict[int, str]] = None,
    operation_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[Triple], List[EditRecord]]:
    """Apply TRIC-style random corruption to a list of triples.

    Parameters
    ----------
    clean_triples
        The starting KG, as a list of (head_index, relation_index, tail_index).
        This list is NOT mutated; the function works on a copy.
    num_corruption_steps
        How many corruption operations to attempt. Some may be skipped if no
        valid target exists (e.g. trying to change_relation on a triple whose
        head and tail are already connected by every relation).
    random_generator
        numpy random Generator. Passing the same generator + same input gives
        a deterministic result, which is essential for reproducibility.
    num_entities, num_relations
        Sizes of the entity and relation vocabularies. Used to sample
        random replacement candidates for add_spurious and the swap ops.
    node_types
        Optional dict mapping entity_index -> type string (e.g. "Person",
        "Country"). Required by the type-aware swap operations; those ops
        are silently disabled if node_types is None.
    operation_weights
        Optional dict mapping operation name -> relative weight. If None,
        uses _DEFAULT_OPERATION_WEIGHTS_WITH_TYPES or _DEFAULT_OPERATION_WEIGHTS_NO_TYPES.
        Pass e.g. {"change_relation": 1} to enable ONLY change_relation.

    Returns
    -------
    corrupted_triples : list[Triple]
        The resulting KG after `num_corruption_steps` edits.
    edit_records : list[EditRecord]
        One record per *successful* edit. For pure adds the first element
        is None; for pure removes the second element is None; otherwise
        both are populated.
    """
    # ── Resolve the active operation distribution ──────────────────────────
    if operation_weights is None:
        base_weights = (
            _DEFAULT_OPERATION_WEIGHTS_WITH_TYPES if node_types is not None
            else _DEFAULT_OPERATION_WEIGHTS_NO_TYPES
        )
    else:
        # Caller can pass arbitrary keys; ignore any not in our known list
        base_weights = {
            name: weight for name, weight in operation_weights.items()
            if name in _ALL_OPERATIONS
        }

    # Drop type-aware operations if we don't have type information
    if node_types is None:
        active_operation_weights = {
            name: weight for name, weight in base_weights.items()
            if name not in _TYPE_AWARE_OPERATIONS
        }
    else:
        active_operation_weights = dict(base_weights)

    if not active_operation_weights:
        return list(clean_triples), []

    operation_names = list(active_operation_weights.keys())
    operation_probabilities = np.array(
        [active_operation_weights[name] for name in operation_names],
        dtype=float,
    )
    operation_probabilities /= operation_probabilities.sum()

    # ── Mutable working state ─────────────────────────────────────────────
    # We maintain TWO data structures that mirror each other:
    #   working_triples : the same triples in list form (preserves order,
    #                     supports random-index access)
    #   triple_set      : same triples in set form (gives O(1) "does this
    #                     edge exist?" lookups, which we need many times
    #                     when finding valid swap/change_relation targets)
    working_triples: List[Triple] = list(clean_triples)
    triple_set: set = set(working_triples)
    edit_records: List[EditRecord] = []

    def pick_existing_triple_index() -> Optional[int]:
        """Pick a random index into `working_triples`, or None if empty."""
        if not working_triples:
            return None
        return int(random_generator.integers(len(working_triples)))

    # ── Apply `num_corruption_steps` random edits ─────────────────────────
    for _ in range(num_corruption_steps):
        # Sample one operation per step from the weighted distribution
        chosen_operation = operation_names[
            random_generator.choice(len(operation_names), p=operation_probabilities)
        ]

        # ─── Pure delete: remove an existing triple ─────────────────────
        if chosen_operation == "remove":
            triple_index = pick_existing_triple_index()
            if triple_index is None:
                continue
            removed_triple = working_triples.pop(triple_index)
            triple_set.discard(removed_triple)
            edit_records.append((removed_triple, None))
            continue

        # ─── Pure add: invent a new triple that doesn't already exist ───
        if chosen_operation == "add_spurious":
            # Las Vegas style: keep sampling until we find a non-existing,
            # non-self-loop triple. Cap at 50 attempts to avoid hangs in
            # edge cases (very small entity/relation spaces).
            for _ in range(50):
                random_head_index     = int(random_generator.integers(num_entities))
                random_tail_index     = int(random_generator.integers(num_entities))
                random_relation_index = int(random_generator.integers(num_relations))
                candidate_triple = (
                    random_head_index, random_relation_index, random_tail_index,
                )
                # Reject self-loops and triples that already exist
                if (random_head_index != random_tail_index
                        and candidate_triple not in triple_set):
                    working_triples.append(candidate_triple)
                    triple_set.add(candidate_triple)
                    edit_records.append((None, candidate_triple))
                    break
            continue

        # ─── Substitution ops (swap or change_relation) ────────────────
        # All of these pick an existing triple and replace ONE of its
        # three components with a different valid one.
        triple_index = pick_existing_triple_index()
        if triple_index is None:
            continue
        head_index, relation_index, tail_index = working_triples[triple_index]
        new_triple: Optional[Triple] = None

        if chosen_operation == "change_relation":
            # Find a relation r' != r such that (h, r', t) doesn't already exist.
            valid_replacement_relations = [
                alternative_relation
                for alternative_relation in range(num_relations)
                if alternative_relation != relation_index
                and (head_index, alternative_relation, tail_index) not in triple_set
            ]
            if valid_replacement_relations:
                chosen_new_relation_index = int(
                    valid_replacement_relations[
                        random_generator.integers(len(valid_replacement_relations))
                    ]
                )
                new_triple = (head_index, chosen_new_relation_index, tail_index)

        elif chosen_operation == "swap_subject_same_type":
            # Replace the head with another entity of the SAME type, such
            # that the resulting (h', r, t) doesn't already exist.
            valid_replacement_heads = [
                candidate_head_index
                for candidate_head_index in range(num_entities)
                if candidate_head_index != head_index
                and candidate_head_index != tail_index
                and node_types.get(candidate_head_index) == node_types.get(head_index)
                and (candidate_head_index, relation_index, tail_index) not in triple_set
            ]
            if valid_replacement_heads:
                chosen_new_head_index = int(
                    valid_replacement_heads[
                        random_generator.integers(len(valid_replacement_heads))
                    ]
                )
                new_triple = (chosen_new_head_index, relation_index, tail_index)

        elif chosen_operation == "swap_object_same_type":
            valid_replacement_tails = [
                candidate_tail_index
                for candidate_tail_index in range(num_entities)
                if candidate_tail_index != tail_index
                and candidate_tail_index != head_index
                and node_types.get(candidate_tail_index) == node_types.get(tail_index)
                and (head_index, relation_index, candidate_tail_index) not in triple_set
            ]
            if valid_replacement_tails:
                chosen_new_tail_index = int(
                    valid_replacement_tails[
                        random_generator.integers(len(valid_replacement_tails))
                    ]
                )
                new_triple = (head_index, relation_index, chosen_new_tail_index)

        elif chosen_operation == "swap_subject_diff_type":
            valid_replacement_heads = [
                candidate_head_index
                for candidate_head_index in range(num_entities)
                if candidate_head_index != head_index
                and candidate_head_index != tail_index
                and node_types.get(candidate_head_index) != node_types.get(head_index)
                and (candidate_head_index, relation_index, tail_index) not in triple_set
            ]
            if valid_replacement_heads:
                chosen_new_head_index = int(
                    valid_replacement_heads[
                        random_generator.integers(len(valid_replacement_heads))
                    ]
                )
                new_triple = (chosen_new_head_index, relation_index, tail_index)

        elif chosen_operation == "swap_object_diff_type":
            valid_replacement_tails = [
                candidate_tail_index
                for candidate_tail_index in range(num_entities)
                if candidate_tail_index != tail_index
                and candidate_tail_index != head_index
                and node_types.get(candidate_tail_index) != node_types.get(tail_index)
                and (head_index, relation_index, candidate_tail_index) not in triple_set
            ]
            if valid_replacement_tails:
                chosen_new_tail_index = int(
                    valid_replacement_tails[
                        random_generator.integers(len(valid_replacement_tails))
                    ]
                )
                new_triple = (head_index, relation_index, chosen_new_tail_index)

        # If no valid target existed for the chosen op (e.g. all relations
        # already connect this head/tail pair), skip this step.
        if new_triple is None:
            continue

        clean_triple = working_triples[triple_index]
        triple_set.discard(clean_triple)
        triple_set.add(new_triple)
        working_triples[triple_index] = new_triple
        edit_records.append((clean_triple, new_triple))

    return working_triples, edit_records
