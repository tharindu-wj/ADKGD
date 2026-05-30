"""Build (clean_triple, target_triple) training pairs directly from a triple list.

For each round (we do `num_rounds` of them) we do three things:
  1. Optionally permute entity indices - data augmentation.
  2. Apply TRIC corruption to the (permuted) triple list. TRIC returns the
     modified triple list AND a list of "edit records" - one per successful
     corruption step.
  3. Harvest the SUBSTITUTION edits (records where both clean and corrupted
     are populated) and turn them into rows of a (M, 6) long tensor:
        [clean_h, clean_r, clean_t,  target_h, target_r, target_t]
     That tensor is what the model trains on.

There is no adjacency tensor anywhere in this builder - the whole flow is
triple-level, which is what makes it scale to FB15k-237 size.
"""
from typing import Callable, List, Tuple

import numpy as np
import torch


# (head_index, relation_index, tail_index)
Triple = Tuple[int, int, int]


def build_triple_pair_dataset(
    clean_triples: List[Triple],
    num_rounds: int,
    num_corruption_steps_per_round: int,
    random_generator: np.random.Generator,
    *,
    num_entities: int,
    num_relations: int,
    corruption_fn: Callable,
    permute_entities: bool = True,
) -> torch.Tensor:
    """Build a (M, 6) long tensor of (clean, corrupted) triple index pairs.

    Parameters
    ----------
    clean_triples
        The clean KG as a list of (head_index, relation_index, tail_index).
    num_rounds
        How many "passes" to do. Each pass optionally permutes the KG and
        applies TRIC. Roughly: more rounds -> more training pairs.
    num_corruption_steps_per_round
        How many edits TRIC attempts per round. Passed straight through to
        corruption_fn.
    random_generator
        numpy random Generator. Same generator + same input -> same output.
    num_entities, num_relations
        Sizes of the entity / relation vocabularies. Needed for permutation
        and also passed to corruption_fn (which needs them for add_spurious
        and swap ops).
    corruption_fn
        A callable with signature ``corruption_fn(triples, num_corruption_steps,
        rng) -> (corrupted_triples, edit_records)``. Typically TRIC pre-bound
        with functools.partial. The returned edit_records are what we
        actually harvest training pairs from.
    permute_entities
        If True (default), each round random-remaps the entity index space.
        This is a cheap data-augmentation trick: the same KG structure is
        presented to the model with many different entity ID assignments,
        which forces the model to learn the *pattern* rather than specific
        IDs. Without this, the model would only see 18 distinct clean
        triples in the dummy KG and risk overfitting.

    Returns
    -------
    torch.Tensor of shape (M, 6), dtype torch.long.
    Columns: [clean_h, clean_r, clean_t, target_h, target_r, target_t].

    M = total number of substitution-style edits harvested across all rounds.
    Pure adds and pure removes (which lack a (clean, corrupted) pair) are
    excluded.
    """
    # All rows we'll eventually stack into the output tensor
    triple_pair_rows: List[List[int]] = []

    for _ in range(num_rounds):
        # ── Step 1: optionally permute entity indices ─────────────────
        if permute_entities:
            # entity_permutation[i] = where original entity i lands.
            # If perm = [3, 7, 1, ...] then "Alice" (originally row 0)
            # gets row 3 in this round.
            entity_permutation = random_generator.permutation(num_entities)
            current_triples = [
                (
                    int(entity_permutation[head_index]),
                    relation_index,
                    int(entity_permutation[tail_index]),
                )
                for head_index, relation_index, tail_index in clean_triples
            ]
        else:
            current_triples = list(clean_triples)

        # ── Step 2: apply TRIC corruption ─────────────────────────────
        # corruption_fn returns (corrupted_triples, edit_records).
        # We only need edit_records here; the corrupted KG itself is
        # implicit in the records.
        _corrupted_triples, edit_records = corruption_fn(
            current_triples, num_corruption_steps_per_round, random_generator,
        )

        # ── Step 3: harvest substitution edits as training rows ───────
        # Each "edit" looks like (clean_triple, corrupted_triple). For pure
        # adds, clean_triple is None; for pure removes, corrupted_triple is
        # None. We only keep edits where BOTH are populated, because the
        # model trains on (input, target) pairs.
        for clean_triple, corrupted_triple in edit_records:
            if clean_triple is None or corrupted_triple is None:
                continue   # skip pure add/remove ops - no training pair to form
            triple_pair_rows.append([
                clean_triple[0],     clean_triple[1],     clean_triple[2],
                corrupted_triple[0], corrupted_triple[1], corrupted_triple[2],
            ])

    # Empty case - no successful corruptions across all rounds
    if not triple_pair_rows:
        return torch.zeros((0, 6), dtype=torch.long)

    return torch.tensor(triple_pair_rows, dtype=torch.long)
