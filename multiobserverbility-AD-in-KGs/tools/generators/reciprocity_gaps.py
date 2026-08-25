"""Generator: one-way records of relations that are mostly two-way.

If 98% of a relation's triples appear in both directions, that relation is
mutual by usage -- and the remaining one-way edges are exactly what a
mutuality norm is about. Note that a planted FALSE fact on a mutual relation
is usually one-way too (nobody planted its reverse), so this generator can
surface falsehoods as a side effect. The judge decides which is which.

Deterministic, no model, no labels.
"""
import collections

NAME = "reciprocity_gaps"

#: a relation must be at least this symmetric before its one-way edges count.
#: Below this, one-way is the relation's normal shape, not a gap.
MIN_SYMMETRY = 0.5


def find(scope_ids, ctx):
    """All one-way edges on mostly-symmetric relations, worst-first.

    Ordered by the relation's symmetry, descending -- a single one-way edge
    on a 98%-symmetric relation is stranger than one on a 60%-symmetric one.
    """
    by_relation = collections.defaultdict(list)
    for triple in ctx.triples:
        if triple[1] in scope_ids:
            by_relation[triple[1]].append(triple)

    candidates = []
    for relation_id, triples in by_relation.items():
        present = set(triples)
        one_way = [t for t in triples if (t[2], t[1], t[0]) not in present]
        symmetry = 1 - len(one_way) / len(triples)
        if symmetry < MIN_SYMMETRY or not one_way:
            continue
        for head, relation, tail in one_way:
            note = (f"recorded one way only, on a relation that is "
                    f"{symmetry:.0%} two-way")
            candidates.append((symmetry, (head, relation, tail), note))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [(triple, note) for _, triple, note in candidates]
