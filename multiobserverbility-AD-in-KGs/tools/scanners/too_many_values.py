"""Scanner: entities with several values where one is the rule.

WORKED EXAMPLE (invented entities -- no dataset supplies these):

    M1 --serial number-- SN-100
    M1 --serial number-- SN-733     <- two serials, on a relation where
                                       96% of entities have exactly one

BOTH edges are served, and the note lists both values, because the anomaly
is the PAIR -- either edge alone looks perfectly normal. A duplicate can be
a recording error, a planted extra value, or a legitimately double-valued
case; the observer decides which. (Planting a fake value onto an entity
that already has a real one CREATES this pattern, which is why this simple
count often out-catches cleverer machinery.)

HOW IT COUNTS:
    1. Group the scope's triples by relation, then by head entity.
    2. Count heads holding exactly one value. Keep the relation only if
       that share is at least MOSTLY_SINGLE -- otherwise several values is
       the relation's normal shape.
    3. For every kept relation, emit each edge of every head holding two
       or more values, noting all of that head's values.

Some relations hold one value per entity in almost every record. Where a
relation is single-valued for at least MOSTLY_SINGLE of its heads, heads
carrying two or more values are what a cardinality norm is about. Each offending EDGE is a candidate (verdicts are
per triple); its note lists all the values, because the anomaly is the pair,
not either edge alone.

Deterministic, no model, no labels.
"""
import collections

NAME = "too_many_values"

#: a relation counts as "typically single-valued" when this share of its
#: heads carry exactly one value
MOSTLY_SINGLE = 0.9


def find(scope_ids, ctx):
    """Every edge of a multi-valued head on a typically-single relation."""
    tails_of_head = collections.defaultdict(lambda: collections.defaultdict(list))
    for head, relation, tail in ctx.triples:
        if relation in scope_ids:
            tails_of_head[relation][head].append(tail)

    candidates = []
    for relation_id, heads in tails_of_head.items():
        single = sum(1 for tails in heads.values() if len(tails) == 1)
        if single / len(heads) < MOSTLY_SINGLE:
            continue                    # multi-valued is normal here
        for head, tails in sorted(heads.items()):
            if len(tails) < 2:
                continue
            listed = ", ".join(ctx.entity_label(t) for t in sorted(tails))
            note = (f"{len(tails)} values on a relation where "
                    f"{single / len(heads):.0%} of entities have one: {listed}")
            for tail in sorted(tails):
                candidates.append(((head, relation_id, tail), note))
    return candidates
