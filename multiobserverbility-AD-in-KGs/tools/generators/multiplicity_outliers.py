"""Generator: entities with several values where one is the rule.

A person has one place of birth. If a relation is single-valued for at least
MOSTLY_SINGLE of its heads, then heads carrying two or more values are what a
cardinality norm is about. Each offending EDGE is a candidate (verdicts are
per triple); its note lists all the values, because the anomaly is the pair,
not either edge alone.

Deterministic, no model, no labels.
"""
import collections

NAME = "multiplicity_outliers"

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
