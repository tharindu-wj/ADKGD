"""Scanner: entities whose kind does not fit the slot they sit in.

This scanner never reads meanings -- it only counts TYPE LABELS. "Odd"
means exactly one thing: the KIND of entity in this seat is not the kind
that fills most of this seat.

WORKED EXAMPLE (invented entities -- no dataset supplies these):

    tails of "stored at":   95% carry the type "warehouse"
    X --stored at-- lorry   <- "lorry" carries the type "vehicle"

A vehicle in a warehouse-seat may be an error or an unusual-but-true
record -- "odd", never "wrong"; the observer settles it. The signal earns
its keep because a genuinely WRONG fact often puts the wrong KIND of thing
in a slot: a disease sitting in a place-seat looks bad from the type
labels alone, no world knowledge required.

HOW IT COUNTS:
    1. For each relation in scope, collect the entities in each slot
       (heads and tails separately).
    2. Count type labels across the slot. dominant = the commonest type,
       covered = its share. Keep the slot only if covered is at least
       DOMINANCE -- with no dominant kind there is nothing to clash with.
    3. Point at each entity in that slot NOT carrying the dominant type --
       ONCE per entity, however many edges it sits in, so one repeat
       offender cannot flood a reading budget with copies of one question.

If nearly all of a relation's heads share one type, a head without that
type is what a typing norm is about. Dominance is measured from the data (no schema file exists),
so this is "unusual for this graph", not "invalid by decree" -- the judge
decides whether unusual is wrong.

Deterministic, no model, no labels.
"""
import collections

NAME = "odd_types"

#: a type must cover this share of a slot before an outsider is a clash
DOMINANCE = 0.8


def find(scope_ids, ctx):
    """Edges whose head or tail lacks its slot's dominant type."""
    heads_of = collections.defaultdict(list)
    tails_of = collections.defaultdict(list)
    for head, relation, tail in ctx.triples:
        if relation in scope_ids:
            heads_of[relation].append(head)
            tails_of[relation].append(tail)

    candidates = []
    for relation_id in heads_of:
        for slot_name, occupants in (("head", heads_of[relation_id]),
                                     ("tail", tails_of[relation_id])):
            counts = collections.Counter()
            for entity in occupants:
                for type_label in ctx.entity_types.get(entity) or []:
                    counts[type_label] += 1
            if not counts:
                continue
            dominant, covered = counts.most_common(1)[0]
            if covered / len(occupants) < DOMINANCE:
                continue                # no dominant type, nothing to clash with
            # One candidate PER OFFENDING ENTITY, not per edge. A single
            # entity can sit in dozens of edges of one relation and clash
            # identically in all of them (measured; see DESIGN.md) -- a judge
            # with a bounded budget must not be handed dozens of copies of
            # one question.
            edges_of_entity = collections.defaultdict(list)
            for triple in ctx.triples:
                if triple[1] != relation_id:
                    continue
                entity = triple[0] if slot_name == "head" else triple[2]
                if dominant not in (ctx.entity_types.get(entity) or []):
                    edges_of_entity[entity].append(triple)
            for entity, edges in sorted(edges_of_entity.items()):
                what = ", ".join(ctx.entity_types.get(entity) or []) or "untyped"
                note = (f"{slot_name} is {what}; "
                        f"{covered / len(occupants):.0%} of this relation's "
                        f"{slot_name}s are {dominant}")
                if len(edges) > 1:
                    note += (f" -- this entity sits in {len(edges)} such "
                             f"edges; judging one judges the pattern")
                candidates.append((edges[0], note))
    # A triple can clash in both slots; keep the first note it earned.
    unique = {}
    for triple, note in candidates:
        unique.setdefault(triple, note)
    return sorted(unique.items())
