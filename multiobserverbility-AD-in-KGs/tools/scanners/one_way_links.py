"""Scanner: one-way records of relations that are mostly two-way.

WORKED EXAMPLE (invented entities -- no dataset supplies these):

    recorded:  A --linked with-- B    and    B --linked with-- A
    recorded:  A --linked with-- C    but NOT C --linked with-- A

If "linked with" appears reversed for 90% of its edges, the A--C edge is
the odd one out: a bond of a mostly-two-way kind, written down on one side
only. Whether that is a half-recorded TRUE fact or a fabricated edge (fakes
are usually one-way -- nobody plants the reverse) is the observer's call,
never this scanner's.

HOW IT COUNTS:
    1. Group the scope's triples by relation.
    2. Put each relation's (head, tail) pairs in a set; an edge is ONE-WAY
       when its (tail, head) mirror is absent from that set.
    3. symmetry = 1 - one_way/total. Keep the relation only if symmetry is
       at least MIN_SYMMETRY -- below that, one-way IS the relation's
       normal shape, not a gap.
    4. Emit every one-way edge, round-robin across relations, so one bulky
       relation cannot bury the others' gaps pages deep.

If 98% of a relation's triples appear in both directions, that relation is
mutual by usage -- and the remaining one-way edges are exactly what a
mutuality norm is about. Note that a planted FALSE fact on a mutual relation
is usually one-way too (nobody planted its reverse), so this scanner can
surface falsehoods as a side effect. The judge decides which is which.

Deterministic, no model, no labels.
"""
import collections

NAME = "one_way_links"

#: a relation must be at least this symmetric before its one-way edges count.
#: Below this, one-way is the relation's normal shape, not a gap.
MIN_SYMMETRY = 0.5


def find(scope_ids, ctx):
    """All one-way edges on mostly-symmetric relations, INTERLEAVED.

    Round-robin across relations rather than exhausting one at a time.
    Measured reason (see DESIGN.md): one bulky relation can hold hundreds of
    gaps, burying every other relation's gaps pages deep -- a bounded reading
    budget never saw them. A page should be a cross-section of the scope,
    not the front of its largest queue.
    """
    by_relation = collections.defaultdict(list)
    for triple in ctx.triples:
        if triple[1] in scope_ids:
            by_relation[triple[1]].append(triple)

    queues = []
    for relation_id, triples in by_relation.items():
        present = set(triples)
        one_way = [t for t in triples if (t[2], t[1], t[0]) not in present]
        symmetry = 1 - len(one_way) / len(triples)
        if symmetry < MIN_SYMMETRY or not one_way:
            continue
        note = (f"recorded one way only, on a relation that is "
                f"{symmetry:.0%} two-way")
        queues.append((symmetry, [(t, note) for t in sorted(one_way)]))

    # Most-symmetric relation leads each round, then round-robin.
    queues.sort(key=lambda q: -q[0])
    candidates = []
    position = 0
    while any(position < len(queue) for _, queue in queues):
        for _, queue in queues:
            if position < len(queue):
                candidates.append(queue[position])
        position += 1
    return candidates
