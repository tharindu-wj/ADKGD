"""Tool: a few example triples, resolved to readable names.

Deliberately capped. On a graph small enough to read end to end an agent
could find anomalies directly and the pipeline would prove nothing -- the
cap keeps this a peephole, not a window.
"""
import random

from loaders.context import get_context

MAX_TRIPLES = 10


def inspect_triples(relation: str = None, n: int = 5, seed: int = 0) -> str:
    """A few real triples, so you can see what the data looks like.

    Returns up to 10 triples drawn at random, optionally from one relation,
    every id resolved to its label. Use it to see what actually sits in a
    relation before deciding anything about it.

    The triples that come back are ordinary ones, not suspicious ones -- this
    shows the data's shape, and you cannot survey the whole graph through it.

    Args:
        relation: restrict to one relation. Omit for the whole graph.
        n: how many triples. Capped at 10.
        seed: change for a different random draw.
    """
    ctx = get_context()

    pool = ctx.triples
    where = "the whole graph"
    if relation is not None:
        relation_id = ctx.find_relation(relation)
        if relation_id is None:
            known = ", ".join(ctx.all_relation_labels())
            return f"ERROR: no relation '{relation}'. Known relations: {known}."
        pool = [t for t in ctx.triples if t[1] == relation_id]
        where = f"'{ctx.relation_label(relation_id)}'"

    n = max(1, min(int(n), MAX_TRIPLES, len(pool)))
    chosen = random.Random(seed).sample(pool, n)

    lines = [f"{n} of {len(pool)} triples from {where} (capped at {MAX_TRIPLES}):"]
    for triple in chosen:
        lines.append(f"  {ctx.triple_text(triple)}")
    return "\n".join(lines)
