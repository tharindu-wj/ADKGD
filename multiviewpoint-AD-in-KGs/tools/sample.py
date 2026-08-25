"""Tool: a few example triples.

Deliberately capped. On a graph small enough to read end to end, an agent can
find the errors directly and the scorers become decoration -- the cap keeps the
architecture honest at fixture scale as well as at real scale.
"""
import numpy as np

from loaders import graph
from loaders.active import DATASET
from utils import profile

MAX = 10


def sample(relation: str = None, n: int = 5, seed: int = 0) -> str:
    """A few real triples, so you can see what the data looks like.

    Returns up to 10 triples, drawn at random, optionally from one relation.
    Use it to find out what the entity names actually are before you decide
    what a sound triple looks like.

    The cap is deliberate and low. This shows you the SHAPE of the data, not
    its contents -- you cannot audit a graph by reading it through this tool,
    and the triples you get back are unremarkable ones, not suspicious ones.

    Args:
        relation: restrict to one relation. Omit for a sample of the whole graph.
        n: how many triples to return. Capped at 10.
        seed: change it for a different random draw.
    """
    triples = graph.load_triples(DATASET.KG)
    if relation is not None:
        known = [r["relation"] for r in profile.relation_summary(triples)]
        if relation not in known:
            return f"ERROR: no relation '{relation}'. Known relations: {', '.join(known)}."
        triples = [x for x in triples if x[1] == relation]

    n = max(1, min(int(n), MAX))
    rng = np.random.default_rng(seed)
    picked = rng.permutation(len(triples))[:n]
    what = relation or "any relation"

    lines = [f"{n} of {len(triples)} triples with {what} (capped at {MAX}):"]
    for i in picked:
        lines.append("  " + "\t".join(triples[i]))
    return "\n".join(lines)
