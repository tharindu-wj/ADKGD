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
