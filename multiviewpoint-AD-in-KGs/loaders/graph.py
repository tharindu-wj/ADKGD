"""Turn dataset files into objects. No scoring, no labels, no LLM."""
import collections


def load_triples(path):
    """Read a TSV of head/relation/tail.

    Order is preserved and load-bearing: every scorer returns one value per
    triple in exactly this order, and the scripts zip them back together.
    """
    with open(path, encoding="utf-8") as f:
        return [tuple(line.rstrip("\n").split("\t")) for line in f]


def adjacency(triples):
    """N[x] = everything x links to, either direction, any relation.

    Ignoring direction and relation type is deliberate: the question is "do
    these two move in the same circles", not "is there a specific path".
    """
    N = collections.defaultdict(set)
    for h, r, t in triples:
        N[h].add(t)
        N[t].add(h)
    return N
