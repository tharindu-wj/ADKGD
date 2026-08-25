"""Turn a triple file into Python objects. No scoring, no labels, no LLM."""


def load_triples(path):
    """Read a TSV of head/relation/tail ids, order preserved.

    Order matters: anything that scores triples later will return one value
    per triple in exactly this order.
    """
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                head, relation, tail = line.split("\t")
                triples.append((head, relation, tail))
    return triples
