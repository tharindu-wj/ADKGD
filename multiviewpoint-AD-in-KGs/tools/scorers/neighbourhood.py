"""Do the two endpoints already share connections? LOW = anomalous.

    chad locatedin africa   8 of chad's 9 links also touch africa -> 0.889
    chad locatedin europe   0 of chad's 9 links touch europe      -> 0.000

Evidence: the graph itself. No model, no torch.
"""
from loaders import graph

NAME = "neighbourhood"
NEEDS_MODEL = False
DIRECTION = -1          # a LOW value is anomalous


def score(triples, **params):
    """Share of the head's connections that also connect to the tail.

    Leave-one-out: the edge under test is dropped from both sides first, or a
    triple would appear in its own evidence and vouch for itself.
    """
    N = graph.adjacency(triples)
    out = []
    for h, r, t in triples:
        nh = N[h] - {t}
        nt = N[t] - {h}
        out.append(len(nh & nt) / len(nh) if nh else 0.0)
    return out
