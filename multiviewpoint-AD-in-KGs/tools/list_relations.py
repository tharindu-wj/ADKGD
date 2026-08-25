"""Tool: what relations exist, and what shape each one is.

Returns text, because an agent reads it. Errors come back as text too, so the
agent can read the mistake and correct itself rather than crashing the run.
"""
from loaders import graph
from loaders.active import DATASET
from utils import profile


def list_relations() -> str:
    """Every relation in the graph, with its size and shape. Start here.

    Returns the totals for the whole graph, then one row per relation: how many
    triples use it, how many distinct heads and tails it has, and whether it is
    one-to-one, many-to-one or many-to-many.

    This is the only view of the whole graph you get. It tells you what the
    relations ARE called and how they behave; it tells you nothing about
    whether any particular triple is right.
    """
    triples = graph.load_triples(DATASET.KG)
    g = profile.graph_summary(triples)
    rows = profile.relation_summary(triples)

    lines = [f"{g['triples']} triples, {g['entities']} entities, "
             f"{g['relations']} relations.", ""]
    lines.append(f"{'relation':<16}{'triples':>9}{'heads':>8}{'tails':>8}   shape")
    for r in rows:
        lines.append(f"{r['relation']:<16}{r['triples']:>9}{r['heads']:>8}"
                     f"{r['tails']:>8}   {r['cardinality']}")
    lines.append("")
    lines.append("Use describe_relation(name) for one of them in detail.")
    return "\n".join(lines)
