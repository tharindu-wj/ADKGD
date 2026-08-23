"""Tool: what relations exist, and what shape each one is.

Returns text, because an agent reads it. Errors come back as text too, so the
agent can read the mistake and correct itself rather than crashing the run.
"""
from loaders import graph
from loaders.active import DATASET
from utils import profile


def list_relations() -> str:
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
