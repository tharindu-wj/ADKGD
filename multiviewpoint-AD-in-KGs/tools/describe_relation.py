"""Tool: everything countable about one relation."""
from loaders import graph
from loaders.active import DATASET
from utils import profile


def describe_relation(relation: str) -> str:
    triples = graph.load_triples(DATASET.KG)
    d = profile.relation_detail(triples, relation)
    if d is None:
        known = ", ".join(r["relation"] for r in profile.relation_summary(triples))
        return f"ERROR: no relation '{relation}'. Known relations: {known}."

    common = ", ".join(f"{t} ({n})" for t, n in d["commonest_tails"])
    return "\n".join([
        f"{d['relation']}: {d['triples']} triples.",
        f"  {d['heads']} distinct heads, {d['tails']} distinct tails.",
        f"  shape: {d['cardinality']} "
        f"({d['tails_per_head']:.1f} tails per head, "
        f"{d['heads_per_tail']:.1f} heads per tail)",
        f"  symmetry: {d['symmetry']:.0%} of its triples have their reverse present",
        f"  tail vocabulary: {d['tail_purity']:.0%} exclusive to this relation",
        f"  {d['tails_seen_once']} of its tails appear exactly once",
        f"  commonest tails: {common}",
    ])
