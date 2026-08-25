"""Tool: everything countable about one relation."""
from loaders import graph
from loaders.active import DATASET
from utils import profile


def describe_relation(relation: str) -> str:
    """Everything countable about one relation.

    Returns its triple count, how many distinct heads and tails it has, its
    cardinality, what share of its triples also appear reversed (symmetry),
    how exclusive its tail vocabulary is to it rather than shared with other
    relations, how many tails occur only once, and its commonest tails.

    Use it on any relation your goal concerns. Symmetry and cardinality say
    what SHAPE a sound triple has; the commonest tails show you what kind of
    thing sits in that slot. None of it is evidence about a specific triple.

    Args:
        relation: the relation name, exactly as list_relations spells it.
    """
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
