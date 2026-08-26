"""Tool: everything known about one relation -- meaning AND shape."""
import collections

from loaders.context import get_context


def describe_relation(relation: str) -> str:
    """What one relation means, and how it behaves in this graph.

    Returns the description shipped with the dataset (what the relation MEANS),
    then what the data shows: triple count, distinct heads and tails, how many
    tails a head usually has and vice versa, how often triples appear reversed
    (symmetry), the commonest tails, and a few real examples.

    The description is knowledge; the counts are this graph. Neither says
    whether any particular triple is true.

    Args:
        relation: the relation's name, as listed by describe_dataset.
    """
    ctx = get_context()

    relation_id = ctx.find_relation(relation)
    if relation_id is None:
        known = ", ".join(ctx.all_relation_labels())
        return f"ERROR: no relation '{relation}'. Known relations: {known}."

    using = [t for t in ctx.triples if t[1] == relation_id]
    heads = {h for h, r, t in using}
    tails = {t for h, r, t in using}
    tail_counts = collections.Counter(t for h, r, t in using)

    # Symmetry: how many triples have their reverse present under the SAME
    # relation. High for relations that are mutual by nature.
    present = set(using)
    reversed_too = sum(1 for h, r, t in using if (t, r, h) in present)

    label = ctx.relation_label(relation_id)
    description = ctx.relations[relation_id]["description"]
    commonest = ", ".join(f"{ctx.entity_label(t)} ({n})"
                          for t, n in tail_counts.most_common(5))

    lines = [
        f"{label}",
        f"  meaning: {description or '(no description shipped)'}",
        "",
        f"  {len(using)} triples. {len(heads)} distinct heads, "
        f"{len(tails)} distinct tails.",
        f"  {len(using) / len(heads):.1f} tails per head, "
        f"{len(using) / len(tails):.1f} heads per tail.",
        f"  symmetry: {100 * reversed_too / len(using):.0f}% of its triples "
        f"also appear reversed.",
        f"  commonest tails: {commonest}",
        "",
        "  examples:",
    ]
    for triple in using[:3]:
        lines.append(f"    {ctx.triple_text(triple)}")
    return "\n".join(lines)
