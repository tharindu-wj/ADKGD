"""Tool: the whole dataset at a glance.

Returns text, because an agent reads it. Errors from every tool in this
project come back as text too, so an agent can read its mistake and correct
itself instead of crashing the run.
"""
import collections

from loaders.active import DATASET
from loaders.context import get_context


def describe_dataset() -> str:
    """Every relation in the graph, named and counted. Start here.

    Returns the dataset totals, then one row per relation: its name, how many
    triples use it, and how many distinct head and tail entities it connects.
    All names are real labels -- use them as-is with the other tools.

    This is the only view of the whole graph you get. It tells you what
    exists and how much; it says nothing about whether any triple is true.
    """
    ctx = get_context()

    triples_using = collections.Counter()
    heads_of = collections.defaultdict(set)
    tails_of = collections.defaultdict(set)
    for head, relation, tail in ctx.triples:
        triples_using[relation] += 1
        heads_of[relation].add(head)
        tails_of[relation].add(tail)

    # The one sentence of domain comes from the dataset's own CARD -- the
    # single sanctioned home for dataset-specific text. Hardcoding it here
    # once leaked one dataset's domain into every dataset's tool.
    lines = [f"{len(ctx.triples)} triples, {len(ctx.entities)} entities, "
             f"{len(ctx.relations)} relations. {DATASET.CARD}", ""]
    lines.append(f"{'relation':<42}{'triples':>9}{'heads':>8}{'tails':>8}")
    for relation_id, count in triples_using.most_common():
        label = ctx.relation_label(relation_id)
        lines.append(f"{label:<42}{count:>9}"
                     f"{len(heads_of[relation_id]):>8}"
                     f"{len(tails_of[relation_id]):>8}")
    lines.append("")
    lines.append("Use describe_relation(name) for what one of them means, "
                 "with examples.")
    return "\n".join(lines)
