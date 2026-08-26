"""Tool: the definitions store, made queryable -- one name at a time."""
from loaders.context import get_context

#: how many close names to offer when a term is not found
MAX_SUGGESTIONS = 8


def explain_term(term: str) -> str:
    """What one name means: its description, its kind, and how it is used.

    Works for entities and relations alike. For an entity: its shipped
    description, what KINDS of thing it is (its types), and how often it
    appears in the graph. For a relation: its description and usage.

    Use it whenever a name in the data is one you want to understand before
    building on it. It tells you what things ARE -- never whether a
    particular fact about them is true.

    Args:
        term: an entity or relation name, exactly as the other tools show it.
    """
    ctx = get_context()
    term = (term or "").strip()
    if not term:
        return "ERROR: name something to look up."

    entity_id = ctx.find_entity(term)
    if entity_id is not None:
        return _describe_entity(ctx, entity_id)

    relation_id = ctx.find_relation(term)
    if relation_id is not None:
        return _describe_relation(ctx, relation_id)

    close = _close_names(ctx, term)
    if close:
        return (f"ERROR: nothing named '{term}'. "
                f"Close names: {', '.join(close)}.")
    return (f"ERROR: nothing named '{term}' in this graph's vocabulary. "
            f"Names come from describe_dataset and sample.")


def _describe_entity(ctx, entity_id):
    info = ctx.entities[entity_id]
    types = ", ".join(ctx.entity_types.get(entity_id) or []) or "(untyped)"

    as_head = [t for t in ctx.triples if t[0] == entity_id]
    as_tail = [t for t in ctx.triples if t[2] == entity_id]

    lines = [
        f"{info['label']}  (entity)",
        f"  is a: {types}",
        f"  description: {info['description'] or '(none shipped)'}",
        f"  appears as head in {len(as_head)} triples, "
        f"as tail in {len(as_tail)}.",
    ]
    for triple in (as_head + as_tail)[:3]:
        lines.append(f"    {ctx.triple_text(triple)}")
    return "\n".join(lines)


def _describe_relation(ctx, relation_id):
    info = ctx.relations[relation_id]
    using = [t for t in ctx.triples if t[1] == relation_id]
    lines = [
        f"{info['label']}  (relation)",
        f"  description: {info['description'] or '(none shipped)'}",
        f"  used by {len(using)} triples. "
        f"describe_relation('{info['label']}') has the full picture.",
    ]
    if using:
        lines.append(f"    {ctx.triple_text(using[0])}")
    return "\n".join(lines)


def _close_names(ctx, term):
    """Names containing the term, so a near-miss teaches the right spelling."""
    needle = term.lower()
    close = [info["label"] for info in ctx.relations.values()
             if needle in info["label"].lower()]
    close += [info["label"] for info in ctx.entities.values()
              if needle in info["label"].lower()]
    return close[:MAX_SUGGESTIONS]
