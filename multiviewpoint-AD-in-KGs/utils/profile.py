"""Describe a graph by counting it. No LLM, no labels, no model.

This is what an agent reasons from before it decides anything. Everything here
is a pure function of the triples it is handed.
"""
import collections


def _pools(triples):
    heads = collections.defaultdict(set)
    tails = collections.defaultdict(set)
    for h, r, t in triples:
        heads[r].add(h)
        tails[r].add(t)
    return heads, tails


def cardinality(triples, rel):
    """one-to-one / one-to-many / many-to-one / many-to-many.

    Read off the two fan-outs: how many tails an average head has, and how many
    heads an average tail has. A value near 1 means that side is determined.
    """
    per_head = collections.Counter()
    per_tail = collections.Counter()
    for h, r, t in triples:
        if r != rel:
            continue
        per_head[h] += 1
        per_tail[t] += 1
    tph = sum(per_head.values()) / len(per_head) if per_head else 0.0
    hpt = sum(per_tail.values()) / len(per_tail) if per_tail else 0.0
    many_t, many_h = tph > 1.5, hpt > 1.5
    name = {(False, False): "one-to-one", (True, False): "one-to-many",
            (False, True): "many-to-one", (True, True): "many-to-many"}[(many_t, many_h)]
    return {"tails_per_head": tph, "heads_per_tail": hpt, "cardinality": name}


def symmetry(triples, rel):
    """Share of this relation's triples whose reverse is also present.

    Near 1 means the relation is symmetric in this graph (bordering, married).
    Near 0 means it is directional (contained in, born in).
    """
    present = {(h, t) for h, r, t in triples if r == rel}
    if not present:
        return 0.0
    return sum(1 for h, t in present if (t, h) in present) / len(present)


def tail_purity(triples, rel):
    """How exclusively this relation owns the tails it uses. 1.0 = completely.

    Set membership does NOT work here. On a dirty graph every wrong triple adds
    its tail to the pool, so a handful of errors makes two disjoint vocabularies
    look 83% shared -- measured on this fixture, where the true overlap is zero.

    Frequency survives that. A real region appears as a locatedin tail sixty
    times and as a neighbor tail once; a country wrongly placed in a locatedin
    slot appears there once and as a neighbor tail four times. Averaged over the
    relation's triples, the rare intruders barely move the number.
    """
    as_tail_of = collections.Counter()
    as_tail_anywhere = collections.Counter()
    for h, r, t in triples:
        as_tail_of[(r, t)] += 1
        as_tail_anywhere[t] += 1

    mine = [(t, n) for (r, t), n in as_tail_of.items() if r == rel]
    if not mine:
        return 0.0
    # weighted by how often each tail is used, so one odd tail cannot dominate
    total = sum(n for _, n in mine)
    return sum(n * (n / as_tail_anywhere[t]) for t, n in mine) / total


def relation_summary(triples):
    """One row per relation. What list_relations shows."""
    heads, tails = _pools(triples)
    counts = collections.Counter(r for _, r, _ in triples)
    out = []
    for rel in sorted(counts):
        card = cardinality(triples, rel)
        out.append({"relation": rel, "triples": counts[rel],
                    "heads": len(heads[rel]), "tails": len(tails[rel]),
                    "cardinality": card["cardinality"]})
    return out


def relation_detail(triples, rel):
    """Everything known about one relation. What describe_relation shows."""
    heads, tails = _pools(triples)
    if rel not in heads:
        return None
    card = cardinality(triples, rel)
    tail_freq = collections.Counter(t for _, r, t in triples if r == rel)
    common = tail_freq.most_common(5)
    return {"relation": rel,
            "triples": sum(1 for _, r, _ in triples if r == rel),
            "heads": len(heads[rel]), "tails": len(tails[rel]),
            **card,
            "symmetry": symmetry(triples, rel),
            "tail_purity": tail_purity(triples, rel),
            "tails_seen_once": sum(1 for v in tail_freq.values() if v == 1),
            "commonest_tails": common}


def graph_summary(triples):
    """Size of the whole thing."""
    ents = {x for h, _, t in triples for x in (h, t)}
    return {"triples": len(triples), "entities": len(ents),
            "relations": len({r for _, r, _ in triples})}
