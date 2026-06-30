"""Partner-template provider for the KGSAGE discriminator.

A "partner template" answers: given an anchor triple (h, r, t), which relation
r' makes the role-swapped triple (t, r', h) a CONTRADICTION? This is the
generator's recon-prior supervision (Phase B of training).

Returned structure:

    { anchor_rel_id : [ (partner_rel_id, confidence), ... ] }

The miner derives everything from two length-1 rule confidences:

    sym_conf(r)     = P( r(t,h)  | r(h,t) )    symmetry confidence
    inv_conf(r, r') = P( r'(t,h) | r(h,t) )    inverse confidence

which are exactly the AMIE/AnyBURL standard-confidence values for the rules

    r(X,Y)  <= r(Y,X)        (symmetry)
    r'(Y,X) <= r(X,Y)        (inverse)

computed by pure counting -- no Java, no rule file, no external tool. This is
the same co-occurrence counting that data/audit_dataset.py (Test 1.3) performs;
the thresholds are imported from there so the two stay locked in step.
"""
from collections import defaultdict

# Import the thresholds from the audit so the rule miner and Test 1.3 can never
# drift apart -- they share the same anti-symmetric classification.
from kgsage.data.audit_dataset import MIN_SUPPORT, SYM_RATIO


def _count_cooccurrence(kg):
    """Shared first pass over training triples.

    Returns:
        support : {r: number of (h, r, t) anchors}
        coocur  : {(r, r'): number of anchors (h, r, t) that also have (t, r', h)}

    mine_partner_templates derives its templates from these two dicts. Relation
    filtering by min_support happens in the caller, not here.
    """
    triples = kg["triples_train"]

    support = defaultdict(int)
    edges_by_pair = defaultdict(set)         # (h, t) -> {relations going h -> t}
    for h, r, t in triples:
        support[r] += 1
        edges_by_pair[(h, t)].add(r)

    coocur = defaultdict(int)                # (r, r') -> reverse co-occurrence
    for h, r, t in triples:
        for r_prime in edges_by_pair.get((t, h), ()):   # relations going t -> h
            coocur[(r, r_prime)] += 1

    return support, coocur




def mine_partner_templates(kg, min_support=MIN_SUPPORT,
                           symmetry_threshold=SYM_RATIO,
                           symmetric_exclude_names=None):
    """v2-A provider -- contradiction templates from length-1 rule confidences.

    Pure-Python, KG-agnostic. For each relation r:
      - sym_conf(r) = P(r(t,h) | r(h,t)) decides whether r is symmetric.
          high          -> r symmetric    -> (t, r, h) is a TRUE fact, NOT a
                                              contradiction -> no template
          low / absent  -> r anti-symmetric -> (t, r, h) IS a contradiction
                                              -> emit a self role-swap template
      - real inverses (cross pairs with high inv_conf) are excluded as traps --
        a genuine inverse fact is not a contradiction.

    Crucially, a relation with NO reverse edge at all (sym_conf absent -> 0.0)
    still yields a template. That is the mechanism that makes inverse-removed
    KGs tractable: the observed miner returns nothing for such relations, the
    rule miner classifies them correctly as anti-symmetric.

    Args:
        symmetric_exclude_names : optional iterable of relation STRINGS known to
            be symmetric. Only needed on fully one-directional KGs (YAGO 4.5)
            where the data cannot reveal symmetry; a ~20-entry hand/LLM list. On
            FB15K-237 / WN18RR / NELL-995 leave it None -- symmetry is detected
            from the data (verified by verify_partner_templates.py).

    Returns: { anchor_rel: [(partner_rel, confidence), ...] }
    """
    support, coocur = _count_cooccurrence(kg)

    sym_conf = {}                            # r -> symmetry confidence
    inverse_pairs = set()                    # (r, r') real inverses -> exclude
    for (r, r_prime), c in coocur.items():
        if support[r] < min_support:
            continue
        conf = c / support[r]
        if r == r_prime:
            sym_conf[r] = conf
        elif conf >= symmetry_threshold:
            inverse_pairs.add((r, r_prime))

    # Optional name-based symmetry exclusion (the YAGO hybrid).
    exclude_ids = set()
    if symmetric_exclude_names:
        rel2id = kg["rel2id"]
        exclude_ids = {rel2id[s] for s in symmetric_exclude_names if s in rel2id}

    # Emit one self role-swap template per non-symmetric relation.
    templates = defaultdict(list)
    for r, sup in support.items():
        if sup < min_support:
            continue
        s = sym_conf.get(r, 0.0)
        if s >= symmetry_threshold or r in exclude_ids:
            continue                         # symmetric -> no template
        templates[r].append((r, 1.0 - s))    # (t, r, h) is the contradiction

    # Prune real-inverse traps WITHOUT materialising empty keys. (Assigning
    # `templates[r] = [...]` on a defaultdict would create an empty entry for a
    # symmetric relation that anchors an inverse pair -- the bug the verify
    # probe caught on FB15K-237's educational_institution relations.)
    for (r, r_prime) in inverse_pairs:
        if r in templates:
            templates[r] = [(rp, c) for (rp, c) in templates[r] if rp != r_prime]

    return {r: ts for r, ts in templates.items() if ts}
