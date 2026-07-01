"""Contradiction-bias training targets for the KGSAGE conditional GAN (B1a).

WHY THIS EXISTS
    Conditioning the generator on the context table E' only pays off if the
    TRAINING TARGET depends on E' — otherwise there is no gradient telling the
    generator to USE the context (see kgsage.gan.train's docstring). This module
    builds targets that are:
      * TYPE-VALID  — a legal filler for the relation (drawn from the relation's
                      observed head/tail entities), so the negative is a
                      believable near-miss, not obvious junk; and
      * CONTEXT-DISTANT — far from the anchor entity's neighbourhood context, so
                      the corrupted slot CONTRADICTS what the graph implies
                      (the "converging-context contradiction").

    Example (the drawing): for (Person P, lives_in, Country A) where P's whole
    neighbourhood points at Country A, the tail target becomes a type-valid but
    context-distant country (Country B) — the believable-but-false near-miss the
    ADKGD neighbourhood channel must learn to flag.

HOW THE DISTANCE WORKS
    RGCN message passing pulls graph-neighbours together in E', so an entity that
    is genuinely connected to the head (its true country) sits CLOSE to E'[head],
    while an unrelated-but-type-valid entity sits FAR. Weighting candidates by
    `1 - cos(E'[candidate], E'[anchor])` therefore prefers the contradictions.
    For a TAIL corruption the anchor is the HEAD (tail should contradict the
    head's context); for a HEAD corruption the anchor is the TAIL.

EFFICIENCY
    The slot assignment and the k_candidates type-valid fillers are fixed ONCE in
    __init__ (the only Python-side loops). Each epoch, `build_targets(E')` is pure
    tensor work: score the fixed candidates against the CURRENT E' and draw one
    per triple with probability proportional to context distance.
"""
import numpy as np
import torch
import torch.nn.functional as F


class ContradictionTargetSampler:
    """Builds context-distant, type-valid corruption targets, refreshed per epoch.

    Args:
        real_all     : LongTensor [N, 3] of real (head, relation, tail) triples.
        n_ent, n_rel : vocabulary sizes.
        k_candidates : how many type-valid fillers to consider per triple.
        seed         : seed for the (numpy) one-time slot/candidate sampling.
        device       : device to hold the precomputed tensors on.
    """

    def __init__(self, real_all, n_ent, n_rel, k_candidates=20, seed=0, device=None):
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.k_candidates = k_candidates
        self.device = device or real_all.device
        rng = np.random.default_rng(seed)

        triples = real_all.detach().cpu().numpy()
        n_triples = triples.shape[0]
        heads, relations, tails = triples[:, 0], triples[:, 1], triples[:, 2]

        # The real triples, one slot of which each target will overwrite.
        self.base = torch.as_tensor(triples, dtype=torch.long, device=self.device)

        # --- Type-valid pools: which entities actually appear as the head /
        # tail of each relation. Corrupting to a value from the pool keeps the
        # triple a legal near-miss. Relations never seen fall back to all entities.
        head_pool = {r: set() for r in range(n_rel)}
        tail_pool = {r: set() for r in range(n_rel)}
        for head, relation, tail in triples:
            head_pool[relation].add(head)
            tail_pool[relation].add(tail)
        all_entities = np.arange(n_ent, dtype=np.int64)
        head_pool = {r: (np.array(sorted(s), dtype=np.int64) if s else all_entities)
                     for r, s in head_pool.items()}
        tail_pool = {r: (np.array(sorted(s), dtype=np.int64) if s else all_entities)
                     for r, s in tail_pool.items()}

        # --- One-time slot assignment (0 = head, 1 = relation, 2 = tail),
        # uniform — matches the random baseline's slot distribution so that
        # B0 (random negatives) vs B1 (these) stays a one-variable comparison.
        self.slots = rng.integers(0, 3, size=n_triples)

        # --- Relation-slot targets: a fixed random DIFFERENT relation. Relations
        # have no context vector, so the contradiction bias does not apply to them.
        relation_replacement = relations.copy()
        for i in np.where(self.slots == 1)[0]:
            if n_rel > 1:
                new_relation = rng.integers(0, n_rel)
                while new_relation == relations[i]:
                    new_relation = rng.integers(0, n_rel)
                relation_replacement[i] = new_relation
        self.relation_replacement = torch.as_tensor(
            relation_replacement, dtype=torch.long, device=self.device)

        # --- Entity-slot candidates (fixed). For tail-slot triples: candidates
        # from tail_pool[r], anchor = head. For head-slot triples: candidates
        # from head_pool[r], anchor = tail.
        self.tail_rows = np.where(self.slots == 2)[0]
        self.head_rows = np.where(self.slots == 0)[0]

        self.tail_candidates = self._sample_pools(tail_pool, relations[self.tail_rows], rng)
        self.head_candidates = self._sample_pools(head_pool, relations[self.head_rows], rng)

        self.tail_anchor = torch.as_tensor(heads[self.tail_rows], dtype=torch.long, device=self.device)
        self.head_anchor = torch.as_tensor(tails[self.head_rows], dtype=torch.long, device=self.device)
        self.tail_true = torch.as_tensor(tails[self.tail_rows], dtype=torch.long, device=self.device)
        self.head_true = torch.as_tensor(heads[self.head_rows], dtype=torch.long, device=self.device)
        self.tail_rows_t = torch.as_tensor(self.tail_rows, dtype=torch.long, device=self.device)
        self.head_rows_t = torch.as_tensor(self.head_rows, dtype=torch.long, device=self.device)

    def _sample_pools(self, pool_arrays, relations_subset, rng):
        """Draw k_candidates type-valid fillers per triple -> LongTensor [n, k]."""
        k = self.k_candidates
        candidates = np.empty((len(relations_subset), k), dtype=np.int64)
        for i, relation in enumerate(relations_subset):
            pool = pool_arrays[relation]
            candidates[i] = pool[rng.integers(0, len(pool), size=k)]  # with replacement
        return torch.as_tensor(candidates, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def build_targets(self, entity_context):
        """Assemble [N, 3] targets against the CURRENT context table E'.

        Exactly one slot per row is overwritten (the slot fixed in __init__):
        entity slots get a context-distant type-valid filler; the relation slot
        gets its fixed random replacement.
        """
        targets = self.base.clone()
        # Relation slot (no-op for non-relation rows: replacement == original there).
        targets[:, 1] = self.relation_replacement
        # Tail slot: context-distant tail (anchor = head).
        if self.tail_rows_t.numel() > 0:
            targets[self.tail_rows_t, 2] = self._pick_context_distant(
                self.tail_candidates, self.tail_anchor, self.tail_true, entity_context)
        # Head slot: context-distant head (anchor = tail).
        if self.head_rows_t.numel() > 0:
            targets[self.head_rows_t, 0] = self._pick_context_distant(
                self.head_candidates, self.head_anchor, self.head_true, entity_context)
        return targets

    def _pick_context_distant(self, candidates, anchor, true_value, entity_context):
        """Sample one filler per row with probability proportional to context distance.

        Args:
            candidates    : LongTensor [n, k] type-valid fillers.
            anchor        : LongTensor [n] entity whose context we measure against.
            true_value    : LongTensor [n] the real slot value (never selected).
            entity_context: FloatTensor [n_ent, dim] current E'.

        Returns:
            LongTensor [n] chosen filler per row.
        """
        candidate_context = entity_context[candidates]          # [n, k, dim]
        anchor_context = entity_context[anchor].unsqueeze(1)    # [n, 1, dim]
        cosine = F.cosine_similarity(candidate_context, anchor_context, dim=2)  # [n, k]
        weights = (1.0 - cosine).clamp(min=1e-6)                # far-from-context = high weight

        # Never "corrupt" to the true value.
        weights = weights.masked_fill(candidates == true_value.unsqueeze(1), 0.0)
        # Guard: if a row is all-zero (every candidate equalled the true value),
        # fall back to a uniform draw so multinomial stays well-defined.
        row_has_weight = weights.sum(dim=1, keepdim=True) > 0
        weights = torch.where(row_has_weight, weights, torch.ones_like(weights))

        choice = torch.multinomial(weights, num_samples=1).squeeze(1)  # [n] index into k
        return candidates.gather(1, choice.unsqueeze(1)).squeeze(1)
