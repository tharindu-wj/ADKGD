"""Build candidate sets for training-time GAN sampling.

For each positive (h, r, t), we produce:
  1. A choice of which slot to corrupt (head or tail) using Bordes-style
     Bernoulli sampling driven by the relation's cardinality stats.
  2. A set of N_S type-coherent candidate entity IDs to fill that slot,
     drawn from headPool[r] or tailPool[r] (built in Phase 1).

The 3-step build per positive:

  STEP 1: Choose slot via Bernoulli(p_head) where
            p_head = tph / (tph + hpt)
            tph = avg tails per head, hpt = avg heads per tail.
          (For 1-N relations: tph >> hpt -> p_head high -> corrupt head
           more often. Replacing the tail of a 1-N relation risks
           generating a triple that is TRUE but happens to be absent
           from training - a "false negative" that confuses learning.
           Same idea symmetrically for N-1.)

  STEP 2: Look up the relevant pool (headPool[r] or tailPool[r]),
          exclude the clean entity, sample N_S candidates uniformly.
          (If the pool is smaller than N_S, fall back to sampling with
          replacement.)

  STEP 3: Return (slot, candidate_ids, weights).
          Weights are uniform 1.0 - the cardinality bias is already
          handled at the slot-selection step (STEP 1). Phase 2.3's
          training loop can post-process these weights if a finer-
          grained per-candidate weighting is needed later.
"""
import random

import torch


DEFAULT_N_CANDIDATES = 64


def choose_slot(r_id, concept_pools, rng=None):
    """Bernoulli sample which slot to corrupt for relation r.

    Returns "head" or "tail". Uses cardinality_stats from Phase 1 to bias
    the choice (Bordes 2013 convention).
    """
    if rng is None:
        rng = random.Random()
    stats = concept_pools.get("cardinality_stats", {}).get(r_id)
    if not stats:
        return "head" if rng.random() < 0.5 else "tail"
    tph = stats["avg_tails_per_head"]
    hpt = stats["avg_heads_per_tail"]
    total = tph + hpt
    if total <= 0:
        return "head" if rng.random() < 0.5 else "tail"
    p_head = tph / total
    return "head" if rng.random() < p_head else "tail"


def build_candidates(positive, concept_pools, slot=None,
                     n_candidates=DEFAULT_N_CANDIDATES, rng=None):
    """Build a candidate set for one positive triple.

    Args:
      positive:        (h_id, r_id, t_id) integer triple.
      concept_pools:   dict loaded from concept_pools.pkl.
      slot:            "head", "tail", or None (sample via Bernoulli).
      n_candidates:    N_S - how many candidates to draw.
      rng:             Python Random for determinism (optional).

    Returns:
      slot:            "head" or "tail" (may differ from input if the
                       requested slot's pool was degenerate).
      candidate_ids:   LongTensor of shape [n_candidates].
      weights:         FloatTensor of shape [n_candidates] (currently 1.0).

    Robustness notes:
      Some narrow FB15K-237 relations have a single entity in one of the
      pools, which is the clean one. We handle this gracefully:
        1. Try the requested (or Bernoulli-sampled) slot first.
        2. If that pool empties after excluding the clean entity, try
           the OTHER slot.
        3. If both pools are degenerate (extremely rare), fall back to
           uniform-random sampling across all entities for the original
           slot. This matches the legacy GAN's fallback behavior and
           lets training continue without a crash.
    """
    if rng is None:
        rng = random.Random()

    h_id, r_id, t_id = positive

    first_slot = slot if slot is not None else choose_slot(r_id, concept_pools, rng)
    other_slot = "tail" if first_slot == "head" else "head"

    # Try requested slot, then the other, then full-vocab fallback.
    for try_slot in (first_slot, other_slot):
        pool = (concept_pools["headPool"][r_id] if try_slot == "head"
                else concept_pools["tailPool"][r_id])
        clean_id = h_id if try_slot == "head" else t_id
        available = [e for e in pool if e != clean_id]
        if not available:
            continue
        if len(available) >= n_candidates:
            sampled = rng.sample(available, n_candidates)
        else:
            sampled = [rng.choice(available) for _ in range(n_candidates)]
        candidate_ids = torch.tensor(sampled, dtype=torch.long)
        weights = torch.ones(n_candidates, dtype=torch.float32)
        return try_slot, candidate_ids, weights

    # Both type-coherent pools degenerate. Last-resort: uniform random
    # across the full entity vocab on the originally-requested slot.
    n_entities = len(concept_pools["entity_to_id"])
    clean_id = h_id if first_slot == "head" else t_id
    sampled = []
    while len(sampled) < n_candidates:
        cand = rng.randint(0, n_entities - 1)
        if cand != clean_id:
            sampled.append(cand)
    candidate_ids = torch.tensor(sampled, dtype=torch.long)
    weights = torch.ones(n_candidates, dtype=torch.float32)
    return first_slot, candidate_ids, weights


def build_candidate_triples(positive, slot, candidate_ids):
    """Expand a positive + slot + candidates into full candidate triples.

    Returns a (N_candidates, 3) LongTensor where each row is (h, r, t)
    for one candidate. Useful when scoring with the discriminator's
    score(h_ids, r_ids, t_ids) API.
    """
    h_id, r_id, t_id = positive
    n = candidate_ids.shape[0]
    if slot == "head":
        # Each row: (cand, r, t_id)
        h_col = candidate_ids
        r_col = torch.full((n,), r_id, dtype=torch.long)
        t_col = torch.full((n,), t_id, dtype=torch.long)
    elif slot == "tail":
        h_col = torch.full((n,), h_id, dtype=torch.long)
        r_col = torch.full((n,), r_id, dtype=torch.long)
        t_col = candidate_ids
    else:
        raise ValueError(f"Unknown slot: {slot!r}")
    return torch.stack([h_col, r_col, t_col], dim=1)
