"""Inference internals for Phase 3 corruption.

This module is the engine room. It loads a KGSAGE checkpoint, reconstructs
the trained G + D, and provides the one-positive-in -> one-negative-out
function that the KGCorrupter class wraps.

The 7-step inference per positive:

  STEP 1: Resolve string entity/relation names to integer IDs.
          (G and D operate on int IDs; consumers pass strings.)

  STEP 2: Choose the slot to corrupt (head or tail) via Bernoulli
          sampling on cardinality stats - same logic as training.

  STEP 3: Build the type-coherent candidate pool (N_S=64 candidates)
          using build_candidates(). Falls back to the other slot if
          the requested pool is degenerate.

  STEP 4: Score all candidates with the frozen G, apply cardinality
          weights, take softmax -> P_G.

  STEP 5: Sample one candidate from P_G using the Python random
          module (NOT torch.multinomial) - this keeps determinism
          purely seed-driven and avoids torch's global RNG state.

  STEP 6: Validate the candidate triple:
            - Not a self-loop (h != t).
            - Not present in the real KG (concept_pools.real_triple_set).
          Failed candidates are RETRIED up to `max_retries` times by
          drawing a different sample from the SAME candidate pool.
          (Re-sampling the pool would be wasteful - the candidates are
          already type-coherent; we just need a different draw.)

  STEP 7: If all retries fail, FALL BACK to uniform-random sampling
          across the full entity vocab for the chosen slot. This case
          is logged and counted so the bridge can report frequency.

Output: a string triple (h, r, t), translated back from internal IDs.
"""
import logging
import random

import torch

from experiments.gan.adversarial.candidate_pool import (
    build_candidates, build_candidate_triples,
)
from experiments.gan.adversarial.discriminator import TransEDiscriminator
from experiments.gan.adversarial.generator import CandidateScorer


_log = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path, device=None):
    """Reconstruct (G, D) from a KGSAGE checkpoint produced by adversarial/train.py.

    Args:
      checkpoint_path: path to <DATASET>_kgsage.pt.
      device:          torch device; defaults to CUDA if available else CPU.

    Returns:
      (G, D, payload) - the loaded models (in eval mode) and the raw
      checkpoint payload dict so callers can inspect metadata like
      dataset_hash, embedding_dim, etc.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

    n_ent = payload["n_entities"]
    n_rel = payload["n_relations"]
    dim = payload["embedding_dim"]
    hidden = payload["hidden_dim"]

    D = TransEDiscriminator(n_ent, n_rel, dim=dim).to(device)
    D.load_state_dict(payload["discriminator_state"])
    D.eval()

    G = CandidateScorer(embedding_dim=dim, hidden_dim=hidden).to(device)
    G.load_state_dict(payload["generator_state"])
    G.eval()

    return G, D, payload


def corrupt_one(positive_ids, G, D, concept_pools, real_triple_set,
                rng, n_candidates=64, max_retries=10, slot=None):
    """Produce one negative triple in integer-ID form.

    Args:
      positive_ids:     (h_id, r_id, t_id) tuple of ints.
      G, D:             trained Generator and Discriminator (eval mode).
      concept_pools:    state dict loaded from concept_pools.pkl.
      real_triple_set:  Set[Tuple[int,int,int]] - the training KG, for
                        collision rejection (usually concept_pools["real_triple_set"]).
      rng:              Python Random instance (for determinism).
      n_candidates:     N_S size of the candidate pool.
      max_retries:      times to re-sample from the SAME pool on
                        collision/self-loop before falling back.
      slot:             "head", "tail", or None (Bernoulli sample).

    Returns:
      (negative_triple_ids, info_dict)
        negative_triple_ids: (h, r, t) tuple of ints.
        info_dict:           {"slot": str, "retries": int,
                              "uniform_fallback": bool}
    """
    h_id, r_id, t_id = positive_ids

    # STEPS 2-3: choose slot + build pool ONCE; we'll re-sample inside.
    chosen_slot, candidate_ids, weights = build_candidates(
        positive_ids, concept_pools, slot=slot,
        n_candidates=n_candidates, rng=rng,
    )
    candidate_triples = build_candidate_triples(
        positive_ids, chosen_slot, candidate_ids
    )  # [N_S, 3]

    # STEP 4: G scores -> P_G.
    with torch.no_grad():
        P_G = G.distribution_from_ids(
            candidate_triples, D.E, D.R, weights=weights,
        )  # [N_S]
    # Use Python-side sampling for seed-driven determinism.
    P_G_list = P_G.tolist()

    # STEPS 5-6: sample + validate, retrying on failure.
    indices_seen = set()
    for attempt in range(max_retries):
        idx = rng.choices(range(len(P_G_list)), weights=P_G_list, k=1)[0]
        # If we've already tried this candidate, try again (cheap dedupe).
        if idx in indices_seen and attempt < max_retries - 1:
            continue
        indices_seen.add(idx)

        cand = candidate_triples[idx].tolist()
        cand = (int(cand[0]), int(cand[1]), int(cand[2]))
        if cand[0] == cand[2]:                  # self-loop
            continue
        if cand in real_triple_set:             # real-KG collision
            continue
        return cand, {
            "slot": chosen_slot,
            "retries": attempt,
            "uniform_fallback": False,
        }

    # STEP 7: uniform-random fallback.
    fake = _uniform_fallback(positive_ids, chosen_slot, concept_pools,
                              real_triple_set, rng)
    _log.warning(
        "uniform_fallback fired for positive=%s slot=%s after %d retries",
        positive_ids, chosen_slot, max_retries,
    )
    return fake, {
        "slot": chosen_slot,
        "retries": max_retries,
        "uniform_fallback": True,
    }


def _uniform_fallback(positive_ids, slot, concept_pools, real_triple_set, rng):
    """Last-resort uniform-random sample across the full entity vocab."""
    h_id, r_id, t_id = positive_ids
    n_entities = len(concept_pools["entity_to_id"])
    clean_id = h_id if slot == "head" else t_id
    # Try up to a generous number of vocab draws to avoid pathological loops.
    for _ in range(1000):
        new_id = rng.randint(0, n_entities - 1)
        if new_id == clean_id:
            continue
        if slot == "head":
            candidate = (new_id, r_id, t_id)
        else:
            candidate = (h_id, r_id, new_id)
        if candidate[0] == candidate[2]:
            continue
        if candidate in real_triple_set:
            continue
        return candidate
    # Truly degenerate vocab - return SOMETHING rather than raising.
    new_id = rng.randint(0, n_entities - 1)
    return (new_id, r_id, t_id) if slot == "head" else (h_id, r_id, new_id)
