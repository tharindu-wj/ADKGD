"""KGCorrupter - the public corruption primitive.

This is the ONE class consumers (ADKGD bridge, future detectors,
ablation scripts) instantiate. It wraps the inference internals in
infer.py and presents a simple string-triple-in / string-triple-out
interface.

Usage:
    from experiments.gan.corruption.api import KGCorrupter

    corrupter = KGCorrupter(
        checkpoint_path="experiments/gan/outputs/checkpoints/FB15K-237_cgsp.pt",
        concept_pools_path="experiments/gan/outputs/concept_pools/FB15K-237.pkl",
    )

    # One triple at a time:
    negative = corrupter.corrupt(("/m/0d3k14", "/people/person/nationality", "/m/09c7w0"))

    # Or a batch:
    negatives = corrupter.corrupt_batch(list_of_positive_triples, seed=42)

Determinism contract (from PIPELINE.md):
  - Same (triple, seed) -> same output triple.
  - Output never equals the input positive.
  - Output is never a self-loop.
  - Output is never in the real KG.
  - Fallback to uniform random NEVER happens silently - it is logged
    and counted (see .stats() for cumulative counters).
"""
import logging
import random

import torch

from experiments.gan.concept.concept_pools import load_pools
from experiments.gan.corruption.infer import corrupt_one, load_checkpoint


_log = logging.getLogger(__name__)


# Defaults match PIPELINE.md.
DEFAULT_N_CANDIDATES = 64
DEFAULT_MAX_RETRIES = 10


class KGCorrupter:
    """The CGSP corruption primitive.

    Init is heavy (loads checkpoint + concept pools); corrupt() is
    cheap (typically <1 ms per call once init is done).
    """

    def __init__(self, checkpoint_path, concept_pools_path,
                 n_candidates=DEFAULT_N_CANDIDATES,
                 max_retries=DEFAULT_MAX_RETRIES,
                 device=None):
        """Load the trained pair + concept pools.

        Args:
          checkpoint_path:    path to <DATASET>_cgsp.pt (Phase 2 output).
          concept_pools_path: path to <DATASET>.pkl     (Phase 1 output).
          n_candidates:       candidate pool size at inference.
          max_retries:        attempts to redraw on self-loop / real-KG
                              collision before falling back to uniform.
          device:             torch device; defaults to CUDA if available.
        """
        # Load Phase 1 concept pools first - we need vocab to validate
        # the checkpoint matches.
        self.concept_pools = load_pools(concept_pools_path)
        self.real_triple_set = self.concept_pools["real_triple_set"]
        self.entity_to_id = self.concept_pools["entity_to_id"]
        self.relation_to_id = self.concept_pools["relation_to_id"]
        self.id_to_entity = self.concept_pools["id_to_entity"]
        self.id_to_relation = self.concept_pools["id_to_relation"]

        # Load Phase 2 checkpoint.
        self.G, self.D, self.payload = load_checkpoint(checkpoint_path, device=device)

        # Cache invalidation: checkpoint's dataset_hash must match pools'.
        ckpt_hash = self.payload.get("dataset_hash")
        pools_hash = self.concept_pools.get("dataset_hash")
        if ckpt_hash and pools_hash and ckpt_hash != pools_hash:
            _log.warning(
                "dataset_hash mismatch: checkpoint=%s vs concept_pools=%s. "
                "The trained model was likely produced from a different KG.",
                ckpt_hash[:16], pools_hash[:16],
            )

        self.n_candidates = n_candidates
        self.max_retries = max_retries

        # Cumulative counters (reset via reset_stats()).
        self._stats = _make_empty_stats()

        # Default RNG used when the caller doesn't pass seed=.
        # New default rng on each corrupt() call would be NON-deterministic
        # across calls; persistent rng gives reproducibility within a session.
        self._default_rng = random.Random(0)

    # ─── public API ──────────────────────────────────────────────

    def corrupt(self, positive, slot=None, seed=None):
        """Generate one negative triple for a positive.

        Args:
          positive:   (h_str, r_str, t_str) tuple of strings.
          slot:       "head", "tail", or None (Bernoulli on cardinality).
          seed:       int for full determinism on this call. If None,
                      uses the internal session rng (still deterministic
                      across calls IF init seed and call order are fixed).

        Returns:
          (h_str, r_str, t_str) - the negative triple.
        """
        # STEP 1: resolve strings to ints.
        h_str, r_str, t_str = positive
        try:
            positive_ids = (
                self.entity_to_id[h_str],
                self.relation_to_id[r_str],
                self.entity_to_id[t_str],
            )
        except KeyError as exc:
            raise KeyError(
                f"Unknown entity or relation in positive {positive!r}: {exc}"
            ) from exc

        rng = random.Random(seed) if seed is not None else self._default_rng

        # STEPS 2-7 inside infer.corrupt_one.
        neg_ids, info = corrupt_one(
            positive_ids, self.G, self.D,
            self.concept_pools, self.real_triple_set,
            rng=rng,
            n_candidates=self.n_candidates,
            max_retries=self.max_retries,
            slot=slot,
        )

        # Update counters.
        self._stats["processed"] += 1
        self._stats["retries"] += info["retries"]
        if info["uniform_fallback"]:
            self._stats["uniform_fallbacks"] += 1
        self._stats[f"slot_{info['slot']}"] += 1

        # STEP 8: ids back to strings.
        return (
            self.id_to_entity[neg_ids[0]],
            self.id_to_relation[neg_ids[1]],
            self.id_to_entity[neg_ids[2]],
        )

    def corrupt_batch(self, positives, seed=None):
        """Generate negatives for a list of positives.

        Semantics: identical to calling corrupt() in a loop. We expose
        this as a separate method so a future GPU-batched implementation
        can drop in without callers changing.

        Args:
          positives: iterable of (h_str, r_str, t_str) tuples.
          seed:      int for determinism. If provided, child seeds are
                     derived deterministically from it - same `seed` +
                     same `positives` order -> same output sequence.

        Returns:
          list of (h_str, r_str, t_str) tuples, row-aligned with input.
        """
        positives = list(positives)
        out = []
        if seed is None:
            for pos in positives:
                out.append(self.corrupt(pos))
        else:
            # Derive a per-item seed deterministically from the batch seed.
            seed_rng = random.Random(seed)
            for pos in positives:
                item_seed = seed_rng.randint(0, 2**31 - 1)
                out.append(self.corrupt(pos, seed=item_seed))
        return out

    def stats(self):
        """Return a copy of the cumulative counters dict."""
        return dict(self._stats)

    def reset_stats(self):
        """Zero the counters (used between experiments)."""
        self._stats = _make_empty_stats()


def _make_empty_stats():
    return {
        "processed": 0,
        "retries": 0,
        "uniform_fallbacks": 0,
        "slot_head": 0,
        "slot_tail": 0,
    }
