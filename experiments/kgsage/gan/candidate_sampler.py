"""Per-triple candidate sets (paper Phase 2: Adversarial Generator Training).

The generator never scores the whole entity vocabulary during training.
Instead, each training triple gets a small candidate set of size K (the trainer
passes NUM_CANDIDATES), and the generator only ranks those. This module builds
the candidate sets.

Design choices, in plain terms:
  - Candidates come from the relation's type pool — the entities actually
    observed in that slot for that relation in the training split. This keeps
    candidates type-plausible (a "place of birth" candidate is a place).
  - Sampling is a mixture: half the draws uniform over the type pool, half
    proportional to how frequent each entity is in that slot. The frequent
    (popular) entities must be present, otherwise the discriminators could
    never teach the generator to rank them down.
  - Each candidate also gets log q(x) — the log-probability that the mixture
    sampled it. The generator subtracts this from its scores (the "logQ
    correction", Yi et al., RecSys 2019), which cancels the sampling
    frequency out of the softmax. Without it, popular entities would win
    simply by being drawn more often.
  - Sampling is WITH replacement; duplicates are harmless (identical score,
    identical correction).

Nothing here judges whether the anchor's neighbourhood corroborates a
candidate — that is the neighbourhood discriminator's job. This module only
decides which candidates get to be scored at all.
"""

from __future__ import annotations

import numpy as np
import torch

HEAD, TAIL = 0, 2


class CandidateSampler:
    """Samples a type-pool candidate set per triple, with the logQ correction.

    NOTE: `cand_k` (the checkpoint payload key for K) is a frozen name; here
    the same quantity is the constructor's `k`.
    """

    def __init__(self, triples, n_ent: int, n_rel: int, k: int = 256,
                 mix_uniform: float = 0.5, seed: int = 0):
        self.n_ent = n_ent
        self.k = k                       # candidate-set size (NUM_CANDIDATES)
        self.mix = mix_uniform           # fraction of draws that are uniform
        self.rng = np.random.default_rng(seed)

        # For every (slot, relation) pair: the type pool of observed filler ids
        # and their in-slot frequency distribution.
        pools: dict[tuple[int, int], dict] = {}
        counts: dict[tuple[int, int], dict] = {}
        for head, relation, tail in triples:
            for slot, entity in ((HEAD, head), (TAIL, tail)):
                slot_counts = counts.setdefault((slot, relation), {})
                slot_counts[entity] = slot_counts.get(entity, 0) + 1
        for key, slot_counts in counts.items():
            pool_ids = np.fromiter(slot_counts.keys(), dtype=np.int64)
            frequency = np.fromiter(slot_counts.values(), dtype=np.float64)
            pools[key] = {"ids": pool_ids, "p_freq": frequency / frequency.sum()}
        self.pools = pools

    def sample(self, relation_ids, slot: int, include=None):
        """Draw a candidate set for each relation in a batch, at one slot.

        relation_ids : LongTensor/array [batch]
        include      : optional [batch, I] ids forced into every row (e.g. the
                       true_filler, so the discriminator always sees it).
        Returns (candidate_ids [batch, K(+I)], log_q [batch, K(+I)]) as torch
        tensors.
        """
        batch_size = len(relation_ids)
        num_included = 0 if include is None else include.shape[1]
        candidate_ids = np.zeros((batch_size, self.k + num_included),
                                 dtype=np.int64)
        log_q = np.zeros((batch_size, self.k + num_included), dtype=np.float32)
        num_uniform = int(round(self.k * self.mix))

        for i in range(batch_size):
            key = (slot, int(relation_ids[i]))
            pool = self.pools.get(key)
            if pool is None or len(pool["ids"]) == 0:
                # Relation/slot has no type pool (never seen in training):
                # fall back to uniform over the whole vocabulary.
                ids = self.rng.integers(0, self.n_ent, size=self.k)
                sample_prob = np.full(self.k, 1.0 / self.n_ent)
            else:
                pool_ids, pool_freq = pool["ids"], pool["p_freq"]
                uniform_draws = pool_ids[
                    self.rng.integers(0, len(pool_ids), size=num_uniform)]
                frequency_draws = pool_ids[
                    self.rng.choice(len(pool_ids), size=self.k - num_uniform,
                                    p=pool_freq)]
                ids = np.concatenate([uniform_draws, frequency_draws])
                # q(x) under the actual mixture over this type pool.
                position = {e: j for j, e in enumerate(pool_ids)}
                freq_of_draw = np.array([pool_freq[position[e]] for e in ids])
                sample_prob = (self.mix / len(pool_ids)
                               + (1.0 - self.mix) * freq_of_draw)
            candidate_ids[i, :self.k] = ids
            log_q[i, :self.k] = np.log(sample_prob + 1e-12)

            if include is not None:
                forced = np.asarray(include[i], dtype=np.int64)
                candidate_ids[i, self.k:] = forced
                pool = self.pools.get(key)
                if pool is not None:
                    position = {e: j for j, e in enumerate(pool["ids"])}
                    freq_of_forced = np.array(
                        [pool["p_freq"][position[e]] if e in position else 0.0
                         for e in forced])
                    forced_prob = (self.mix / max(len(pool["ids"]), 1)
                                   + (1.0 - self.mix) * freq_of_forced)
                else:
                    forced_prob = np.full(num_included, 1.0 / self.n_ent)
                log_q[i, self.k:] = np.log(forced_prob + 1e-12)
        return torch.from_numpy(candidate_ids), torch.from_numpy(log_q)
