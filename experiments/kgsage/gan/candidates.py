"""Per-triple candidate sets for the candidate-scoring generator (KGSAGE-2).

WHY
    The v1 generator's global Linear(hidden -> n_ent) output head is the
    physical home of the measured popularity prior (per-entity weight columns
    that survive deleting the anchor context). KGSAGE-2 removes that head:
    logits exist only for a per-triple CANDIDATE SET, scored by interaction
    with the anchor's query vector. This module supplies those sets.

DESIGN
    - Candidates come from the relation/slot TYPE POOL (same pools the masks
      use), sampled as a mixture: `mix_uniform` of the draws uniform over the
      pool, the rest proportional to in-slot frequency (hard, popular
      candidates must be present so the discriminators can push them down -- a
      candidate the generator never sees, it can never learn to reject).
    - Every batch row also force-includes requested ids (true filler for
      discriminator batches; decode-time shortlists).
    - Returns log q(x) of the ACTUAL mixture per candidate, for the logQ
      correction (Yi et al., RecSys'19): logits' = logits - log q, so the
      sampling frequency of popular candidates cancels out of the softmax and
      the popularity prior has nowhere to hide in training statistics either.
    - Sampling is WITH replacement (duplicate candidates in a row are
      harmless: identical logits, identical logQ).
"""

from __future__ import annotations

import numpy as np
import torch

HEAD, TAIL = 0, 2


class CandidateSampler:
    def __init__(self, triples, n_ent: int, n_rel: int, k: int = 256,
                 mix_uniform: float = 0.5, seed: int = 0):
        self.n_ent, self.k, self.mix = n_ent, k, mix_uniform
        self.rng = np.random.default_rng(seed)

        # per (slot, relation): pool ids + in-slot frequency distribution
        pools: dict[tuple[int, int], dict] = {}
        counts: dict[tuple[int, int], dict] = {}
        for h, r, t in triples:
            for slot, e in ((HEAD, h), (TAIL, t)):
                c = counts.setdefault((slot, r), {})
                c[e] = c.get(e, 0) + 1
        for key, c in counts.items():
            ids = np.fromiter(c.keys(), dtype=np.int64)
            freq = np.fromiter(c.values(), dtype=np.float64)
            pools[key] = {"ids": ids, "p_freq": freq / freq.sum()}
        self.pools = pools

    def sample(self, r_ids, slot: int, include=None):
        """Candidates for a batch of relations at one slot.

        r_ids   : LongTensor/array [B]
        include : optional LongTensor/array [B, I] ids always present per row
        Returns (cand [B, K], logq [B, K]) as torch tensors. K = self.k
        (+ I if include given).
        """
        B = len(r_ids)
        n_inc = 0 if include is None else include.shape[1]
        cand = np.zeros((B, self.k + n_inc), dtype=np.int64)
        logq = np.zeros((B, self.k + n_inc), dtype=np.float32)
        n_uni = int(round(self.k * self.mix))

        for i in range(B):
            key = (slot, int(r_ids[i]))
            pool = self.pools.get(key)
            if pool is None or len(pool["ids"]) == 0:      # unseen relation/slot
                ids = self.rng.integers(0, self.n_ent, size=self.k)
                q = np.full(self.k, 1.0 / self.n_ent)
            else:
                pids, pf = pool["ids"], pool["p_freq"]
                uni = pids[self.rng.integers(0, len(pids), size=n_uni)]
                frq = pids[self.rng.choice(len(pids), size=self.k - n_uni, p=pf)]
                ids = np.concatenate([uni, frq])
                # q(x) of the actual mixture over the pool
                pos = {e: j for j, e in enumerate(pids)}
                fx = np.array([pf[pos[e]] for e in ids])
                q = self.mix / len(pids) + (1.0 - self.mix) * fx
            cand[i, :self.k] = ids
            logq[i, :self.k] = np.log(q + 1e-12)
            if include is not None:
                inc = np.asarray(include[i], dtype=np.int64)
                cand[i, self.k:] = inc
                key_pool = self.pools.get(key)
                if key_pool is not None:
                    pos = {e: j for j, e in enumerate(key_pool["ids"])}
                    fx = np.array([key_pool["p_freq"][pos[e]] if e in pos else 0.0
                                   for e in inc])
                    qi = self.mix / max(len(key_pool["ids"]), 1) + (1.0 - self.mix) * fx
                else:
                    qi = np.full(n_inc, 1.0 / self.n_ent)
                logq[i, self.k:] = np.log(qi + 1e-12)
        return torch.from_numpy(cand), torch.from_numpy(logq)
