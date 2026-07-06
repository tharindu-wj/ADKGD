"""Training-time candidate masks for the adversarial loop.

The generator must sample from the same constrained space the deployed
negatives come from: the relation's TRAIN-split type pool, minus every
known-true filler of the query across ALL splits (1-N safe), minus the
self-loop entity. These masks are applied to the slot logits as -inf BEFORE
Gumbel noise, so a violating candidate is unsampleable by construction --
the old "would-collide" pathology becomes a structural impossibility rather
than a monitored rate.

Everything here lives in the GAN's own id space (kgsage.data.loaders vocab).
Pool masks are dense bool tensors built once ([2, n_rel, n_ent]; ~7 MB on
FB15K-237, ~0.9 MB on WN18RR); the per-query all-true masks are sparse dicts
gathered per batch.

Empty-pool guard: if a (slot, relation) pool empties after masking (tiny 1-N
relations), the batch helper falls back to the ALL-ENTITIES universe minus
known-true minus self for those rows and reports them, so no row is ever
fully -inf (which would make Gumbel sampling NaN).
"""

from __future__ import annotations

import torch

HEAD, TAIL = 0, 2
NEG_INF = float("-inf")


class CandidateMasks:
    def __init__(self, kg: dict, n_ent: int, n_rel: int):
        """`kg` is kgsage.data.loaders.load_kg output: needs 'triples_train'
        (pools) and 'triple_set_all' (falseness oracle over all splits)."""
        self.n_ent, self.n_rel = n_ent, n_rel

        pool = torch.zeros(2, n_rel, n_ent, dtype=torch.bool)  # [head/tail, r, e]
        for h, r, t in kg["triples_train"]:
            pool[0, r, h] = True
            pool[1, r, t] = True
        self.pool = pool  # index 0 = head pool, 1 = tail pool

        self.true_tails: dict[tuple[int, int], list[int]] = {}
        self.true_heads: dict[tuple[int, int], list[int]] = {}
        for h, r, t in kg["triple_set_all"]:
            self.true_tails.setdefault((h, r), []).append(t)
            self.true_heads.setdefault((r, t), []).append(h)

    def logits_mask(self, h_ids: torch.Tensor, r_ids: torch.Tensor,
                    t_ids: torch.Tensor, slot: int) -> tuple[torch.Tensor, int]:
        """Additive mask [B, n_ent] (0 allowed / -inf banned) for one slot.

        Returns (mask, n_pool_fallback_rows). Guarantees >=1 allowed entity
        per row.
        """
        B = h_ids.shape[0]
        pool_idx = 0 if slot == HEAD else 1
        allowed = self.pool[pool_idx, r_ids].clone()          # [B, n_ent]

        # ban every known-true filler + the self-loop entity
        for i in range(B):
            h, r, t = int(h_ids[i]), int(r_ids[i]), int(t_ids[i])
            if slot == TAIL:
                banned = self.true_tails.get((h, r))
                self_row = h
            else:
                banned = self.true_heads.get((r, t))
                self_row = t
            if banned:
                allowed[i, banned] = False
            allowed[i, self_row] = False

        # empty-pool fallback: universe minus banned minus self
        empty = ~allowed.any(dim=1)
        n_fallback = int(empty.sum())
        if n_fallback:
            for i in torch.nonzero(empty, as_tuple=False).flatten().tolist():
                h, r, t = int(h_ids[i]), int(r_ids[i]), int(t_ids[i])
                row = torch.ones(self.n_ent, dtype=torch.bool)
                if slot == TAIL:
                    banned = self.true_tails.get((h, r))
                    row[h] = False
                else:
                    banned = self.true_heads.get((r, t))
                    row[t] = False
                if banned:
                    row[banned] = False
                allowed[i] = row

        mask = torch.zeros(B, self.n_ent)
        mask[~allowed] = NEG_INF
        return mask, n_fallback
