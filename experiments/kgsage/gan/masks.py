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
    def __init__(self, kg: dict, n_ent: int, n_rel: int, support_max: int | None = None):
        """`kg` is kgsage.data.loaders.load_kg output: needs 'triples_train'
        (pools) and 'triple_set_all' (falseness oracle over all splits).

        support_max: None = off. Integer >= 0 adds the INVERTED-SUPPORT MASK to
        logits_mask: candidates corroborated by the anchor entity's
        neighbourhood (direct neighbours, or more than support_max shared
        neighbours) are banned, so every sampled corruption is neighbourhood-
        contradicting BY CONSTRUCTION -- during training, not just at decode.
        The anchor is the entity that keeps its slot. Rows whose pool holds no
        unsupported candidate lift the support ban (counted in
        `stat_support_lifted`; a structural property of the graph, not a bug).
        The adjacency matches the decode path: all splits of the KG.
        """
        self.n_ent, self.n_rel = n_ent, n_rel
        self.support_max = support_max

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

        # Support machinery (built lazily on first use; also serves the
        # lambda_support penalty even when the hard mask is off).
        self._adj = None                       # scipy csr, undirected, all splits
        self._all_triples = list(kg["triple_set_all"])
        self._sup_cache: dict[tuple[int, int], torch.Tensor] = {}
        self.stat_support_lifted = 0           # rows that lifted the support ban

    # ---------------- inverted-support machinery ----------------

    def _adjacency(self):
        if self._adj is None:
            import numpy as np
            import scipy.sparse as sp
            rows, cols = [], []
            for h, _, t in self._all_triples:
                rows.append(h); cols.append(t)
                rows.append(t); cols.append(h)
            adj = sp.csr_matrix(
                (np.ones(len(rows), dtype=np.int32), (rows, cols)),
                shape=(self.n_ent, self.n_ent))
            adj.data[:] = 1                    # collapse parallel edges to 0/1
            self._adj = adj
        return self._adj

    def support_row(self, anchor: int, support_max: int = 0) -> torch.Tensor:
        """Bool [n_ent]: candidates SUPPORTED by `anchor`'s neighbourhood --
        direct (1-hop) neighbours, or sharing more than `support_max`
        neighbours with the anchor. These are what the inverted mask bans."""
        key = (anchor, support_max)
        hit = self._sup_cache.get(key)
        if hit is not None:
            return hit
        import torch as _t
        adj = self._adjacency()
        nbr = adj.getrow(anchor)               # 1 x n: N(anchor)
        shared = nbr @ adj                     # 1 x n: |N(anchor) & N(x)|
        ban = _t.from_numpy(
            (nbr.toarray()[0] > 0) | (shared.toarray()[0] > support_max))
        if len(self._sup_cache) < 20000:
            self._sup_cache[key] = ban
        return ban

    def support_rows(self, anchor_ids: torch.Tensor,
                     support_max: int = 0) -> torch.Tensor:
        """Bool [B, n_ent] stack of support_row for a batch of anchors
        (used both by logits_mask and by the lambda_support penalty)."""
        return torch.stack([self.support_row(int(a), support_max)
                            for a in anchor_ids])

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

        # inverted-support ban (train-time twin of the decode mask). Applied
        # per row; a row left empty lifts ONLY the support ban -- the
        # correctness bans above are never relaxed.
        if self.support_max is not None:
            for i in range(B):
                anchor = int(h_ids[i]) if slot == TAIL else int(t_ids[i])
                sup = self.support_row(anchor, self.support_max)
                kept = allowed[i] & ~sup
                if kept.any():
                    allowed[i] = kept
                else:
                    self.stat_support_lifted += 1

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
