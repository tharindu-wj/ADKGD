"""Frozen ComplEx as an in-training-loop scorer for the KGSAGE GAN.

Wraps the raw LibKGE tensors (loaded by kgsage.lp_scorer) as an nn.Module with
buffers so the adversarial loop can score:
  - HARD ids       (D-step fakes, fences, filters, diagnostics), and
  - SOFT one-hots  (the generator's straight-through Gumbel samples, so the
                    G-step gradient flows through candidate CHOICE).

Two coordinate systems exist: the scorer's LibKGE rows and the GAN's own
vocab ids (kgsage.data.loaders, first-seen order). `realigned()` materialises
copies of the embedding tables permuted into GAN-id space so every GAN-side
call uses GAN ids natively -- one gather at build time instead of per batch.

Correctness gate (A1): scoring an EXACT one-hot through the soft path must
equal the hard-id score to <=1e-4 -- catches any re|im layout, reciprocal
or permutation mistake in one number.

This module is frozen by construction: all tensors are registered as buffers
(never parameters) and every public method runs under the caller's autograd
only w.r.t. the SOFT inputs, never the tables.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrozenComplEx(nn.Module):
    def __init__(self, ent_emb: torch.Tensor, rel_emb: torch.Tensor,
                 reciprocal: bool):
        super().__init__()
        self.register_buffer("ent_emb", ent_emb.clone().detach())
        self.register_buffer("rel_emb", rel_emb.clone().detach())
        self.reciprocal = reciprocal
        self.n_ent = ent_emb.shape[0]
        self.n_rel_base = rel_emb.shape[0] // 2 if reciprocal else rel_emb.shape[0]
        self.dim = ent_emb.shape[1] // 2
        for buf in self.buffers():
            buf.requires_grad_(False)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_lp_scorer(cls, scorer) -> "FrozenComplEx":
        """From a loaded kgsage.lp_scorer.ComplExScorer (LibKGE row space)."""
        return cls(scorer.ent_emb, scorer.rel_emb, scorer.reciprocal)

    def realigned(self, ent_perm: torch.Tensor, rel_perm: torch.Tensor) -> "FrozenComplEx":
        """Copy with rows permuted into a consumer id space.

        ent_perm[i] = this scorer's row for consumer entity id i;
        rel_perm[i] = this scorer's BASE relation row for consumer relation i.
        Reciprocal models keep their inverse block aligned with the same perm.
        """
        ent = self.ent_emb[ent_perm]
        if self.reciprocal:
            base = self.rel_emb[rel_perm]
            inv = self.rel_emb[rel_perm + self.n_rel_base]
            rel = torch.cat([base, inv], dim=0)
        else:
            rel = self.rel_emb[rel_perm]
        return FrozenComplEx(ent, rel, self.reciprocal)

    # -- complex helpers -----------------------------------------------------

    def _split(self, emb: torch.Tensor):
        return emb[..., :self.dim], emb[..., self.dim:]

    def _tail_query(self, h_rows: torch.Tensor, r_rows: torch.Tensor) -> torch.Tensor:
        s_re, s_im = self._split(self.ent_emb[h_rows])
        p_re, p_im = self._split(self.rel_emb[r_rows])
        return torch.cat([s_re * p_re - s_im * p_im,
                          s_re * p_im + s_im * p_re], dim=-1)

    def _head_query(self, r_rows: torch.Tensor, t_rows: torch.Tensor) -> torch.Tensor:
        if self.reciprocal:
            return self._tail_query(t_rows, r_rows + self.n_rel_base)
        p_re, p_im = self._split(self.rel_emb[r_rows])
        o_re, o_im = self._split(self.ent_emb[t_rows])
        return torch.cat([p_re * o_re + p_im * o_im,
                          p_re * o_im - p_im * o_re], dim=-1)

    # -- hard-id scoring ------------------------------------------------------

    def score_hard(self, h_rows, r_rows, t_rows) -> torch.Tensor:
        """[B] forward-direction scores for hard (h, r, t) ids."""
        return (self._tail_query(h_rows, r_rows) * self.ent_emb[t_rows]).sum(-1)

    def score_tails_all(self, h_rows, r_rows) -> torch.Tensor:
        """[B, n_ent] tail-direction scores (fences, filters, diagnostics)."""
        return self._tail_query(h_rows, r_rows) @ self.ent_emb.T

    def score_heads_all(self, r_rows, t_rows) -> torch.Tensor:
        """[B, n_ent] head-direction scores (reciprocal-aware)."""
        return self._head_query(r_rows, t_rows) @ self.ent_emb.T

    # -- soft (differentiable-in-choice) scoring ------------------------------

    def score_soft_tail(self, h_rows, r_rows, soft_tail: torch.Tensor) -> torch.Tensor:
        """[B] scores where the TAIL is a (near-)one-hot over entities.

        Gradient flows into `soft_tail` only; the tables are buffers.
        soft_tail @ ent_emb is the differentiable embedding lookup.
        """
        return (self._tail_query(h_rows, r_rows) * (soft_tail @ self.ent_emb)).sum(-1)

    def score_soft_head(self, soft_head: torch.Tensor, r_rows, t_rows) -> torch.Tensor:
        """[B] scores where the HEAD is a (near-)one-hot over entities.

        Uses the head-direction query (inverse relation on reciprocal models),
        matching score_heads_all -- the same direction-consistency rule the
        band sampler follows.
        """
        return (self._head_query(r_rows, t_rows) * (soft_head @ self.ent_emb)).sum(-1)
