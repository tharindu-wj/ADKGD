"""Residual contextual discriminator f_theta.

D(x) = s_z(x) + f_theta(x), where s_z is the per-relation z-scored FROZEN
ComplEx score (global plausibility, cannot move) and f_theta is a small
trainable head over the candidate triple's FROZEN RGCN context features:

    f_theta(h, r, t) = beta * tanh( MLP([E'[h] | rho_r | E'[t]]) )

Design constraints carried over from the design review:
  - CANDIDATE-ONLY input (no anchor pair): removes the relation-match pair
    shortcut by construction.
  - beta*tanh bound: the residual can never push the total score out of the
    s_z band, so the worst-case failure degrades gracefully into the band
    sampler instead of running away.
  - rho_r is f_theta's OWN relation table (never shared with the generator),
    and E' is frozen -- the only trainable discriminator capacity is this
    head, so the representation-collusion channel does not exist.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualContextD(nn.Module):
    def __init__(self, n_rel: int, dim: int = 64, hidden: int = 128,
                 beta: float = 1.0):
        super().__init__()
        self.rel_embedding = nn.Embedding(n_rel, dim)
        nn.init.normal_(self.rel_embedding.weight, std=0.1)
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )
        self.beta = beta

    def forward(self, head_emb: torch.Tensor, r_ids: torch.Tensor,
                tail_emb: torch.Tensor) -> torch.Tensor:
        """[B] bounded residuals for candidate triples given FROZEN context
        embeddings for the entity slots (hard rows of E', or soft @ E' for the
        generator's straight-through samples)."""
        x = torch.cat([head_emb, self.rel_embedding(r_ids), tail_emb], dim=1)
        return self.beta * torch.tanh(self.mlp(x)).squeeze(-1)

    def from_ids(self, context: torch.Tensor, h_ids: torch.Tensor,
                 r_ids: torch.Tensor, t_ids: torch.Tensor) -> torch.Tensor:
        """Convenience: residuals for hard-id triples under frozen E'."""
        return self.forward(context[h_ids], r_ids, context[t_ids])
