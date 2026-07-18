"""D_real: the plausibility/realism critic of KGSAGE-2 (LP-free).

Replaces the v1 frozen-ComplEx-plus-residual reward. Fully trainable, so the
adversarial game is live again; spectral normalisation bounds its Lipschitz
constant (the stability role the frozen LP used to play). Scores a triple from
the FROZEN context embeddings of its slots plus its own relation table --
there are no per-entity free parameters, so a global popularity prior has no
private storage here either (it can still be expressed through E' geometry;
that is what D_match exists to counterbalance).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class DReal(nn.Module):
    def __init__(self, dim: int = 64, n_rel: int = None, hidden: int = 128):
        super().__init__()
        self.rel_embedding = nn.Embedding(n_rel, dim)
        nn.init.normal_(self.rel_embedding.weight, std=0.1)
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(3 * dim, hidden)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden, hidden)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden, 1)),
        )

    def forward(self, h_emb: torch.Tensor, r_ids: torch.Tensor,
                x_emb: torch.Tensor) -> torch.Tensor:
        """[B] realism logits for triples given slot EMBEDDINGS (hard rows of
        E', or the generator's straight-through soft embedding)."""
        z = torch.cat([h_emb, self.rel_embedding(r_ids), x_emb], dim=1)
        return self.net(z).squeeze(-1)
