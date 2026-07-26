"""The plausibility discriminator (paper Phase 2: Adversarial Generator
Training).

The plausibility discriminator answers one question about a triple: "could this
be a real fact?" It scores (anchor, relation, candidate) and is trained to
output HIGH for real triples and LOW for generated or mismatched ones. The
generator is trained to push this score UP, which forces its corruptions to
stay plausible instead of drifting into obvious nonsense.

This is one half of the dual-discriminator architecture; the other half is the
neighbourhood discriminator (neighbourhood_discriminator.py), which judges
whether the filler fits the anchor's neighbourhood.

Inputs are rows of the frozen context table E' (see
neighbourhood_context_encoder.py) — the plausibility discriminator has no
per-entity parameters of its own, only a small relation table and an MLP. Every
linear layer is wrapped in spectral normalisation, which limits how sharp this
discriminator's decision surface can get and keeps the adversarial game stable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class PlausibilityDiscriminator(nn.Module):
    """Scores how plausible an (anchor, relation, candidate) triple looks.

    NOTE: `rel_embedding` and `net` are FROZEN names — they ARE the state-dict
    keys stored inside every saved checkpoint, so renaming them makes those
    checkpoints unloadable. The locked generator_*.pt artifacts carry these
    weights under the payload key "dreal_state" (frozen key; "dreal" = the
    plausibility discriminator, this module). The same shorthand appears in the
    training log as the token `D-acc=` — this discriminator's accuracy on real
    versus generated triples.
    """

    def __init__(self, dim: int = 64, n_rel: int = None, hidden: int = 128):
        super().__init__()
        self.rel_embedding = nn.Embedding(n_rel, dim)
        nn.init.normal_(self.rel_embedding.weight, std=0.1)
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(3 * dim, hidden)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden, hidden)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(hidden, 1)),
        )

    def forward(self, anchor_embedding: torch.Tensor, relation_ids: torch.Tensor,
                candidate_embedding: torch.Tensor) -> torch.Tensor:
        """Return one plausibility logit per triple, shape [batch].

        anchor_embedding    : [batch, dim] E' row of the entity that KEEPS its
                              slot (the anchor).
        relation_ids        : [batch] integer relation ids.
        candidate_embedding : [batch, dim] E' row of the filler being judged —
                              either the true_filler, or the generator's
                              straight-through soft embedding of its
                              picked_candidate.
        """
        features = torch.cat([anchor_embedding,
                              self.rel_embedding(relation_ids),
                              candidate_embedding], dim=1)
        return self.net(features).squeeze(-1)
