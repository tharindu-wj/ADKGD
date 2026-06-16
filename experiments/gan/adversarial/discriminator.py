"""Internal TransE Discriminator for the KGSAGE adversarial loop.

The Discriminator D plays two roles simultaneously:
  1. SCORER: produces a real-valued score for any triple (h, r, t).
             Higher score = "more true-looking" per its learned embeddings.
  2. REWARD SOURCE: the Generator's REINFORCE signal is the NEGATIVE of D's
             score on G's sampled negatives. The HARDER a negative looks
             (i.e., the higher D scores it), the BIGGER reward G gets.

TransE is chosen because it's simple, well-understood, and fast - we're
training the Generator, not setting a KG-completion record. The whole
Discriminator gets discarded after training (only the Generator ships
in the corruption-phase checkpoint).

Mechanics:
  E       entity embedding table  [n_entities x dim]
  R       relation embedding table [n_relations x dim]
  score(h, r, t) = -||E(h) + R(r) - E(t)||_p
                   (negative L_p distance; higher = more plausible)

Loss (margin-style, separately invoked by the training loop):
  L_D = mean( clamp(0, margin - score(pos) + score(neg)) )

Phase 2.1 builds and unit-tests D. Phase 2.3 trains it via the REINFORCE
co-training loop.
"""
import torch
import torch.nn as nn


DEFAULT_DIM = 100
DEFAULT_P_NORM = 2
DEFAULT_MARGIN = 0.5


class TransEDiscriminator(nn.Module):
    """TransE-style scorer used as the GAN's discriminator.

    Embedding tables E and R are PUBLIC so that the Generator can share
    them in Phase 2.2 (parameter efficiency; both networks learn from
    the same gradient flow).
    """

    def __init__(self, n_entities, n_relations, dim=DEFAULT_DIM,
                 p_norm=DEFAULT_P_NORM):
        super().__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.dim = dim
        self.p_norm = p_norm

        # Shared embedding tables. Initialised with Xavier uniform, then
        # entity vectors are L2-normalised - this is TransE convention to
        # prevent embeddings from collapsing to zero norm during training.
        self.E = nn.Embedding(n_entities, dim)
        self.R = nn.Embedding(n_relations, dim)
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.xavier_uniform_(self.R.weight)
        with torch.no_grad():
            self.E.weight.div_(
                self.E.weight.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            )

    def score(self, h_ids, r_ids, t_ids):
        """Return a scalar score per triple. Shape: same as h_ids.

        Higher = more "true-looking". Implementation note: we return the
        NEGATIVE distance, not the distance itself, so the rest of the
        codebase can consistently use "higher score = better".
        """
        h = self.E(h_ids)
        r = self.R(r_ids)
        t = self.E(t_ids)
        return -torch.norm(h + r - t, p=self.p_norm, dim=-1)

    def margin_loss(self, pos_h, pos_r, pos_t,
                    neg_h, neg_r, neg_t, margin=DEFAULT_MARGIN):
        """Standard pairwise margin ranking loss.

        Pushes pos_score UP and neg_score DOWN until the gap exceeds
        `margin`. Triples beyond the margin contribute zero loss.
        """
        pos_score = self.score(pos_h, pos_r, pos_t)
        neg_score = self.score(neg_h, neg_r, neg_t)
        return torch.clamp(margin - pos_score + neg_score, min=0).mean()

    def renormalize_entities(self):
        """Project entity embeddings back to unit norm.

        Call after each optimizer step during training. TransE relies on
        bounded entity norms; without renormalisation embeddings can
        drift, making the margin loss meaningless.
        """
        with torch.no_grad():
            self.E.weight.div_(
                self.E.weight.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            )
