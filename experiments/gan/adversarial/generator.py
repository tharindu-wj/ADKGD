"""KGSAGE Generator: candidate scorer for adversarial negative sampling.

(Architecture follows the CGSP framework of Tong et al. 2026.)

The Generator G learns to RANK candidate triples by how "good" they are
as negatives - good meaning "hard for the Discriminator to dismiss".
For each positive (h, r, t), G is queried over N_S candidate triples
(produced by candidate_pool.build_candidates) and returns N_S scalar
scores. A softmax turns those scores into a distribution P_G, from
which one negative is sampled. REINFORCE then updates G to assign
HIGHER probability to candidates D found confusing.

Two architectural notes:

  1. SHARED EMBEDDINGS WITH D.
     G does not own E (entity table) or R (relation table). It uses
     D's embeddings at forward time. This means G's optimizer only
     updates G's MLP head; D's optimizer updates E and R. The two
     networks learn from the same representations but optimize
     different things.

  2. THE MLP HEAD IS WHAT MAKES G DIFFERENT FROM D.
     D scores triples by a fixed function (-||E(h)+R(r)-E(t)||).
     G scores triples through a LEARNED MLP over concat([E(h), R(r),
     E(t)]). The MLP can express non-linear preferences D's distance
     metric can't, which is exactly the flexibility we need to pick
     hard-for-D negatives.

The MLP shape (input -> hidden -> hidden/2 -> 1) is small and standard
- the work is in the embeddings, not the head.
"""
import torch
import torch.nn as nn


DEFAULT_HIDDEN_DIM = 256


class CandidateScorer(nn.Module):
    """The Generator. Trainable parameters = MLP head only.

    Embeddings are owned by the Discriminator and passed in at forward
    time. This keeps gradients to embeddings flowing through D's
    optimizer (not G's), and keeps G's parameter count small.
    """

    def __init__(self, embedding_dim, hidden_dim=DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # MLP over concat([h_emb, r_emb, t_emb]) -> scalar score per triple.
        self.mlp = nn.Sequential(
            nn.Linear(3 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def score_from_emb(self, h_emb, r_emb, t_emb):
        """Score triples given pre-fetched embeddings.

        Args:
          h_emb, r_emb, t_emb: tensors of shape [..., dim].
                               All three must share the same leading dims.

        Returns:
          scores: tensor of shape [...] (the last dim of the MLP is squeezed).
        """
        features = torch.cat([h_emb, r_emb, t_emb], dim=-1)
        return self.mlp(features).squeeze(-1)

    def score_from_ids(self, h_ids, r_ids, t_ids, E, R):
        """Score triples by entity/relation IDs.

        Args:
          h_ids, r_ids, t_ids: LongTensors of arbitrary matching shape.
          E, R:                Discriminator's embedding tables.

        Returns:
          scores: tensor of the same shape as h_ids.
        """
        h_emb = E(h_ids)
        r_emb = R(r_ids)
        t_emb = E(t_ids)
        return self.score_from_emb(h_emb, r_emb, t_emb)

    def distribution_from_ids(self, candidate_triple_ids, E, R, weights=None):
        """Compute the softmax distribution P_G over candidate triples.

        Args:
          candidate_triple_ids: LongTensor of shape [..., N_S, 3].
                                Last dim holds (h_id, r_id, t_id) per row.
          E, R:                 Discriminator's embedding tables.
          weights:              optional FloatTensor of shape [..., N_S]
                                for cardinality weighting (multiplied into
                                the scores before softmax).

        Returns:
          P_G: FloatTensor of shape [..., N_S] - rows sum to 1 along last dim.
        """
        h_ids = candidate_triple_ids[..., 0]
        r_ids = candidate_triple_ids[..., 1]
        t_ids = candidate_triple_ids[..., 2]
        scores = self.score_from_ids(h_ids, r_ids, t_ids, E, R)
        if weights is not None:
            scores = scores * weights
        return torch.softmax(scores, dim=-1)

    def n_trainable_params(self):
        """Total trainable parameter count (MLP only - embeddings live in D)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
