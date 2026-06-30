"""KGSAGE GAN for knowledge-graph triple corruption.

The job of this GAN:
  Given a real triple like (Alice, born_in, Australia), produce a fake-but-
  plausible triple like (Carol, born_in, Australia) — same shape, slightly
  wrong content.

Two networks:
  KGSAGEGenerator     — takes a real triple + noise, outputs the fake triple.
  KGSAGEDiscriminator — sees a pair (real, candidate) and scores whether the
                        candidate looks real. The generator tries to fool it.

Both are plain MLPs that learn their OWN entity/relation embedding tables from
scratch during GAN training. One slot (head, relation, or tail) is corrupted
per generated negative — the convention used by the ADKGD baseline.
"""
import torch
import torch.nn as nn


class KGSAGEGenerator(nn.Module):
    """Generator: real triple + noise -> fake triple logits (one head per slot)."""

    def __init__(self, n_ent, n_rel, dim=64, z_dim=16, hidden=256):
        super().__init__()
        # Embedding tables: row i is a vector representing entity (or relation) i.
        # These are LEARNED during training — the GAN figures out what makes a
        # good vector representation of each entity/relation.
        self.ent_emb = nn.Embedding(n_ent, dim)
        self.rel_emb = nn.Embedding(n_rel, dim)
        nn.init.normal_(self.ent_emb.weight, std=0.1)
        nn.init.normal_(self.rel_emb.weight, std=0.1)

        # MLP: combine (h_emb, r_emb, t_emb, noise) into one hidden vector.
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Three "heads" — each predicts logits over a different vocabulary:
        #   head_out -> which entity should the corrupted head be?
        #   rel_out  -> which relation?
        #   tail_out -> which entity for the tail?
        # The corrupter (in inference.py) picks ONE of these per call.
        self.head_out = nn.Linear(hidden, n_ent)
        self.rel_out = nn.Linear(hidden, n_rel)
        self.tail_out = nn.Linear(hidden, n_ent)

        # Save sizes so the checkpoint loader can rebuild this model.
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.dim = dim
        self.z_dim = z_dim
        self.hidden = hidden

    def lookup(self, h, r, t):
        """Look up the embedding vectors for a batch of (h, r, t) triples."""
        return self.ent_emb(h), self.rel_emb(r), self.ent_emb(t)

    def forward(self, h, r, t, z):
        """Run the generator.

        Inputs:
          h, r, t : (batch,) integer IDs of the real triple
          z       : (batch, z_dim) random noise vector

        Returns three logit tensors:
          head_logits : (batch, n_ent)  — scores over all entities for new head
          rel_logits  : (batch, n_rel)  — scores over all relations
          tail_logits : (batch, n_ent)  — scores over all entities for new tail
        """
        h_emb, r_emb, t_emb = self.lookup(h, r, t)
        x = torch.cat([h_emb, r_emb, t_emb, z], dim=1)
        hidden = self.mlp(x)
        return self.head_out(hidden), self.rel_out(hidden), self.tail_out(hidden)


class KGSAGEDiscriminator(nn.Module):
    """Discriminator: sees a (real triple, candidate triple) pair and scores it.

    High score = "candidate looks like a real fact in this graph."
    Low score  = "candidate looks fake."

    The generator wins when the discriminator can't tell its outputs apart
    from real triples.
    """

    def __init__(self, dim=64, hidden=128):
        super().__init__()
        # Input is the two triple embeddings concatenated:
        #   real_emb     -> 3 * dim
        #   candidate    -> 3 * dim
        #   total        -> 6 * dim
        self.mlp = nn.Sequential(
            nn.Linear(6 * dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),  # one scalar score per pair
        )

    def forward(self, real_emb, candidate_emb):
        """Score a batch of (real, candidate) embedding pairs.

        real_emb, candidate_emb : (batch, 3, dim) each
        Returns : (batch, 1) scores (logits — apply sigmoid for probability)
        """
        x = torch.cat([real_emb.flatten(1), candidate_emb.flatten(1)], dim=1)
        return self.mlp(x)


def gumbel_softmax(logits, tau=1.0):
    """Differentiable categorical sample.

    Normal `argmax` is not differentiable, so we can't backprop through it.
    The Gumbel-Softmax trick:
      1. Add Gumbel noise to the logits (so the sample is random).
      2. Take softmax with a low temperature `tau` (output becomes near-one-hot).
    The result looks like a one-hot vector but is smooth, so gradients flow.

    Used during training so we can sample a fake triple, look up its
    embedding, and let the gradient flow back into the generator.
    """
    noise = torch.rand_like(logits).clamp_(1e-10, 1.0 - 1e-10)
    gumbel = -torch.log(-torch.log(noise))
    return torch.softmax((logits + gumbel) / tau, dim=-1)


def soft_embedding(soft_h, soft_r, soft_t, ent_weight, rel_weight):
    """Turn near-one-hot vectors back into embeddings, differentiably.

    `soft @ embedding_table` is the matrix-multiplication way of saying
    "look up the embedding". When `soft` is exactly one-hot, this is the same
    as `embedding_table[argmax(soft)]`. When `soft` is near-one-hot (from
    Gumbel-Softmax), the result is a smooth average that supports gradients.
    """
    h_emb = soft_h @ ent_weight
    r_emb = soft_r @ rel_weight
    t_emb = soft_t @ ent_weight
    return torch.stack([h_emb, r_emb, t_emb], dim=1)
