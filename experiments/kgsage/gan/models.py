"""Simple GAN for knowledge-graph triple corruption.

The job of this GAN:
  Given a real triple like (Alice, born_in, Australia), produce a fake-but-
  plausible triple like (Carol, born_in, Australia) — same shape, slightly
  wrong content.

Two networks:
  Generator     — takes a real triple + noise, outputs the fake triple.
  Discriminator — sees a pair (real, candidate) and scores whether the
                  candidate looks real. The generator tries to fool it.

Both are plain MLPs. No conv layers, no attention, no spectral norm — the
point of this version is to make the GAN concept readable.
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generator: real triple + noise -> fake triple logits."""

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
        # The corrupter (in corrupt_triples.py) picks ONE of these to actually use
        # per call (the others are wasted compute but easier to reason about).
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


class Discriminator(nn.Module):
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


# ===========================================================================
# Phase 2 — the PAIR-AWARE KGSAGE Generator + Discriminator.
#
# Spec: docs/THESIS_PLAN_pairgan_contradictions.md, "Phase 2" (lines 293-366).
#
# Difference from the simple Generator/Discriminator above:
#   * They learn their own embeddings from scratch and corrupt ONE slot.
#   * These two are CONDITIONED on the frozen Phase 1 encoder embeddings, and
#     work on the role-swap structure: anchor (h, r, t) + partner (t, r', h).
#     The head/tail swap is STRUCTURAL — only the partner relation r' is sampled.
# ===========================================================================


class KGSAGEGenerator(nn.Module):
    """Anchor (h, r, t) -> distribution over partner relations r'.

    The role-swap (h <-> t) is forced structurally; this network only decides
    WHICH relation r' makes (t, r', h) a contradiction. Conditioned on the
    Phase 1 encoder embeddings (post-RGCN entity embeddings + DistMult relation
    embeddings) passed in as tensors via `load_encoder_embeddings`.

    Args:
        ent_emb_pretrained : (n_ent, dim) post-RGCN entity embeddings
        rel_emb_pretrained : (n_rel, dim) DistMult relation embeddings
        freeze_ent/rel     : if True, keep the pretrained table fixed; default
                             False so the GAN may fine-tune (plan line 306).
    """

    def __init__(self, ent_emb_pretrained, rel_emb_pretrained, hidden=256,
                 freeze_ent=False, freeze_rel=False):
        super().__init__()
        self.ent_emb = nn.Embedding.from_pretrained(ent_emb_pretrained, freeze=freeze_ent)
        self.rel_emb = nn.Embedding.from_pretrained(rel_emb_pretrained, freeze=freeze_rel)
        self.n_ent = ent_emb_pretrained.size(0)
        self.n_rel = rel_emb_pretrained.size(0)
        self.dim = ent_emb_pretrained.size(1)
        self.hidden = hidden
        self.mlp = nn.Sequential(
            nn.Linear(3 * self.dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_rel),   # logits over partner relations
        )

    def forward(self, h, r, t):
        """h, r, t : (batch,) integer IDs. Returns (batch, n_rel) logits."""
        x = torch.cat([self.ent_emb(h), self.rel_emb(r), self.ent_emb(t)], dim=-1)
        return self.mlp(x)


class KGSAGEDiscriminator(nn.Module):
    """Score the joint plausibility of (anchor, partner) under the role-swap.

    Sees 6 embeddings: anchor (h, r, t) + partner (t, r', h). Returns one logit
    per pair (sigmoid -> probability the pair is a genuine contradiction).

    The relation/entity tables default to FROZEN here — the discriminator scores
    in the fixed Phase 1 geometry, which also lets the generator's differentiable
    `soft @ rel_emb.weight` partner embedding flow gradients cleanly through a
    constant matrix.
    """

    def __init__(self, ent_emb_pretrained, rel_emb_pretrained, hidden=256,
                 freeze_ent=True, freeze_rel=True):
        super().__init__()
        self.ent_emb = nn.Embedding.from_pretrained(ent_emb_pretrained, freeze=freeze_ent)
        self.rel_emb = nn.Embedding.from_pretrained(rel_emb_pretrained, freeze=freeze_rel)
        self.n_ent = ent_emb_pretrained.size(0)
        self.n_rel = rel_emb_pretrained.size(0)
        self.dim = ent_emb_pretrained.size(1)
        self.hidden = hidden
        self.mlp = nn.Sequential(
            nn.Linear(6 * self.dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, h, r, t, r_partner):
        """Score anchor (h, r, t) + partner (t, r_partner, h).

        r_partner may be either:
          * a (batch,) LongTensor of relation IDs (a HARD partner — used for the
            real-positive and discriminator-negative passes), or
          * a (batch, dim) FloatTensor partner-relation embedding (a SOFT sample
            from the generator — differentiable; used for the adversarial G step).
        """
        rp_emb = self.rel_emb(r_partner) if r_partner.dim() == 1 else r_partner
        h_e, t_e = self.ent_emb(h), self.ent_emb(t)
        x = torch.cat([
            h_e, self.rel_emb(r), t_e,   # anchor  (h, r, t)
            t_e, rp_emb,          h_e,   # partner (t, r', h)  <- role-swap
        ], dim=-1)
        return self.mlp(x)


def load_encoder_embeddings(encoder_ckpt_path, kg, device=None):
    """Extract the Phase 1 embeddings the KGSAGE GAN conditions on.

    Returns (ent_emb, rel_emb, ent2id, rel2id) where:
        ent_emb : (n_ent, dim) POST-RGCN entity embeddings — structure-aware,
                  computed by running the trained encoder forward ONCE on the
                  full training graph (NOT the raw layer-0 lookup table).
        rel_emb : (n_rel, dim) DistMult relation embeddings — the table Test 1.2
                  validated as encoding anti-symmetric structure.

    torch_geometric is imported LAZILY inside this function so that the rest of
    the GAN — and the random-init smoke path in train_kgsage.py — runs on
    machines without PyG (PyG is HPC-only in this project).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from kgsage.encoder.models import KGSAGELinkPredictor  # lazy: pulls in PyG
    model, ent2id, rel2id = KGSAGELinkPredictor.load_pretrained(encoder_ckpt_path, device)
    with torch.no_grad():
        ent_emb = model.encoder(
            kg["edge_index"].to(device), kg["edge_type"].to(device),
        ).detach().clone()                                       # (n_ent, dim) post-RGCN
        rel_emb = model.decoder.rel_emb.weight.detach().clone()  # (n_rel, dim)
    return ent_emb, rel_emb, ent2id, rel2id
