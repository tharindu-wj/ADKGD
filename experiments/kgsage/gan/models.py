"""Neural modules for the pair-aware KGSAGE GAN.

The role-swap contradiction generator + discriminator and the helpers they share:

  KGSAGEGenerator         anchor (h, r, t) -> distribution over partner relations r'
  KGSAGEDiscriminator     scores anchor (h, r, t) + role-swap partner (t, r', h)
  load_encoder_embeddings post-RGCN entity + DistMult relation embeddings from a
                          Phase 1 encoder checkpoint (what the GAN conditions on)
  gumbel_softmax          differentiable categorical sample (training helper)
"""
import torch
import torch.nn as nn


def gumbel_softmax(logits, tau=1.0):
    """Differentiable categorical sample.

    Normal `argmax` is not differentiable, so we can't backprop through it.
    The Gumbel-Softmax trick:
      1. Add Gumbel noise to the logits (so the sample is random).
      2. Take softmax with a low temperature `tau` (output becomes near-one-hot).
    The result looks like a one-hot vector but is smooth, so gradients flow.

    Used during training so we can sample a partner relation, look up its
    embedding, and let the gradient flow back into the generator.
    """
    noise = torch.rand_like(logits).clamp_(1e-10, 1.0 - 1e-10)
    gumbel = -torch.log(-torch.log(noise))
    return torch.softmax((logits + gumbel) / tau, dim=-1)


# ===========================================================================
# The PAIR-AWARE KGSAGE Generator + Discriminator.
#
# Spec: docs/THESIS_PLAN_pairgan_contradictions.md, "Phase 2".
#
# Both are CONDITIONED on the frozen Phase 1 encoder embeddings and work on the
# role-swap structure: anchor (h, r, t) + partner (t, r', h). The head/tail swap
# is STRUCTURAL — only the partner relation r' is sampled.
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
    the GAN — and the random-init smoke path in train.py — runs on machines
    without PyG (PyG is HPC-only in this project).
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
