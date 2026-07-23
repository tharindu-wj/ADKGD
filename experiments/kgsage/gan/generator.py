"""KGSAGE generator architecture (candidate_v2).

CandidateScoringGenerator scores a per-triple CANDIDATE SET instead of the
whole vocabulary, conditioned on the frozen context table E' (kgsage.gan.encoder)
and the anchor's neighbourhood sketch (kgsage.gan.sketch). It holds NO
per-entity output parameters, so a popularity ranking has no private storage.
The two discriminators live in kgsage.gan.d_real / kgsage.gan.d_match; the
trainer is kgsage.gan.train. gumbel_softmax provides the straight-through
selection over the candidate scores (hard forward, soft gradients).
"""
import torch
import torch.nn as nn


def gumbel_softmax(logits, tau=1.0, hard=False, mask=None, generator=None):
    """Differentiable categorical sample (Gumbel-Softmax trick).

    A plain argmax is not differentiable, so we cannot backprop through a hard
    pick. Instead we (1) add Gumbel noise to the logits, then (2) take a
    low-temperature softmax so the result is near one-hot but smooth. Gradients
    then flow, letting us sample a corrupted slot, embed it, and update G.

    Optional additions (all default-off; the base call behaves identically):
      mask      : additive [B, n] mask (0 allowed / -inf banned), applied BEFORE
                  the noise so banned candidates are unsampleable by
                  construction (type pools + known-true + self-loop bans).
      hard      : straight-through estimator -- one-hot(argmax) on the forward
                  pass, soft gradient on the backward pass. Removes the
                  soft-vs-hard artifact the discriminator could exploit and
                  matches the hard decode used at deployment.
      generator : optional torch.Generator for reproducible sampling.
    """
    if mask is not None:
        logits = logits + mask
    noise = torch.empty_like(logits)
    noise.uniform_(generator=generator).clamp_(1e-10, 1.0 - 1e-10)
    gumbel = -torch.log(-torch.log(noise))
    soft = torch.softmax((logits + gumbel) / tau, dim=-1)
    if not hard:
        return soft
    one_hot = torch.zeros_like(soft).scatter_(
        -1, soft.argmax(dim=-1, keepdim=True), 1.0)
    return one_hot - soft.detach() + soft  # forward: one-hot, backward: soft


class CandidateScoringGenerator(nn.Module):
    """KGSAGE-2 generator: scores a per-triple CANDIDATE SET instead of the
    whole vocabulary. There are deliberately NO per-entity output parameters --
    the v1 global Linear(hidden, n_ent) head is where the measured popularity
    prior lived (its ranking survived deleting the anchor context entirely).

    Query side  : q = MLP([E'(h) | rho_r | E'(t) | proj(sketch(anchor))])
                  -- the Bloom sketch is the set-readable neighbourhood input
                  that the 64-d pooled E' provably cannot provide.
    Candidate   : f(x) = MLP(E'(x))          (shared tower, no free per-entity
                  weights, so a popularity prior has no storage)
    Logits      : q . f(x) / sqrt(d)  -  logq(x)   (logQ sampling correction)
    Slot        : one query projection per slot (head / tail), shared trunk.
    """

    def __init__(self, dim=64, sketch_bits=8192, d_model=128, hidden=256,
                 n_rel=None):
        super().__init__()
        self.relation_embedding = nn.Embedding(n_rel, dim)
        nn.init.normal_(self.relation_embedding.weight, std=0.1)
        self.sketch_proj = nn.Linear(sketch_bits, dim, bias=False)
        self.trunk = nn.Sequential(
            nn.Linear(4 * dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.q_head = nn.Linear(hidden, d_model)   # query when corrupting HEAD
        self.q_tail = nn.Linear(hidden, d_model)   # query when corrupting TAIL
        self.cand_tower = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, d_model),
        )
        self.d_model = d_model
        self.sketch_bits = sketch_bits

    def forward(self, h_ids, r_ids, t_ids, entity_context, sketch_rows,
                cand_ids, cand_logq, slot):
        """logits [B, K] over the candidate set for one slot.

        sketch_rows : float [B, sketch_bits] -- sketch of the ANCHOR entity
                      (the slot-keeping one), already gathered by the caller.
        cand_ids    : long [B, K]; cand_logq: float [B, K].
        """
        cond = torch.cat([
            entity_context[h_ids],
            self.relation_embedding(r_ids),
            entity_context[t_ids],
            self.sketch_proj(sketch_rows),
        ], dim=1)
        hid = self.trunk(cond)
        q = (self.q_tail if slot == 2 else self.q_head)(hid)      # [B, d]
        f = self.cand_tower(entity_context[cand_ids])             # [B, K, d]
        logits = torch.einsum("bd,bkd->bk", q, f) / (self.d_model ** 0.5)
        return logits - cand_logq
