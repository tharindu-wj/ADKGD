"""The candidate-scoring generator G (paper Phase 2: Adversarial Generator
Training).

G's job: given a real triple (h, r, t) and a slot to corrupt (head or tail),
score that triple's candidate set so that the best-scoring candidates are
plausible-but-false fillers which the anchor's neighbourhood does NOT
corroborate. The entity keeping its slot is the anchor; the original value of
the corrupted slot is the true_filler; the candidate G selects is the
picked_candidate, and the triple it emits is a corruption.

It is a scoring model, not a vocabulary-wide classifier: it has NO per-entity
output parameters. (An earlier design with a global output layer collapsed
into ranking entities by popularity — per-entity weights are exactly where a
popularity shortcut can be stored. Removing them closes that door.)

How a score is produced:
  Query      : q = MLP([E'(h) | relation | E'(t) | projected sketch(anchor)])
               — the membership sketch is the set-readable neighbourhood
               signal that the pooled 64-d E' cannot carry (see
               membership_sketch.py).
  Candidate  : f(x) = MLP(E'(x)) — a shared tower over the candidate's row of
               the context table E', identical for every entity.
  Logit      : q · f(x) / sqrt(d)  -  log q(x)   (logQ sampling correction,
               see candidate_sampler.py).
  Slot       : one query projection per corrupted slot (head / tail),
               shared trunk underneath.
"""
import torch
import torch.nn as nn


def gumbel_softmax(logits, tau=1.0, hard=False, mask=None, generator=None):
    """Differentiable categorical sample (the Gumbel-Softmax trick).

    A plain argmax pick is not differentiable, so the generator could not
    learn from what happens to its pick. Instead: (1) add Gumbel noise to the
    logits, (2) take a low-temperature softmax, giving a nearly-one-hot but
    smooth selection that gradients can flow through.

    Options (all default-off; the base call behaves identically):
      mask      : additive [batch, n] mask (0 = allowed, -inf = banned),
                  applied BEFORE the noise so banned candidates can never be
                  sampled.
      hard      : straight-through estimator — exact one-hot on the forward
                  pass, soft gradient on the backward pass. Matches the hard
                  picked_candidate used at corruption time, so the
                  discriminators never see a soft-vs-hard difference they
                  could exploit.
      generator : optional torch.Generator for reproducible sampling.
    """
    if mask is not None:
        logits = logits + mask
    uniform_noise = torch.empty_like(logits)
    uniform_noise.uniform_(generator=generator).clamp_(1e-10, 1.0 - 1e-10)
    gumbel_noise = -torch.log(-torch.log(uniform_noise))
    soft_sample = torch.softmax((logits + gumbel_noise) / tau, dim=-1)
    if not hard:
        return soft_sample
    one_hot = torch.zeros_like(soft_sample).scatter_(
        -1, soft_sample.argmax(dim=-1, keepdim=True), 1.0)
    return one_hot - soft_sample.detach() + soft_sample  # fwd: one-hot, bwd: soft


class CandidateScoringGenerator(nn.Module):
    """Scores per-triple candidate sets, conditioned on E' + the anchor's
    membership sketch.

    NOTE: `relation_embedding`, `sketch_proj`, `trunk`, `q_head`, `q_tail`
    and `cand_tower` are FROZEN names — they ARE the state-dict keys of the
    locked generator_*.pt artifacts, so renaming any of them makes every
    saved checkpoint unloadable. Decoder: `cand_tower` = the shared candidate
    tower f(x); `sketch_proj` projects the membership sketch down to `dim`.
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

    def forward(self, head_ids, relation_ids, tail_ids, context_table,
                anchor_sketch_rows, candidate_ids, candidate_log_q, slot):
        """Return logits [batch, K] over each row's candidate set.

        context_table      : the frozen context table E', [n_ent, dim] — the
                             OUTPUT of the neighbourhood context encoder.
        anchor_sketch_rows : float [batch, sketch_bits] — membership sketch of
                             each row's ANCHOR (the entity keeping its slot),
                             already gathered by the caller.
        candidate_ids      : long [batch, K] candidate entity ids.
        candidate_log_q    : float [batch, K] log sampling probabilities, for
                             the logQ correction.
        slot               : which slot is being corrupted (0 = head, 2 = tail).
        """
        conditioning = torch.cat([
            context_table[head_ids],
            self.relation_embedding(relation_ids),
            context_table[tail_ids],
            self.sketch_proj(anchor_sketch_rows),
        ], dim=1)
        hidden = self.trunk(conditioning)
        query = (self.q_tail if slot == 2 else self.q_head)(hidden)     # [B, d]
        candidate_features = self.cand_tower(context_table[candidate_ids])
        logits = torch.einsum("bd,bkd->bk",
                              query, candidate_features) / (self.d_model ** 0.5)
        return logits - candidate_log_q
