"""KGSAGE conditional GAN for knowledge-graph triple corruption.

WHAT THIS GAN DOES
    Given a real triple like (Alice, born_in, Australia), the generator produces
    a fake-but-plausible triple with ONE slot corrupted, e.g.
    (Alice, born_in, Canada). Those corruptions become a downstream detector's training negatives.

NEIGHBOURHOOD CONDITIONING
    The generator does NOT learn its own entity embeddings. Instead it is
    CONDITIONED on E' — the context embeddings produced by the RGCN encoder
    (kgsage.gan.encoder), where E'[e] summarises entity e's neighbourhood. So the
    generator sees each entity's surroundings and can pick a corruption that is
    type-valid but contradicts the head's converging context (a "near-miss" the
    downstream detector's neighbourhood channel must then learn to catch).

    The generator's conditioning vector is:
        [ E'[head] | relation_embedding[relation] | E'[tail] | noise ]
    Entity slots come from E' (supplied by the encoder at forward time); the
    relation slot uses the generator's OWN learned relation embedding table.

THE NETWORK
    KGSAGEGenerator - entity context + noise -> corrupted-triple logits (3 heads)
    (the discriminator lives in kgsage.gan.discriminator + frozen_complex)

The entity embedding table used for conditioning is E'; in the trainer
(kgsage.gan.train) E' is LP-warmup-trained and then FROZEN for the
adversarial phase.
"""
import torch
import torch.nn as nn


class KGSAGEGenerator(nn.Module):
    """Generator: entity context + noise -> logits for a one-slot corruption.

    Conditioned on externally-supplied entity context vectors E' (from the RGCN
    encoder) instead of learning its own entity embeddings. It keeps a small
    learned RELATION embedding table, because relations are not entities and the
    encoder only contextualises entities.
    """

    def __init__(self, n_ent, n_rel, dim=64, z_dim=16, hidden=256):
        super().__init__()

        # Learned relation embedding table. Entities are deliberately NOT stored
        # here — the generator receives entity context E' from the encoder at
        # forward time (so the encoder, not the generator, owns entity vectors).
        self.relation_embedding = nn.Embedding(n_rel, dim)
        nn.init.normal_(self.relation_embedding.weight, std=0.1)

        # MLP that turns the conditioning vector into a shared hidden state.
        # Input width = E'[head] (dim) + relation_embedding (dim)
        #             + E'[tail] (dim) + noise (z_dim)  =  3*dim + z_dim.
        self.conditioning_mlp = nn.Sequential(
            nn.Linear(3 * dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Three prediction heads — each maps the hidden state to a distribution
        # over the vocabulary for ONE slot. The corrupter (inference.py) uses
        # exactly one of these per generated negative.
        self.head_slot_predictor = nn.Linear(hidden, n_ent)      # entity for a new head
        self.relation_slot_predictor = nn.Linear(hidden, n_rel)  # relation for a new relation
        self.tail_slot_predictor = nn.Linear(hidden, n_ent)      # entity for a new tail

        # Remember sizes so the checkpoint loader can rebuild this module.
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.dim = dim
        self.z_dim = z_dim
        self.hidden = hidden

    def gather_conditioning_embeddings(self, head_ids, relation_ids, tail_ids,
                                       entity_context):
        """Collect the vectors that condition generation for a batch of triples.

        Entity slots are looked up in `entity_context` (E' from the encoder) so
        the generator sees each entity's neighbourhood; the relation slot uses
        the generator's own learned relation embedding.

        Args:
            head_ids, relation_ids, tail_ids : LongTensor [batch] real-triple ids.
            entity_context : FloatTensor [n_ent, dim] = E' from the encoder.

        Returns:
            (head_context, relation_embedding, tail_context), each [batch, dim].
        """
        head_context = entity_context[head_ids]                 # E'[head]
        tail_context = entity_context[tail_ids]                 # E'[tail]
        relation_embedding = self.relation_embedding(relation_ids)
        return head_context, relation_embedding, tail_context

    def forward(self, head_ids, relation_ids, tail_ids, noise, entity_context):
        """Run the generator for a batch of real triples.

        Args:
            head_ids, relation_ids, tail_ids : LongTensor [batch] real-triple ids.
            noise          : FloatTensor [batch, z_dim] random noise (adds variety
                             so the same triple can yield different corruptions).
            entity_context : FloatTensor [n_ent, dim] = E' from the RGCN encoder.
                             The trainer passes a frozen, detached E'.

        Returns three logit tensors:
            head_logits     : [batch, n_ent] scores over entities for a new head
            relation_logits : [batch, n_rel] scores over relations
            tail_logits     : [batch, n_ent] scores over entities for a new tail
        """
        head_context, relation_embedding, tail_context = self.gather_conditioning_embeddings(
            head_ids, relation_ids, tail_ids, entity_context,
        )
        conditioning_vector = torch.cat(
            [head_context, relation_embedding, tail_context, noise], dim=1,
        )
        hidden_state = self.conditioning_mlp(conditioning_vector)
        return (
            self.head_slot_predictor(hidden_state),
            self.relation_slot_predictor(hidden_state),
            self.tail_slot_predictor(hidden_state),
        )


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
