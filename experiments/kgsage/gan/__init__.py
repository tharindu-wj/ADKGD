"""KGSAGE GAN layer.

The adversarial Generator + Discriminator pair that produces plausible-but-wrong
triples for ADKGD's `--neg_source gan` training path.

Current architecture (Phase 1 placeholder):
  - Simple 3-layer MLP Generator + Discriminator (the "simple GAN")
  - Single-slot corruption per call (head OR relation OR tail, chosen uniformly)
  - Trained via margin + reconstruction loss against random negatives

Phase 2 (future) upgrade plan:
  - Replace Generator with a pair-aware head: anchor (h, r, t) -> partner r'
  - Add encoder conditioning from the Phase 1 RGCN+DistMult checkpoint
  - Adversarially-trained on role-swap contradiction structure

Partner-template providers (the Phase 2 discriminator's positive supervision):
  observed_2cycle_templates  - v1: positives from observed reverse edges
                               (breaks on inverse-removed KGs like YAGO 4.5)
  mine_partner_templates     - v2-A: positives from length-1 rule confidences
                               (KG-agnostic, pure Python; see partner_templates.py)
  verify_partner_templates   - regression test / decision gate confirming the
                               v2-A miner reproduces Test 1.3's classification

Public API:
  Generator       - emits (h', r', t') logits given an input triple + noise
  Discriminator   - scores (real, candidate) triple pairs
  gumbel_softmax  - reparameterised soft argmax (training helper)
  soft_embedding  - lookup into an embedding table via a soft distribution
  observed_2cycle_templates / mine_partner_templates - partner-template providers

Generation (load checkpoint + produce negatives at inference) lives at
`kgsage.inference` — same pipeline used by `kgsage_bridge` to feed ADKGD.
"""
from kgsage.gan.models import (
    Generator,
    Discriminator,
    gumbel_softmax,
    soft_embedding,
)
from kgsage.gan.partner_templates import (
    observed_2cycle_templates,
    mine_partner_templates,
)

__all__ = [
    "Generator",
    "Discriminator",
    "gumbel_softmax",
    "soft_embedding",
    "observed_2cycle_templates",
    "mine_partner_templates",
]
