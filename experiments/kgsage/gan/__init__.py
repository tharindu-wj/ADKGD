"""KGSAGE GAN layer.

The adversarial Generator + Discriminator pair that produces plausible-but-wrong
triples for ADKGD's `--neg_source gan` training path.

Two architectures live here:

  Phase 1 placeholder — the SIMPLE GAN (Generator/Discriminator):
  - Simple 3-layer MLP Generator + Discriminator
  - Single-slot corruption per call (head OR relation OR tail, chosen uniformly)
  - Trained via margin + reconstruction loss against random negatives
  - Still used by ADKGD's `--neg_source gan` baseline. Train: cli.train_gan.

  Phase 2 — the PAIR-AWARE GAN (KGSAGEGenerator/KGSAGEDiscriminator):
  - Generator: anchor (h, r, t) -> distribution over partner relations r'
  - Conditioned on the Phase 1 RGCN+DistMult encoder embeddings
    (post-RGCN entity embeddings + DistMult relation table) via
    load_encoder_embeddings
  - Adversarially trained on role-swap contradiction structure; positives are
    the partner-template providers below. Train: cli.train_kgsage_gan.
    Generation: kgsage.inference.generate_kgsage_partners.

Partner-template providers (the Phase 2 discriminator's positive supervision):
  observed_2cycle_templates  - v1: positives from observed reverse edges
                               (breaks on inverse-removed KGs like YAGO 4.5)
  mine_partner_templates     - v2-A: positives from length-1 rule confidences
                               (KG-agnostic, pure Python; see partner_templates.py)
  verify_partner_templates   - regression test / decision gate confirming the
                               v2-A miner reproduces Test 1.3's classification

Public API:
  Generator             - (simple) emits (h', r', t') logits given a triple + noise
  Discriminator         - (simple) scores (real, candidate) triple pairs
  KGSAGEGenerator       - (Phase 2) anchor (h, r, t) -> partner-relation logits
  KGSAGEDiscriminator   - (Phase 2) scores anchor + role-swap partner (t, r', h)
  load_encoder_embeddings - post-RGCN ent_emb + DistMult rel_emb from a Phase 1 ckpt
  gumbel_softmax        - reparameterised soft argmax (training helper)
  soft_embedding        - lookup into an embedding table via a soft distribution
  observed_2cycle_templates / mine_partner_templates - partner-template providers

Generation (load checkpoint + produce negatives at inference) lives at
`kgsage.inference` — same pipeline used by `kgsage_bridge` to feed ADKGD.
"""
from kgsage.gan.models import (
    Generator,
    Discriminator,
    gumbel_softmax,
    soft_embedding,
    KGSAGEGenerator,
    KGSAGEDiscriminator,
    load_encoder_embeddings,
)
from kgsage.gan.partner_templates import (
    observed_2cycle_templates,
    mine_partner_templates,
)

__all__ = [
    # Phase 1 placeholder — simple single-slot-corruption GAN.
    "Generator",
    "Discriminator",
    "gumbel_softmax",
    "soft_embedding",
    # Phase 2 — pair-aware role-swap contradiction GAN.
    "KGSAGEGenerator",
    "KGSAGEDiscriminator",
    "load_encoder_embeddings",
    # Partner-template providers (Phase 2 discriminator positive supervision).
    "observed_2cycle_templates",
    "mine_partner_templates",
]
