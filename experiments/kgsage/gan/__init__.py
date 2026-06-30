"""KGSAGE GAN layer.

The pair-aware role-swap contradiction generator that supplies ADKGD's training
negatives on the `--neg_source gan` path.

  KGSAGEGenerator       - anchor (h, r, t) -> distribution over partner relations r'
  KGSAGEDiscriminator   - scores anchor (h, r, t) + role-swap partner (t, r', h)
  load_encoder_embeddings - post-RGCN entity + DistMult relation embeddings from a
                            Phase 1 encoder checkpoint (what the GAN conditions on)
  gumbel_softmax        - reparameterised soft argmax (training helper)
  mine_partner_templates- the discriminator's recon-prior supervision (rule-mined
                          anti-symmetric partner relations; see partner_templates.py)

Training lives at `kgsage.gan.train` (cli.train_gan). Generation (load checkpoint
+ produce negatives) lives at `kgsage.inference` — the same pipeline `kgsage_bridge`
uses to feed ADKGD.
"""
from kgsage.gan.models import (
    gumbel_softmax,
    KGSAGEGenerator,
    KGSAGEDiscriminator,
    load_encoder_embeddings,
)
from kgsage.gan.partner_templates import mine_partner_templates

__all__ = [
    "KGSAGEGenerator",
    "KGSAGEDiscriminator",
    "load_encoder_embeddings",
    "gumbel_softmax",
    "mine_partner_templates",
]
