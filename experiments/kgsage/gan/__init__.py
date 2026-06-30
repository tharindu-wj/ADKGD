"""KGSAGE GAN layer.

The conditional GAN that produces single-slot-corruption negatives for ADKGD's
`--neg_source gan` training path.

  KGSAGEGenerator     - real triple + noise -> head/rel/tail logits (3 heads)
  KGSAGEDiscriminator - scores a (real, candidate) triple pair
  gumbel_softmax      - reparameterised soft argmax (training helper)
  soft_embedding      - lookup into an embedding table via a soft distribution

Training lives at `kgsage.gan.train` (cli.train_gan). Generation (load checkpoint
+ produce negatives) lives at `kgsage.inference` — the same pipeline `kgsage_bridge`
uses to feed ADKGD.
"""
from kgsage.gan.models import (
    KGSAGEGenerator,
    KGSAGEDiscriminator,
    gumbel_softmax,
    soft_embedding,
)

__all__ = [
    "KGSAGEGenerator",
    "KGSAGEDiscriminator",
    "gumbel_softmax",
    "soft_embedding",
]
