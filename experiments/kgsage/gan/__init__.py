"""KGSAGE GAN layer.

The conditional GAN that produces single-slot-corruption negatives for ADKGD's
`--neg_source gan` training path.

  KGSAGEGenerator     - real triple + noise -> head/rel/tail logits (3 heads)
  gumbel_softmax      - reparameterised soft argmax (training helper)

Training lives at `kgsage.gan.train_aii` (the A-ii trainer). Generation (load checkpoint
+ produce negatives) lives at `kgsage.inference` — the same pipeline `kgsage_bridge`
uses to feed ADKGD.
"""
from kgsage.gan.models import (
    KGSAGEGenerator,
    gumbel_softmax,
)

__all__ = [
    "KGSAGEGenerator",
    "gumbel_softmax",
]
