"""KGSAGE GAN layer.

The conditional GAN that produces single-slot-corruption negatives for a
downstream detector's training path (e.g. ADKGD's `--neg_source gan`).

  KGSAGEGenerator     - real triple + noise -> head/rel/tail logits (3 heads)
  gumbel_softmax      - reparameterised soft argmax (training helper)

Training lives at `kgsage.gan.train` (the trainer). Generation (load checkpoint
+ produce negatives) lives at `kgsage.inference` — the same pipeline `kgsage_bridge`
uses to feed the downstream detector.
"""
from kgsage.gan.models import (
    KGSAGEGenerator,
    gumbel_softmax,
)

__all__ = [
    "KGSAGEGenerator",
    "gumbel_softmax",
]
