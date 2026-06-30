"""KGSAGE <-> ADKGD bridge.

The single Python module in this repo that knows about BOTH the standalone
KGSAGE package and the ADKGD detector. ADKGD's `dataset.py` imports the
three-function contract from here:

  - load_gan(ckpt_path)
  - generate(triples, *, payload, adkgd_id2ent, ..., rng=None)
  - render_stats(stats)

The KGSAGE conditional GAN implements this contract, so the call site in
`dataset.py` never needs to change as the generator evolves.

WHY THIS IS A SEPARATE FOLDER (not inside experiments/kgsage/):
  We want `kgsage/` to ship as a standalone library. A standalone library
  can't know about ADKGD's specific types. So integration glue lives here.
"""
from kgsage_bridge.bridge import load_gan, generate, render_stats

__all__ = ["load_gan", "generate", "render_stats"]
