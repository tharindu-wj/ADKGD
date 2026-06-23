"""KGSAGE ↔ ADKGD bridge.

The only Python module in this repo that knows about BOTH the standalone
KGSAGE package and the ADKGD detector. Its job is to translate between
KGSAGE's library API and the three-function bridge contract that ADKGD's
dataset.py expects (load_gan, generate, render_stats — same as
experiments/gan/adkgd_bridge.py for the simple GAN).

WHY THIS IS A SEPARATE FOLDER (not inside experiments/kgsage/):
  KGSAGE is meant to ship as a standalone package — `pip install kgsage`
  someday. A standalone package can't know about ADKGD. So integration code
  lives here instead, sibling to `kgsage/` rather than nested inside it.

Currently empty — populated in Phase 4 of the thesis plan, once the KGSAGE
GAN exists. Until then, ADKGD continues to use `experiments/gan/adkgd_bridge.py`
(the simple conditional GAN bridge).
"""
__all__ = []
