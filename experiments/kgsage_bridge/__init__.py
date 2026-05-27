"""KGSAGE <-> ADKGD bridge.

The single Python module in this repo that knows about BOTH the standalone
KGSAGE package and the ADKGD detector. ADKGD's `dataset.py` imports the
three-function contract from here:

  - load_gan(ckpt_path)
  - generate(triples, *, payload, adkgd_id2ent, ..., rng=None)
  - render_stats(stats)

KGSAGE implements this contract, so the call site in `dataset.py` never needs
to change as the generator evolves. All three names are FROZEN: dataset.py
lives outside experiments/ and imports them by name.

WHY THIS FOLDER EXISTS (and lives here, not in the KGSAGE repo):
  KGSAGE ships as a standalone, pip-installable library, and a standalone
  library cannot know about ADKGD's specific id spaces or types. So the
  integration glue lives on the detector's side of the seam -- here. This is
  the only module in this repo permitted to import `kgsage.*`.

THE VOCABULARY SEAM (this is the point of the folder):
  Inside `kgsage/` the emitted false triple is a CORRUPTION — that is the
  package's own word for the object it produces.
  Crossing into the detector, the SAME object is called a NEGATIVE when it
  trains ADKGD and an ANOMALY when it evaluates ADKGD, because that is
  ADKGD's vocabulary for what it is being fed.
  One object, three words, each correct on its own side of this file. That
  is a deliberate boundary, not terminology drift — so the `gan`/`negative`
  wording below is kept on purpose.
"""
from kgsage_bridge.bridge import load_gan, generate, render_stats

__all__ = ["load_gan", "generate", "render_stats"]
