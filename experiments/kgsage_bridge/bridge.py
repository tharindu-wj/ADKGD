"""KGSAGE ↔ ADKGD bridge — placeholder for Phase 4 integration.

PURPOSE:
  Provides the three-function ADKGD-side API (load_gan, generate,
  render_stats) that `dataset.py:Reader._gan_negatives` already calls
  when invoked with `--neg_source kgsage`. Same contract as
  `experiments/gan/adkgd_bridge.py` for the simple GAN.

IMPLEMENTATION STATUS:
  STUB. Phase 4 will fill these in once the Phase 2 KGSAGE Generator
  exists. Until then:
    - ADKGD with `--neg_source random` (the paper baseline) works.
    - ADKGD with `--neg_source gan`    (simple conditional GAN) works.
    - ADKGD with `--neg_source kgsage` (this file) ImportErrors with a
      clear message pointing to the thesis plan.

WHAT load_gan / generate / render_stats WILL DO (Phase 4):

  load_gan(ckpt_path, device) → loaded KGSAGE model + ent2id + rel2id
    Loads the joint encoder+GAN checkpoint produced by Phase 2 training.
    Returns the model in eval mode, plus its STRING vocab so the caller
    can translate between ADKGD's IDs and KGSAGE's IDs.

  generate(model, batch, adkgd_ent2id, adkgd_rel2id, n_per_anchor=1) → negatives
    For each anchor (h, r, t) in batch, the KGSAGE Generator emits the
    role-swapped partner (t, r', h) where r' is a contradicting relation
    sampled from the learned distribution. Vocab translation happens at
    the boundary so ADKGD only ever sees its own IDs.
    Returns a tensor of shape (B*n_per_anchor, 3) in ADKGD's vocab.

  render_stats(stats_dict) → multi-line string
    Formats KGSAGE generation stats (mode-collapse score, structural
    correctness, OOV rate) for ADKGD's training log.

WHY THIS FILE IS OUTSIDE kgsage/:
  We want kgsage/ to be a standalone package that doesn't know about
  ADKGD. Application code that knows about both worlds lives here instead.

NEXT TASKS (when Phase 2 of KGSAGE is built):
  1. Decide on KGSAGE checkpoint format (consensus with kgsage.gan.train).
  2. Implement load_gan() by calling kgsage.KGSAGE.load_pretrained().
  3. Implement generate() with vocab translation.
  4. Implement render_stats() with the metrics from Phase 3.
  5. Wire up SLURM script: experiments/slurm/run_adkgd_with_kgsage_fb15k237.slurm.
"""


def load_gan(ckpt_path, device=None, **kwargs):
    """Load a KGSAGE checkpoint for ADKGD bridge use.

    Phase 4 implementation. See module docstring for the planned signature.
    """
    raise NotImplementedError(
        "KGSAGE bridge not yet implemented — Phase 2 (KGSAGE Generator) must "
        "exist first. See experiments/docs/THESIS_PLAN_pairgan_contradictions.md."
    )


def generate(model, batch, adkgd_ent2id, adkgd_rel2id, n_per_anchor=1, **kwargs):
    """Generate role-swap contradiction partners for an ADKGD training batch.

    Phase 4 implementation. See module docstring for the planned signature.
    """
    raise NotImplementedError(
        "KGSAGE bridge not yet implemented — Phase 2 (KGSAGE Generator) must "
        "exist first. See experiments/docs/THESIS_PLAN_pairgan_contradictions.md."
    )


def render_stats(stats):
    """Format generation statistics for ADKGD's log output.

    Phase 4 implementation. See module docstring for the planned signature.
    """
    return "  (KGSAGE bridge: stats reporting not yet implemented)"
