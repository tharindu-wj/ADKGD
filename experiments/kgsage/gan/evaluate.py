"""KGSAGE GAN generation-quality metrics — Phase 3 placeholder.

Phase 3 of the thesis evaluates the trained KGSAGE GAN's output quality
before it's plugged into ADKGD. The planned metrics are:

  precision         - what fraction of generated negatives are non-trivially wrong
                      (not in the training graph, not a self-loop, not duplicated)
  recall            - coverage of the contradiction templates uncovered in Test 1.3
  diversity         - entropy of the per-anchor partner-relation distribution
                      (low = mode-collapsed onto one or two partners)
  mode_collapse     - top-K most-frequent partner relations as a fraction of all output
  structural_check  - the role-swap shape is preserved (h<->t, r->r')

This file is intentionally a placeholder until Phase 2 (the pair-aware
KGSAGEGenerator) exists. The current simple GAN doesn't have meaningful
generation-quality metrics beyond train/val loss curves; those live in
`kgsage.gan.train`.

When Phase 2 lands, populate this module with:

    def evaluate_quality(checkpoint_path, *, dataset, n_samples=10_000):
        '''Run the full quality battery on a trained KGSAGE GAN checkpoint.
        Returns a dict of metric_name -> value, plus a JSON-serialisable detail
        block for the thesis writeup.'''
        ...

    def report(results):
        '''Pretty-print the metrics dict for a PASS/MARGINAL/FAIL decision gate.'''
        ...
"""
__all__ = []
