"""Phase 3 - Corruption Module.

Uses the trained CGSP Generator + Discriminator to produce hard,
type-coherent negatives at inference time.

Public API:
  from experiments.gan.corruption.api import KGCorrupter

  corrupter = KGCorrupter(
      checkpoint_path="experiments/gan/outputs/checkpoints/FB15K-237_cgsp.pt",
      concept_pools_path="experiments/gan/outputs/concept_pools/FB15K-237.pkl",
  )
  negative = corrupter.corrupt(positive_string_triple, seed=42)

Sub-modules:
  api       KGCorrupter class - the primitive consumers instantiate.
  infer     load_checkpoint() + one-triple inference internals.
  adkgd_bridge   Consumer 1: ~30-line ADKGD wrapper (Phase 3.2).
"""
