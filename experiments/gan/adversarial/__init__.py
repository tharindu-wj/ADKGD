"""Phase 2 - Adversarial Module.

The GAN core: a Generator that learns to score hard candidates and a
Discriminator that learns to tell real triples from the Generator's
suggestions. Together they implement Full CGSP training via REINFORCE.

Sub-modules:
  discriminator   internal TransE scorer (D)
  candidate_pool  per-positive candidate set builder
  generator       MLP candidate scorer (G)   - Phase 2.2
  train           REINFORCE training loop    - Phase 2.3

This module consumes the concept_pools.pkl produced by Phase 1 and
produces a trained checkpoint consumed by Phase 3.
"""
