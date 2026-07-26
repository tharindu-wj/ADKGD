"""KGSAGE adversarial training stack (dual-discriminator, candidate_v2).

Modules, named after the paper's Methodology section:
    train                         -- the trainer (python -m kgsage.gan.train):
                                     dual-discriminator game, PI-controlled
                                     penalty weight, per-epoch snapshots
    generator                     -- CandidateScoringGenerator + gumbel_softmax
    plausibility_discriminator    -- D_real: "could this triple be real?"
    neighbourhood_discriminator   -- D_match: "does the filler fit this
                                     anchor's neighbourhood?"
    neighbourhood_context_encoder -- RGCN warm-up producing the frozen
                                     context table E' (needs PyG)
    membership_sketch             -- Bloom sketches of 1-2 hop neighbour sets
    candidate_sampler             -- per-triple candidate sets + logQ correction

Corruption generation (load a checkpoint + produce negatives) lives at
`kgsage.corruption_generation` -- the same pipeline `kgsage_bridge` uses to
feed the downstream detector.
"""
from kgsage.gan.generator import CandidateScoringGenerator, gumbel_softmax
from kgsage.gan.plausibility_discriminator import PlausibilityDiscriminator
from kgsage.gan.neighbourhood_discriminator import NeighbourhoodDiscriminator

__all__ = [
    "CandidateScoringGenerator",
    "gumbel_softmax",
    "PlausibilityDiscriminator",
    "NeighbourhoodDiscriminator",
]
