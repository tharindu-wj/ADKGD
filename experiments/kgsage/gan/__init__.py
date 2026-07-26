"""KGSAGE adversarial training stack: the dual-discriminator architecture.

Read `gan/` as "the adversarial training stack" -- it holds the context
encoder, the membership sketches and the candidate sampler as well as the GAN
itself. The folder name is kept because `python -m kgsage.gan.train` is the
documented entry point used by the SLURM launchers, commands.md and the
notebook.

Architecture string in every checkpoint: "candidate_v2" (frozen literal; it
names this dual-discriminator architecture and is compared as a string in
corruption_generation.py and knockout_eval.py).

Modules, named after the paper's Methodology section:

  Phase 1 -- Neighbourhood Context Encoding
    neighbourhood_context_encoder -- NeighbourhoodContextEncoder: RGCN warm-up
                                     whose OUTPUT is the frozen context table
                                     E' (needs PyG)
    membership_sketch             -- build_membership_sketches: one Bloom
                                     membership sketch per entity over its
                                     1-2 hop neighbour set

  Phase 2 -- Adversarial Generator Training
    candidate_sampler             -- CandidateSampler: per-triple candidate
                                     sets + the logQ correction
    generator                     -- CandidateScoringGenerator (the generator)
                                     + gumbel_softmax
    plausibility_discriminator    -- PlausibilityDiscriminator: "could this
                                     triple be real?"
    neighbourhood_discriminator   -- NeighbourhoodDiscriminator: "does the
                                     filler fit this anchor's neighbourhood?"
    train                         -- the trainer (python -m kgsage.gan.train):
                                     the dual-discriminator game with a PI
                                     controller on alpha, the weight of the
                                     neighbourhood penalty, plus per-epoch
                                     snapshots

  Phase 3 -- Corruption Generation
    (not in this package) `kgsage.corruption_generation` loads a checkpoint and
    emits one corruption per input triple -- the same pipeline `kgsage_bridge`
    uses to feed the downstream detector, where a corruption is called a
    negative during training and an anomaly during evaluation.
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
