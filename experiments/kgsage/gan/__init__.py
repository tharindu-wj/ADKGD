"""KGSAGE adversarial training stack (dual-discriminator, candidate_v2).

Modules:
    train      -- the trainer (python -m kgsage.gan.train): dual-discriminator
                  game, PI-controlled penalty weight, per-epoch snapshots
    generator  -- CandidateScoringGenerator + gumbel_softmax (ST selection)
    d_real     -- realism discriminator (spectral-normed, wrong-anchor class)
    d_match    -- neighbourhood-consistency discriminator (cross-attention)
    encoder    -- RGCN warm-up producing the frozen context table E' (PyG-gated)
    sketch     -- Bloom membership sketches of 1-2 hop neighbour sets
    candidates -- per-triple candidate sampling with the logQ correction

Generation (load checkpoint + produce negatives) lives at `kgsage.inference` --
the same pipeline `kgsage_bridge` uses to feed the downstream detector.
"""
from kgsage.gan.generator import CandidateScoringGenerator, gumbel_softmax
from kgsage.gan.d_real import DReal
from kgsage.gan.d_match import DMatch

__all__ = [
    "CandidateScoringGenerator",
    "gumbel_softmax",
    "DReal",
    "DMatch",
]
