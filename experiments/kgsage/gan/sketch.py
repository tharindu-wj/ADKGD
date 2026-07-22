"""Bloom-style neighbourhood sketches: a fixed-width, set-readable encoding of
each entity's 1-2 hop neighbourhood.

WHY THIS EXISTS
    Wagstaff et al. (ICML 2019) prove a pooled continuous embedding of dimension
    d cannot universally represent set functions over sets larger than d. The
    64-d context table E' therefore CANNOT support membership-style queries
    ("is candidate x connected to anchor h's world?") for realistic
    neighbourhood sizes -- measured directly in this project by the failure of
    every support-penalty arm (see docs/KGSAGE_learned_contradiction_design.md).
    The sketch is the cheap escape: an m-bit hashed indicator of
    N1(h) ∪ N2(h). A dot-product / linear read of two sketches approximates set
    intersection, so a generator or discriminator conditioned on it can LEARN
    corroboration-style signals that are unlearnable from pooled E'.

DESIGN
    - k independent hash functions (salted multiply-shift over 64-bit mixes),
      m bits, uint8 storage: [n_ent, m] tensor (FB15K-237 at m=8192: ~119 MB;
      WN18RR: ~335 MB) -- held on CPU, rows gathered per batch.
    - LOAD FACTOR IS THE DESIGN CONSTRAINT (measured, not guessed): the first
      build at m=4096/n2_cap=4096 saturated (median density 0.81, alien
      false-"in" rate 68% -- Bloom FPR (1-e^{-kn/m})^k with kn/m~2). Defaults
      are now m=8192, n2_cap=1024 (load ~0.27, predicted FPR ~5%): the sketch
      trades N2 coverage on hubs for a readable signal. The exact support mask
      at decode remains the guarantee; the sketch is conditioning signal only.
    - The sketch is NOT a guarantee mechanism (false positives exist by
      construction). Guarantees stay with the exact support mask at decode.
"""

from __future__ import annotations

import numpy as np
import torch

_MIX = 0x9E3779B97F4A7C15          # golden-ratio odd constant (splitmix64 mix)


def _hash_bits(ids: np.ndarray, salt: int, m: int) -> np.ndarray:
    """Vectorised multiply-shift hash of int ids -> bit positions in [0, m)."""
    x = (ids.astype(np.uint64) + np.uint64(salt)) * np.uint64(_MIX)
    x ^= x >> np.uint64(31)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    return (x % np.uint64(m)).astype(np.int64)


def build_sketches(triples, n_ent: int, m: int = 8192, k_hash: int = 2,
                   n2_cap: int = 1024, seed: int = 0,
                   include_two_hop: bool = True) -> torch.Tensor:
    """[n_ent, m] uint8 Bloom sketches of N1 (∪ capped N2) per entity.

    triples: iterable of (h, r, t) int triples (undirected adjacency is built).
    Deterministic given seed (N2 capping uses a seeded RNG).
    """
    rng = np.random.default_rng(seed)
    nbrs: list[set] = [set() for _ in range(n_ent)]
    for h, _, t in triples:
        nbrs[h].add(t)
        nbrs[t].add(h)

    sk = np.zeros((n_ent, m), dtype=np.uint8)
    for e in range(n_ent):
        members = nbrs[e]
        if include_two_hop and members:
            two: set = set()
            for n1 in members:
                two |= nbrs[n1]
            two.discard(e)
            if len(two) > n2_cap:
                two = set(rng.choice(np.fromiter(two, dtype=np.int64),
                                     size=n2_cap, replace=False).tolist())
            members = members | two
        if not members:
            continue
        ids = np.fromiter(members, dtype=np.int64)
        for j in range(k_hash):
            sk[e, _hash_bits(ids, salt=seed * 1000 + j, m=m)] = 1
    return torch.from_numpy(sk)


def sketch_stats(sk: torch.Tensor, sample: int = 2000, seed: int = 0) -> dict:
    """Diagnostics: bit-density distribution + saturation rate."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(sk.shape[0], generator=g)[:sample]
    dens = sk[idx].float().mean(dim=1)
    return {
        "m": sk.shape[1],
        "density_mean": float(dens.mean()),
        "density_p50": float(dens.median()),
        "density_p95": float(dens.quantile(0.95)),
        "saturated_frac(>0.9)": float((dens > 0.9).float().mean()),
        "empty_frac": float((dens == 0).float().mean()),
    }
