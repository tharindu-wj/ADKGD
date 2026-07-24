"""Bloom membership sketches (paper: Neighbourhood Context Encoding).

Each entity gets a fixed-width bit vector (a Bloom filter) marking which
entities sit in its 1–2 hop neighbourhood. This is the generator's
set-readable view of "who belongs to this entity's world".

Why it exists: the pooled 64-d context vector E'[e] provably cannot answer
membership questions over neighbour sets larger than 64 (Wagstaff et al.,
ICML 2019). The sketch is the cheap fix — an m-bit hashed indicator of the
neighbour set. A linear read of a sketch approximates set intersection, so a
model conditioned on it CAN learn corroboration-style signals that a pooled
vector cannot carry.

Two practical points:
  - The load factor matters. At m=4096 the sketch saturated (median bit
    density 0.81) and became unreadable; the defaults m=8192 with the 2-hop
    set capped at 1024 keep density ~0.27 (false-positive rate ~5%).
  - A Bloom filter has false positives BY DESIGN, so the sketch is only a
    conditioning signal. The hard falseness guarantee stays where it always
    was: the exact masks applied at corruption time.
"""

from __future__ import annotations

import numpy as np
import torch

_MIX = 0x9E3779B97F4A7C15          # golden-ratio odd constant (splitmix64 mix)


def _hash_to_bit_positions(member_ids: np.ndarray, salt: int, m: int) -> np.ndarray:
    """Vectorised multiply-shift hash: entity ids -> bit positions in [0, m)."""
    x = (member_ids.astype(np.uint64) + np.uint64(salt)) * np.uint64(_MIX)
    x ^= x >> np.uint64(31)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    return (x % np.uint64(m)).astype(np.int64)


def build_membership_sketches(triples, n_ent: int, m: int = 8192,
                              k_hash: int = 2, n2_cap: int = 1024,
                              seed: int = 0,
                              include_two_hop: bool = True) -> torch.Tensor:
    """Build one m-bit sketch per entity; returns a uint8 tensor [n_ent, m].

    triples : iterable of (head, relation, tail) integer triples. Adjacency is
              treated as undirected.
    n2_cap  : if an entity's 2-hop set is larger than this, a random subset of
              this size is used (keeps hub sketches readable).
    Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)

    # Undirected 1-hop neighbour set per entity.
    neighbour_sets: list[set] = [set() for _ in range(n_ent)]
    for head, _, tail in triples:
        neighbour_sets[head].add(tail)
        neighbour_sets[tail].add(head)

    sketch_matrix = np.zeros((n_ent, m), dtype=np.uint8)
    for entity in range(n_ent):
        members = neighbour_sets[entity]
        if include_two_hop and members:
            two_hop: set = set()
            for neighbour in members:
                two_hop |= neighbour_sets[neighbour]
            two_hop.discard(entity)
            if len(two_hop) > n2_cap:
                two_hop = set(rng.choice(np.fromiter(two_hop, dtype=np.int64),
                                         size=n2_cap, replace=False).tolist())
            members = members | two_hop
        if not members:
            continue
        member_ids = np.fromiter(members, dtype=np.int64)
        for hash_index in range(k_hash):
            positions = _hash_to_bit_positions(
                member_ids, salt=seed * 1000 + hash_index, m=m)
            sketch_matrix[entity, positions] = 1
    return torch.from_numpy(sketch_matrix)


def sketch_stats(sketches: torch.Tensor, sample: int = 2000,
                 seed: int = 0) -> dict:
    """Diagnostics: bit-density distribution + how many sketches saturated."""
    generator = torch.Generator().manual_seed(seed)
    sample_ids = torch.randperm(sketches.shape[0], generator=generator)[:sample]
    density = sketches[sample_ids].float().mean(dim=1)
    return {
        "m": sketches.shape[1],
        "density_mean": float(density.mean()),
        "density_p50": float(density.median()),
        "density_p95": float(density.quantile(0.95)),
        "saturated_frac(>0.9)": float((density > 0.9).float().mean()),
        "empty_frac": float((density == 0).float().mean()),
    }
