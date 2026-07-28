"""End-to-end smoke test for the KGSAGE <-> downstream-detector bridge.

This test deliberately imports BOTH the standalone `kgsage.*` package AND its
sibling `kgsage_bridge` glue, so it lives HERE (with the bridge) rather than
inside the KGSAGE package. That keeps the kgsage repo's own smoke test
(`<kgsage>/smoke_test.py`) free of any cross-package import, so the
package stays independently extractable.

Because it runs on the detector's side of the seam, this file speaks ADKGD's
vocabulary: the triples kgsage calls corruptions are "negatives" here.

Run from repo root (pytorch env):
    PYTHONPATH=experiments python experiments/kgsage_bridge/smoke_test.py
"""
import os
import sys

# Put `experiments/` on sys.path so `kgsage_bridge` resolves. `kgsage` itself
# is pip-installed from its own repo and needs no path help.
_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Appended, not inserted: nothing here should outrank site-packages.
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.append(_EXPERIMENTS_DIR)


def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    print("Python:", sys.version.split()[0])
    print("Platform:", sys.platform)

    # ---- SECTION 1: Imports (both packages) ----
    section("SECTION 1: kgsage + kgsage_bridge imports")

    # Bridge FIRST: it carries the namespace-shadowing guard, so a stale
    # `experiments/kgsage/` directory is reported by name rather than surfacing
    # as a cryptic "cannot import name ... (unknown location)".
    from kgsage_bridge.bridge import load_gan, generate, render_stats
    import kgsage  # noqa: F401  (proves the package resolves)
    print("OK: kgsage_bridge.bridge imports (full implementation, no stubs)")

    # ---- SECTION 2: Bridge end-to-end ----
    section("SECTION 2: kgsage_bridge.bridge end-to-end on a real checkpoint")

    # The bridge calls kgsage.corruption_generation.generate_negatives, which
    # requires a candidate_v2 checkpoint ("candidate_v2" is the frozen arch
    # string for the dual-discriminator architecture). Any real checkpoint works
    # -- the fixture below is derived from whatever this one contains.
    ckpt = os.environ.get("KGSAGE_CKPT", "artifacts/kgsage/generator_wn18rr.pt")
    if not os.path.isfile(ckpt):
        print(f"SKIP: {ckpt} not found.")
        print(f"      Checkpoints are produced by the separate KGSAGE repo and")
        print(f"      copied into artifacts/kgsage/ -- see its README.")
        print(f"      Override the path with KGSAGE_CKPT=<file.pt>.")
        section("ALL CHECKS THAT COULD RUN PASSED")
        return 0

    payload = load_gan(ckpt, device=None)
    print(f"OK: bridge loaded {ckpt}")
    print(f"    n_ent={payload['n_ent']}  n_rel={payload['n_rel']}  device={payload['device']}")

    # Build the test case FROM THE CHECKPOINT rather than a fixed fixture, so this
    # runs against whatever generator is in artifacts/kgsage/.
    #
    # The ADKGD ids are deliberately OFFSET from the generator's. Translating
    # between the two id spaces is the bridge's entire job, and passing the
    # generator's own maps back in would make that translation an identity and
    # test nothing.
    OFFSET = 1000
    real = sorted(payload["real_triple_set"])[:8]
    if not real:
        print("SKIP: checkpoint carries no real triples to corrupt.")
        section("ALL CHECKS THAT COULD RUN PASSED")
        return 0

    # The FULL vocabulary, not just the entities in these 8 triples: a corruption
    # draws its replacement from the generator's whole entity set, and the bridge
    # has to translate that back into an ADKGD id. A partial map would KeyError on
    # the first replacement drawn from outside it -- and in the real pipeline both
    # sides load the same dataset, so both vocabularies are complete.
    gen2adkgd_ent = {e: i + OFFSET for i, e in enumerate(sorted(payload["id2ent"]))}
    gen2adkgd_rel = {r: i + OFFSET for i, r in enumerate(sorted(payload["id2rel"]))}
    adkgd_id2ent = {a: payload["id2ent"][g] for g, a in gen2adkgd_ent.items()}
    adkgd_id2rel = {a: payload["id2rel"][g] for g, a in gen2adkgd_rel.items()}
    adkgd_ent2id = {v: k for k, v in adkgd_id2ent.items()}
    adkgd_rel2id = {v: k for k, v in adkgd_id2rel.items()}
    positives = [(gen2adkgd_ent[h], gen2adkgd_rel[r], gen2adkgd_ent[t])
                 for h, r, t in real]
    print(f"OK: built {len(positives)} positives in an ADKGD id space "
          f"offset by {OFFSET}")

    import numpy as np
    rng = np.random.default_rng(42)
    # The bridge's generate() takes the detector's id maps as keyword args. Its
    # `adkgd_*` names are the ADKGD-side contract (the bridge is the designated
    # ADKGD-aware glue); the standalone kgsage.corruption_generation API
    # underneath is caller-agnostic (triples, id_maps).
    negatives, stats = generate(
        positives,
        payload=payload,
        adkgd_id2ent=adkgd_id2ent,
        adkgd_id2rel=adkgd_id2rel,
        adkgd_ent2id=adkgd_ent2id,
        adkgd_rel2id=adkgd_rel2id,
        rng=rng,
    )
    print(f"OK: generate() returned {len(negatives)} negatives")
    print(f"OK: render_stats: {render_stats(stats)}")

    assert len(negatives) == len(positives), \
        f"1:1 contract broken: {len(negatives)} negatives for {len(positives)} positives"
    print(f"OK: 1:1 contract holds ({len(negatives)} == {len(positives)})")

    # A row that did NOT change is a null corruption (generation failed and the
    # original triple came back); those are the rows in stats["null_indices"].
    n_changed = sum(1 for p, n in zip(positives, negatives) if tuple(p) != tuple(n))
    print(f"OK: {n_changed}/{len(negatives)} negatives differ from their positive")

    # The point of the offset id space: every returned id must be a valid ADKGD
    # id, which is only true if the bridge translated out of the generator's
    # space on the way back.
    for h, r, t in negatives:
        assert h in adkgd_id2ent, f"head {h} is not an ADKGD entity id"
        assert t in adkgd_id2ent, f"tail {t} is not an ADKGD entity id"
        assert r in adkgd_id2rel, f"relation {r} is not an ADKGD relation id"
    print(f"OK: all {len(negatives)} negatives are in the ADKGD id space "
          f"(>= {OFFSET}), so the id translation round-trips")

    # Exactly one ENTITY slot moves; the relation never does.
    for (ph, pr, pt), (nh, nr, nt) in zip(positives, negatives):
        assert pr == nr, f"relation changed: {pr} -> {nr}"
        assert (ph != nh) + (pt != nt) <= 1, "more than one entity slot changed"
    print("OK: single-slot contract holds (relation never corrupted)")

    section("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
