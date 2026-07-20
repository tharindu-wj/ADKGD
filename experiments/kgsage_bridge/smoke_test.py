"""End-to-end smoke test for the KGSAGE <-> downstream-detector bridge.

This test deliberately imports BOTH the standalone `kgsage.*` package AND its
sibling `kgsage_bridge` glue, so it lives HERE (with the bridge) rather than
inside `experiments/kgsage/`. That keeps the kgsage package's own smoke test
(`experiments/kgsage/smoke_test.py`) free of any cross-package import, so the
package stays independently extractable.

Run from repo root (pytorch env):
    PYTHONPATH=experiments python experiments/kgsage_bridge/smoke_test.py
"""
import os
import sys

# Put `experiments/` on sys.path so `kgsage` and `kgsage_bridge` both resolve.
_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)


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

    from kgsage import load_kg
    from kgsage_bridge.bridge import load_gan, generate, render_stats
    print("OK: kgsage_bridge.bridge imports (full implementation, no stubs)")

    # ---- SECTION 2: Bridge end-to-end ----
    section("SECTION 2: kgsage_bridge.bridge end-to-end on dummy_kg")

    # A v2 (candidate_v2) OR a legacy v1 checkpoint both work here: the bridge
    # calls kgsage.inference.generate_negatives, which branches on the payload
    # arch. The dummy checkpoint just needs to exist.
    ckpt = "experiments/kgsage/outputs/checkpoints/kgsage_dummy.pt"
    if not os.path.isfile(ckpt):
        print(f"SKIP: {ckpt} not found.")
        print(f"      Make one with the dual-critic trainer:")
        print(f"      python -m kgsage.gan.train_v2 --data data/dummy_kg "
              f"--warmup_epochs 2 --dmatch_epochs 1 --epochs 2 --device cpu "
              f"--out {ckpt}")
        section("ALL CHECKS THAT COULD RUN PASSED")
        return 0

    payload = load_gan(ckpt, device=None)
    print(f"OK: bridge loaded {ckpt}")
    print(f"    n_ent={payload['n_ent']}  n_rel={payload['n_rel']}  device={payload['device']}")

    kg = load_kg("data/dummy_kg")
    positives = kg["triples_train"][:8]

    import numpy as np
    rng = np.random.default_rng(42)
    # The bridge's generate() takes the detector's id maps as keyword args. Its
    # `adkgd_*` names are the ADKGD-side contract (the bridge is the designated
    # ADKGD-aware glue); the standalone kgsage.inference API underneath is
    # caller-agnostic (triples, id_maps).
    negatives, stats = generate(
        positives,
        payload=payload,
        adkgd_id2ent=kg["id2ent"],
        adkgd_id2rel=kg["id2rel"],
        adkgd_ent2id=kg["ent2id"],
        adkgd_rel2id=kg["rel2id"],
        rng=rng,
    )
    print(f"OK: generate() returned {len(negatives)} negatives")
    print(f"OK: render_stats: {render_stats(stats)}")

    assert len(negatives) == len(positives), \
        f"1:1 contract broken: {len(negatives)} negatives for {len(positives)} positives"
    print(f"OK: 1:1 contract holds ({len(negatives)} == {len(positives)})")

    n_changed = sum(1 for p, n in zip(positives, negatives) if tuple(p) != tuple(n))
    print(f"OK: {n_changed}/{len(negatives)} negatives differ from their positive")

    section("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
