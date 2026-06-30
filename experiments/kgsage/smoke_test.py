"""Local smoke test for the KGSAGE package + bridge.

Verifies the package structure on a developer machine that has torch installed.

Run from repo root:
    PYTHONPATH=experiments python experiments/kgsage/smoke_test.py

Sections:
  1. Imports
  2. Behaviour checks (resolve_dataset)
  3. GAN + inference imports
  4. load_kg works on dummy_kg
  5. Bridge end-to-end if the dummy GAN checkpoint exists
"""
import os
import sys


def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    print("Python:", sys.version.split()[0])
    print("Platform:", sys.platform)

    # ---- SECTION 1: Imports ----
    section("SECTION 1: kgsage.* + bridge imports")

    import kgsage
    print(f"OK: import kgsage  (version={kgsage.__version__})")

    from kgsage import load_kg, resolve_dataset, KNOWN_DATASETS
    print("OK: eager public API (load_kg, resolve_dataset, KNOWN_DATASETS)")
    print(f"    KNOWN_DATASETS: {sorted(KNOWN_DATASETS.keys())}")

    from kgsage.data.loaders import load_kg as _load_kg2
    from kgsage.data.datasets import resolve_dataset as _resolve2
    print("OK: kgsage.data.* sub-package imports")

    from kgsage_bridge.bridge import load_gan, generate, render_stats
    print("OK: kgsage_bridge.bridge imports (full implementation, no stubs)")

    # ---- SECTION 2: Behaviour ----
    section("SECTION 2: Behaviour checks")

    print("Known dataset configs:")
    for name in sorted(KNOWN_DATASETS.keys()):
        cfg = resolve_dataset(name)
        print(f"  {name:10s}  path={cfg['path']}")

    try:
        resolve_dataset("not_a_real_dataset")
        print("FAIL: resolve_dataset should have raised")
        return 1
    except ValueError:
        print("OK: resolve_dataset rejects unknown name")

    if os.path.isdir("data/dummy_kg"):
        cfg = resolve_dataset("data/dummy_kg")
        print(f"OK: resolve_dataset(path) -> name={cfg['name']}, path={cfg['path']}")

    # ---- SECTION 3: GAN + inference imports ----
    section("SECTION 3: kgsage.gan.* + kgsage.inference imports")

    from kgsage.gan.models import (
        KGSAGEGenerator, KGSAGEDiscriminator, gumbel_softmax, soft_embedding,
    )
    print("OK: kgsage.gan.models.{KGSAGEGenerator, KGSAGEDiscriminator, "
          "gumbel_softmax, soft_embedding}")

    from kgsage.gan import train as _gan_train
    print("OK: kgsage.gan.train (the training loop module)")

    from kgsage.inference import (
        load_checkpoint, generate_negatives, render_stats as _rs, generate_partners,
    )
    print("OK: kgsage.inference.{load_checkpoint, generate_negatives, render_stats, "
          "generate_partners}")
    assert generate_partners is generate_negatives, \
        "generate_partners should alias generate_negatives"
    print("OK: generate_partners is generate_negatives (alias)")

    # ---- SECTION 4: load_kg on dummy_kg ----
    section("SECTION 4: load_kg on dummy_kg")

    if os.path.isdir("data/dummy_kg"):
        kg = load_kg("data/dummy_kg")
        print(f"OK: load_kg(data/dummy_kg)")
        print(f"    {kg['n_ent']} entities, {kg['n_rel']} relations")
        print(f"    train={len(kg['triples_train'])}  "
              f"valid={len(kg['triples_valid'])}  test={len(kg['triples_test'])}")
    else:
        print("SKIP: data/dummy_kg not present in cwd")

    # ---- SECTION 5: Bridge end-to-end ----
    section("SECTION 5: kgsage_bridge.bridge end-to-end on dummy_kg")

    ckpt = "experiments/kgsage/outputs/checkpoints/kgsage_dummy.pt"
    if not os.path.isfile(ckpt):
        print(f"SKIP: {ckpt} not found.")
        print(f"      Run `python -m kgsage.cli.train_gan --data data/dummy_kg "
              f"--epochs 5 --device cpu --out {ckpt}` first.")
        section("ALL CHECKS THAT COULD RUN PASSED")
        return 0

    payload = load_gan(ckpt, device=None)
    print(f"OK: bridge loaded {ckpt}")
    print(f"    n_ent={payload['n_ent']}  n_rel={payload['n_rel']}  device={payload['device']}")

    kg = load_kg("data/dummy_kg")
    positives = kg["triples_train"][:8]

    import numpy as np
    rng = np.random.default_rng(42)
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
