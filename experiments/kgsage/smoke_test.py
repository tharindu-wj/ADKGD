"""Local smoke test for the KGSAGE package + bridge.

Verifies the package structure on a developer machine that has torch installed
(PyG is HPC-only here — encoder access is lazy).

Run from repo root:
    PYTHONPATH=experiments python experiments/kgsage/smoke_test.py

Sections:
  1. Imports that don't require torch_geometric
  2. Behaviour checks (resolve_dataset)
  3. Lazy encoder access — verifies PyG import is deferred
  4. GAN imports (require torch but not PyG)
  5. load_kg works on dummy_kg
  6. Bridge end-to-end if the dummy KGSAGE checkpoint exists
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
    from kgsage.data import audit_dataset as _audit_dataset
    print("OK: kgsage.data.* sub-package imports")

    from kgsage.cli import audit_dataset as _cli_audit_dataset
    print("OK: kgsage.cli.audit_dataset shim imports (no torch needed)")

    from kgsage_bridge.bridge import load_gan, generate, render_stats
    print("OK: kgsage_bridge.bridge imports (full implementation, no stubs)")

    # ---- SECTION 2: Behaviour ----
    section("SECTION 2: Behaviour checks")

    print("Known dataset configs:")
    for name in sorted(KNOWN_DATASETS.keys()):
        cfg = resolve_dataset(name)
        print(f"  {name:10s}  path={cfg['path']:20s}  epochs={cfg['epochs']:>4}  "
              f"dim={cfg['dim']}  bases={cfg['num_bases']}  "
              f"expected_mrr={cfg['expected_mrr']}")

    try:
        resolve_dataset("not_a_real_dataset")
        print("FAIL: resolve_dataset should have raised")
        return 1
    except ValueError:
        print("OK: resolve_dataset rejects unknown name")

    if os.path.isdir("data/dummy_kg"):
        cfg = resolve_dataset("data/dummy_kg")
        print(f"OK: resolve_dataset(path) -> name={cfg['name']}, path={cfg['path']}")

    # ---- SECTION 3: Lazy encoder ----
    section("SECTION 3: Lazy encoder access")

    try:
        _ = kgsage.KGSAGEEncoder
        print("UNEXPECTED: encoder accessed without error")
        print("            (torch_geometric must be installed locally — that's fine)")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"OK: encoder lazy-load defers PyG import until access ({type(e).__name__})")
    except AttributeError as e:
        print(f"FAIL: lazy load mechanism broken: {e}")
        return 1

    # ---- SECTION 4: GAN imports (torch but no PyG required) ----
    section("SECTION 4: kgsage.gan.* + kgsage.inference imports")

    from kgsage.gan.models import (
        KGSAGEGenerator, KGSAGEDiscriminator, load_encoder_embeddings, gumbel_softmax,
    )
    print("OK: kgsage.gan.models.{KGSAGEGenerator, KGSAGEDiscriminator, "
          "load_encoder_embeddings, gumbel_softmax}")

    from kgsage.gan import train as _gan_train
    from kgsage.gan import mine_partner_templates
    print("OK: kgsage.gan.train (the training loop) + mine_partner_templates")

    from kgsage.inference import (
        load_kgsage_checkpoint, generate_kgsage_partners,
        generate_partners, render_partner_stats,
    )
    print("OK: kgsage.inference.{load_kgsage_checkpoint, generate_kgsage_partners, "
          "generate_partners, render_partner_stats}")
    assert generate_partners is generate_kgsage_partners, \
        "generate_partners should alias generate_kgsage_partners"
    print("OK: generate_partners is generate_kgsage_partners (alias)")

    # ---- SECTION 5: load_kg on dummy_kg ----
    section("SECTION 5: load_kg on dummy_kg")

    if os.path.isdir("data/dummy_kg"):
        kg = load_kg("data/dummy_kg")
        print(f"OK: load_kg(data/dummy_kg)")
        print(f"    {kg['n_ent']} entities, {kg['n_rel']} relations")
        print(f"    train={len(kg['triples_train'])}  "
              f"valid={len(kg['triples_valid'])}  test={len(kg['triples_test'])}")
        print(f"    edge_index.shape={tuple(kg['edge_index'].shape)}")
    else:
        print("SKIP: data/dummy_kg not present in cwd")

    # ---- SECTION 6: Bridge end-to-end ----
    section("SECTION 6: kgsage_bridge.bridge end-to-end on dummy_kg")

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

    # role-swap shape: for a non-self-loop positive (h,r,t), the negative is (t,r',h)
    (ph, pr, pt), (nh, nr, nt) = positives[0], negatives[0]
    if ph != pt:
        assert nh == pt and nt == ph, "negative is not a role-swap of the positive"
        print(f"OK: role-swap shape — (h,r,t) -> (t,r',h)")

    section("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
