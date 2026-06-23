"""Local smoke test for the KGSAGE package refactor.

Verifies that the package structure is correct on a developer machine that
has torch installed but NOT torch_geometric (which is HPC-only here).

Run from repo root:
    PYTHONPATH=experiments python experiments/kgsage/smoke_test.py

Sections:
  1. Imports that don't require torch_geometric
  2. Behaviour checks (resolve_dataset, bridge stub)
  3. Lazy encoder access — verifies PyG import is deferred
  4. load_kg works on dummy_kg
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

    # ─── SECTION 1: Imports ────────────────────────────────────────────
    section("SECTION 1: Imports that don't need torch_geometric")

    import kgsage
    print(f"OK: import kgsage  (version={kgsage.__version__})")

    from kgsage import load_kg, resolve_dataset, KNOWN_DATASETS
    print("OK: eager public API symbols (load_kg, resolve_dataset, KNOWN_DATASETS)")
    print(f"    KNOWN_DATASETS keys: {sorted(KNOWN_DATASETS.keys())}")

    from kgsage.data.loaders import load_kg as _load_kg2
    from kgsage.data.datasets import resolve_dataset as _resolve2
    from kgsage.data import audit_dataset as _audit_dataset
    print("OK: kgsage.data.* sub-package imports")

    # audit_dataset CLI shim doesn't touch encoder → no PyG needed
    from kgsage.cli import audit_dataset as _cli_audit_dataset
    print("OK: kgsage.cli.audit_dataset shim imports (PyG-free)")

    from kgsage_bridge.bridge import load_gan, generate, render_stats
    print("OK: kgsage_bridge.bridge imports")

    # ─── SECTION 2: Behaviour ──────────────────────────────────────────
    section("SECTION 2: Behaviour checks")

    print("Known dataset configs:")
    for name in sorted(KNOWN_DATASETS.keys()):
        cfg = resolve_dataset(name)
        print(f"  {name:10s}  path={cfg['path']:20s}  epochs={cfg['epochs']:>4}  "
              f"dim={cfg['dim']}  bases={cfg['num_bases']}  "
              f"expected_mrr={cfg['expected_mrr']}")

    # resolve_dataset rejects unknown name
    try:
        resolve_dataset("not_a_real_dataset")
        print("FAIL: resolve_dataset should have raised")
        return 1
    except ValueError:
        print("OK: resolve_dataset rejects unknown name")

    # resolve_dataset accepts custom paths
    if os.path.isdir("data/dummy_kg"):
        cfg = resolve_dataset("data/dummy_kg")
        print(f"OK: resolve_dataset(path) -> name={cfg['name']}, "
              f"path={cfg['path']}, expected_mrr={cfg['expected_mrr']}")

    # bridge stub raises NotImplementedError
    try:
        load_gan("dummy_ckpt")
        print("FAIL: load_gan should have raised NotImplementedError")
        return 1
    except NotImplementedError:
        print("OK: load_gan() raises NotImplementedError as expected")

    # ─── SECTION 3: Lazy encoder ───────────────────────────────────────
    section("SECTION 3: Lazy encoder access")

    try:
        _ = kgsage.KGSAGEEncoder
        print("UNEXPECTED: encoder accessed without error")
        print("            (torch_geometric must be installed locally — that's fine)")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"OK: encoder lazy-load defers PyG import until access ({type(e).__name__})")
        print(f"    Got: {e}")
    except AttributeError as e:
        print(f"FAIL: lazy load mechanism broken: {e}")
        return 1

    # ─── SECTION 4: load_kg ────────────────────────────────────────────
    section("SECTION 4: load_kg on dummy_kg")

    if os.path.isdir("data/dummy_kg"):
        kg = load_kg("data/dummy_kg")
        print(f"OK: load_kg(data/dummy_kg)")
        print(f"    {kg['n_ent']} entities, {kg['n_rel']} relations")
        print(f"    train={len(kg['triples_train'])}  "
              f"valid={len(kg['triples_valid'])}  test={len(kg['triples_test'])}")
        print(f"    edge_index.shape={tuple(kg['edge_index'].shape)}  "
              f"edge_type.shape={tuple(kg['edge_type'].shape)}")
    else:
        print("SKIP: data/dummy_kg not present in cwd")

    # ─── DONE ──────────────────────────────────────────────────────────
    section("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
