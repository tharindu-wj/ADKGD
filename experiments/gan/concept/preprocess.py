"""CLI orchestrator for Phase 1 - the Concept Module.

Reads a KG's training file, runs the appropriate KB-family adapter,
builds concept pools and cardinality classification, and writes three
artefacts:

  STEP 1: data/<DATASET>/entity_types.tsv
          (NTriples-style user-facing TSV. Reviewers can inspect by eye.)

  STEP 2: data/<DATASET>/entity_types_metadata.json
          (Provenance + statistics, including SHA-256 of source files.)

  STEP 3: experiments/gan/outputs/concept_pools/<DATASET>.pkl
          (Internal cache for Phase 2 and Phase 3. Rebuildable from
           the other two files plus train.txt - safe to gitignore.)

Usage from the repo root:
  python -m experiments.gan.concept.preprocess --dataset FB15K-237 --family freebase
"""
import argparse
import sys
from pathlib import Path

# Project root: 4 levels up from this file (concept/preprocess.py is at
# experiments/gan/concept/preprocess.py; root is the ADKGD repo root).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Make `from experiments.gan...` importable when run as a script.
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.gan.concept.adapters import get_adapter  # noqa: E402
from experiments.gan.concept.concept_pools import build_pools, save_pools  # noqa: E402
from experiments.gan.concept.cardinality import classify_cardinality  # noqa: E402


# Per-family namespace label used in metadata.json's type_vocabulary.
_TYPE_NAMESPACE = {
    "freebase": "Freebase top-level domains + role markers",
    "wordnet": "WordNet lexname classes",
    "yago": "SchemaOrg classes",
}


def main():
    """Run Phase 1 preprocessing end-to-end for one dataset."""
    args = _parse_args()

    project_root = Path(args.project_root).resolve()
    data_dir = project_root / "data" / args.dataset
    train_path = data_dir / "train.txt"
    valid_path = data_dir / "valid.txt"
    test_path = data_dir / "test.txt"

    if not train_path.exists():
        print(f"!! train.txt not found at {train_path}", file=sys.stderr)
        return 1

    print(f"[1/4] Running {args.family} adapter on {train_path} ...", flush=True)
    adapter = get_adapter(args.family)
    entity_to_types = adapter.extract_types(str(train_path))
    print(f"      -> extracted types for {len(entity_to_types):,} entities", flush=True)

    print(f"[2/4] Writing user-facing files to {data_dir} ...", flush=True)
    types_tsv = data_dir / "entity_types.tsv"
    types_meta = data_dir / "entity_types_metadata.json"
    adapter.write_types_file(str(types_tsv), entity_to_types)
    source_kg_files = {"train.txt": str(train_path)}
    if valid_path.exists():
        source_kg_files["valid.txt"] = str(valid_path)
    if test_path.exists():
        source_kg_files["test.txt"] = str(test_path)
    adapter.write_metadata_file(
        str(types_meta),
        dataset_name=args.dataset,
        source_kg_files=source_kg_files,
        entity_to_types=entity_to_types,
        type_namespace=_TYPE_NAMESPACE.get(args.family, "Unknown"),
    )

    print(f"[3/4] Building concept pools + cardinality ...", flush=True)
    pools = build_pools(str(train_path), entity_to_types)
    cardinality, cardinality_stats = classify_cardinality(
        str(train_path), pools["relation_to_id"]
    )
    pools["cardinality"] = cardinality
    pools["cardinality_stats"] = cardinality_stats
    pools["source_files"] = source_kg_files

    print(f"[4/4] Saving concept_pools cache ...", flush=True)
    cache_dir = project_root / "experiments" / "gan" / "outputs" / "concept_pools"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{args.dataset}.pkl"
    save_pools(pools, str(cache_path))

    _print_summary(args.dataset, pools, entity_to_types, cardinality,
                   types_tsv, types_meta, cache_path)
    return 0


def _print_summary(dataset, pools, entity_to_types, cardinality,
                   types_tsv, types_meta, cache_path):
    """Print a compact human-readable report of Phase 1 results."""
    print()
    print("=" * 60)
    print(f"  Phase 1 complete  ({dataset})")
    print("=" * 60)
    print(f"  entities:                 {pools['n_entities']:,}")
    print(f"  relations:                {pools['n_relations']:,}")
    n_types = len({t for ts in entity_to_types.values() for t in ts})
    print(f"  types extracted:          {n_types:,}")
    typed_entities = sum(1 for ts in entity_to_types.values() if ts)
    coverage = typed_entities / max(len(entity_to_types), 1)
    print(f"  entity type coverage:     {coverage * 100:.2f}%")

    pool_sizes = list(pools["pool_sizes"].values())
    if pool_sizes:
        median_head = sorted(s["head"] for s in pool_sizes)[len(pool_sizes) // 2]
        median_tail = sorted(s["tail"] for s in pool_sizes)[len(pool_sizes) // 2]
        print(f"  median headPool size:     {median_head:,}")
        print(f"  median tailPool size:     {median_tail:,}")

    card_dist = {}
    for c in cardinality.values():
        card_dist[c] = card_dist.get(c, 0) + 1
    card_str = ", ".join(f"{k}={v}" for k, v in sorted(card_dist.items()))
    print(f"  cardinality distribution: {card_str}")
    print(f"  dataset_hash:             {pools['dataset_hash'][:16]}...")
    print()
    print(f"  Output files:")
    print(f"    {types_tsv}")
    print(f"    {types_meta}")
    print(f"    {cache_path}")


def _parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", required=True,
                    help="dataset folder name under data/ (e.g., FB15K-237)")
    ap.add_argument("--family", required=True,
                    choices=["freebase", "wordnet", "yago"],
                    help="KB family adapter to use")
    ap.add_argument("--project_root", default=str(_PROJECT_ROOT),
                    help=f"repo root (default: {_PROJECT_ROOT})")
    return ap.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
