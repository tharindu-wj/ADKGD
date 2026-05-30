"""CLI: derive relation-signature pseudo-types and write entity_metadata.txt.

The standard FB15k-237 distribution ships with no entity-type metadata.
This script reads `<dataset_dir>/{train,valid,test}.txt`, clusters entities
by their relation signature, and writes the result to
`<dataset_dir>/entity_metadata.txt` in the 3-column format the existing
loader (`src/kg_data/loader.py`) already reads.

After this runs, `kg_data.loader.load_kg_union` will populate `node_types`
non-trivially, which in turn lets `corruption_strategies.tric` activate
its `swap_subject_same_type` / `swap_object_same_type` operations -
the two corruptions that produce type-coherent-but-wrong training
targets for the GAN.

Usage:
    python scripts/build_pseudo_types.py --data data/FB15K
    python scripts/build_pseudo_types.py --data data/dummy_kg --clusters 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src/` importable as the top-level package root (mirrors how the
# notebook does it - see knowledge_graph_clone.ipynb cell that inserts
# `project_path` into sys.path).
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT  = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_data.loader import load_kg_union                            # noqa: E402
from kg_data.relation_signature_types import (                      # noqa: E402
    derive_pseudo_types,
    write_entity_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", required=True, type=Path,
        help="Dataset directory containing train.txt / valid.txt / test.txt",
    )
    parser.add_argument(
        "--clusters", type=int, default=100,
        help="Number of pseudo-types (KMeans k). Default 100; for the dummy "
             "KG use a small value like 4.",
    )
    parser.add_argument(
        "--svd-dim", type=int, default=32,
        help="Truncated-SVD components before clustering. Default 32.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (passed to TruncatedSVD and MiniBatchKMeans).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output path. Defaults to <data>/entity_metadata.txt.",
    )
    args = parser.parse_args()

    output_path = args.out if args.out is not None else (args.data / "entity_metadata.txt")

    print(f"[build_pseudo_types] loading union KG from {args.data} ...", flush=True)
    kg = load_kg_union(args.data)
    print(
        f"[build_pseudo_types] entities={len(kg.entity_id_to_row):,} "
        f"relations={len(kg.relation_id_to_channel):,} "
        f"union_triples={len(kg.triples_idx):,}",
        flush=True,
    )

    print(f"[build_pseudo_types] clustering into {args.clusters} pseudo-types ...", flush=True)
    pseudo_types = derive_pseudo_types(
        kg,
        num_clusters=args.clusters,
        svd_dim=args.svd_dim,
        random_state=args.seed,
    )

    print(f"[build_pseudo_types] writing {output_path} ...", flush=True)
    write_entity_metadata(output_path, pseudo_types)
    print(f"[build_pseudo_types] done. {len(pseudo_types):,} entries written.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
