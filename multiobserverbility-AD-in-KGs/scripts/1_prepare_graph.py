"""Build the graph the audit runs on: train + valid + test, one file.

    python scripts/1_prepare_graph.py

Milestone 1 keeps this deliberately plain -- a merge, nothing more. The
contamination step (mixing in CoDEx's verified-false triples and writing an
answer key) replaces this file's output later; every tool already reads only
the prepared file, so that swap will not touch them.
"""
import sys
from pathlib import Path

# Scripts live in scripts/, so Python puts THAT on sys.path, not the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loaders import graph  # noqa: E402
from loaders.active import DATASET  # noqa: E402

merged = []
seen = set()
for split_file in DATASET.TRIPLE_SPLITS:
    if not split_file.exists():
        raise SystemExit(f"missing {split_file} -- is data/ complete?")
    triples = graph.load_triples(split_file)
    fresh = [t for t in triples if t not in seen]
    seen.update(fresh)
    merged.extend(fresh)
    print(f"  {split_file.name}: {len(triples)} triples, {len(fresh)} new")

DATASET.KG.parent.mkdir(parents=True, exist_ok=True)
with open(DATASET.KG, "w", encoding="utf-8", newline="\n") as f:
    for head, relation, tail in merged:
        f.write(f"{head}\t{relation}\t{tail}\n")

entities = {e for h, r, t in merged for e in (h, t)}
relations = {r for h, r, t in merged}
print(f"\nwrote {DATASET.KG.relative_to(ROOT)}: "
      f"{len(merged)} triples, {len(entities)} entities, {len(relations)} relations")
