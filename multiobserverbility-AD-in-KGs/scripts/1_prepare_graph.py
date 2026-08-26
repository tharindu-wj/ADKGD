"""Build the graph the observers work on -- with known-false facts planted in it.

    python scripts/1_prepare_graph.py
    python scripts/1_prepare_graph.py --negatives 500 --seed 42

Merges train + valid + test, then plants a sample of the dataset's HAND-VERIFIED
false triples (true-looking, human-checked falsehoods -- see DESIGN.md for
examples) and writes the answer key beside the graph:

    prepared/kg.tsv            what every tool and agent reads
    prepared/ground_truth.tsv  what ONLY the evaluator may read

No synthetic corruption: the planted facts are type-consistent,
human-checked falsehoods, which is what a real dirty graph contains.

THE GUARDS LINE MUST READ ALL ZEROS. A planted "negative" that is actually
in the graph would label a true fact as an anomaly -- punishing a detector
for being right, the quiet way results get ruined.
"""
import sys
from pathlib import Path

# Scripts live in scripts/, so Python puts THAT on sys.path, not the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import random  # noqa: E402

from loaders import graph  # noqa: E402
from loaders.active import DATASET  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--negatives", type=int, default=500,
                    help="how many verified-false triples to plant")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ---- merge the real graph -------------------------------------------------
real = []
seen = set()
for split_file in DATASET.TRIPLE_SPLITS:
    if not split_file.exists():
        raise SystemExit(f"missing {split_file} -- is data/ complete?")
    triples = graph.load_triples(split_file)
    fresh = [t for t in triples if t not in seen]
    seen.update(fresh)
    real.extend(fresh)
    print(f"  {split_file.name}: {len(triples)} triples, {len(fresh)} new")

# ---- plant verified-false triples -----------------------------------------
# The loader says where a dataset keeps its negatives; this script only asks.
# A dataset module without NEGATIVE_SPLITS needs a different contamination
# protocol, and should fail here, loudly, rather than inherit this one.
negative_pool = []
for negative_file in DATASET.NEGATIVE_SPLITS:
    negative_pool.extend(graph.load_triples(negative_file))

rng = random.Random(args.seed)
planted = rng.sample(negative_pool, min(args.negatives, len(negative_pool)))

# The guards. Both must be zero, every run.
actually_true = sum(1 for t in planted if t in seen)
duplicates = len(planted) - len(set(planted))
planted = [t for t in dict.fromkeys(planted) if t not in seen]

# One shuffled graph, so the planted rows are not clustered at the end.
everything = real + planted
rng.shuffle(everything)
labels = {t: (1 if t in set(planted) else 0) for t in everything}

DATASET.KG.parent.mkdir(parents=True, exist_ok=True)
with open(DATASET.KG, "w", encoding="utf-8", newline="\n") as f:
    for head, relation, tail in everything:
        f.write(f"{head}\t{relation}\t{tail}\n")
with open(DATASET.TRUTH, "w", encoding="utf-8", newline="\n") as f:
    for triple in everything:
        head, relation, tail = triple
        kind = "verified_false" if labels[triple] else "real"
        f.write(f"{head}\t{relation}\t{tail}\t{labels[triple]}\t{kind}\n")

entities = {e for h, r, t in everything for e in (h, t)}
relations = {r for h, r, t in everything}
print(f"\nwrote {DATASET.KG.relative_to(ROOT)}: {len(everything)} triples "
      f"({len(planted)} planted false), {len(entities)} entities, "
      f"{len(relations)} relations")
print(f"wrote {DATASET.TRUTH.relative_to(ROOT)} -- the evaluator's file, "
      f"nothing else may read it")
print(f"\nGUARDS  planted-but-actually-true {actually_true}   "
      f"duplicates {duplicates}   (all must be 0)")
