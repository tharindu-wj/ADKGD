"""Train the link predictor and score EVERY triple, once, offline.

    python scripts/2_train_scorer.py
    python scripts/2_train_scorer.py --model ComplEx --epochs 200

Trains on the CONTAMINATED graph deliberately -- planted falsehoods included,
as positives. That is the real setting (you have one dirty
graph, not a clean one), and training on clean data then scoring planted
rows would measure memorisation, not implausibility.

Writes three things to prepared/:
    model/           the trained model, for provenance
    scores.npy       one score per kg.tsv row, SAME ORDER
    scores_manifest.json   binds the scores to this exact graph by hash

The manifest is the stale-model guard: unlikely_facts refuses scores
computed for a different graph, because a stale file silently scores triples
that no longer exist -- measured in the predecessor as a fake 100%-recall
result that looked like a triumph.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# torch aborts inside oneDNN on this machine without these -- set BEFORE import
if sys.platform == "win32":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from loaders import graph  # noqa: E402
from loaders.active import DATASET  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="DistMult")
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if not DATASET.KG.exists():
    raise SystemExit(f"missing {DATASET.KG}. Run scripts/1_prepare_graph.py first.")

from pykeen.pipeline import pipeline  # noqa: E402
from pykeen.predict import predict_triples  # noqa: E402
from pykeen.triples import TriplesFactory  # noqa: E402

triples = graph.load_triples(DATASET.KG)
factory = TriplesFactory.from_path(DATASET.KG)

print(f"training {args.model} (dim={args.dim}, epochs={args.epochs}) "
      f"on {factory.num_triples} triples, planted falsehoods included")

result = pipeline(
    training=factory,
    testing=factory,          # we never use the eval metrics; training is the point
    model=args.model,
    model_kwargs=dict(embedding_dim=args.dim),
    training_kwargs=dict(num_epochs=args.epochs),
    random_seed=args.seed,
    device="cpu",
)

DATASET.MODEL_DIR.mkdir(parents=True, exist_ok=True)
result.save_to_directory(DATASET.MODEL_DIR)

# ---- score every triple, in kg.tsv order ----------------------------------
# The factory REORDERS triples, so scores are joined back by label, never by
# position -- the predecessor measured 1272 of 1273 rows moving.
predictions = predict_triples(model=result.model, triples=factory)
frame = predictions.process(factory=factory).df
score_of = {(h, r, t): s for h, r, t, s in zip(
    frame["head_label"], frame["relation_label"],
    frame["tail_label"], frame["score"])}

missing = [t for t in triples if t not in score_of]
if missing:
    raise SystemExit(f"{len(missing)} triples got no score -- label join broke.")
scores = np.array([score_of[t] for t in triples], dtype=np.float32)

np.save(DATASET.SCORES, scores)
DATASET.SCORES_MANIFEST.write_text(json.dumps({
    "kg_sha256": hashlib.sha256(DATASET.KG.read_bytes()).hexdigest(),
    "model": args.model,
    "dim": args.dim,
    "epochs": args.epochs,
    "seed": args.seed,
    "triples_scored": len(scores),
}, indent=2), encoding="utf-8")

print(f"\nwrote {DATASET.SCORES.relative_to(ROOT)}: {len(scores)} scores")
print(f"  spread: min {scores.min():.3f}  median {np.median(scores):.3f}  "
      f"max {scores.max():.3f}  std {scores.std():.4f}")
if scores.std() < 1e-4:
    print("  COLLAPSED: every triple scores the same. Nothing downstream can work.")
