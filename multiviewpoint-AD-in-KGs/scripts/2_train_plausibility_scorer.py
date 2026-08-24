"""Train a knowledge-graph embedding model on the CONTAMINATED graph.

Reads   data/<name>/contaminated_kg.tsv
Writes  models/<name>/<model>/            one folder per architecture

Training on the errors is deliberate. It is what ADKGD does, and it is what
real auditing looks like: you have one dirty graph, not a clean one. Train on
clean data and inject afterwards and you measure memorisation instead -- every
real triple was seen, every fake was not, and a detector then separates seen
from unseen rather than true from false.

    python scripts/2_train_plausibility_scorer.py
    python scripts/2_train_plausibility_scorer.py --model ComplEx --epochs 2000
"""
import os
import sys
from pathlib import Path

# Scripts live in scripts/, so Python puts THAT on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows only: this machine crashes inside MKL/oneDNN without these, but on a
# Linux cluster pinning to one thread would cripple a CPU run for no reason.
if sys.platform == "win32":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse

import torch
from pykeen.pipeline import pipeline
from pykeen.predict import predict_target, predict_triples
from pykeen.triples import TriplesFactory

from loaders.active import DATASET

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="DistMult",
                choices=["DistMult", "TransE", "ComplEx", "RotatE"])
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--epochs", type=int, default=1000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--device", default="auto", help="auto, cpu, or cuda")
args = ap.parse_args()

if not DATASET.KG.exists():
    raise SystemExit(f"missing {DATASET.KG}. Run scripts/1_inject_anomalies.py first.")

device = args.device
if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
if device.startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit("--device cuda asked for, but torch reports no CUDA device")

save_to = DATASET.MODELS / args.model.lower()

# ---- load -----------------------------------------------------------------
n_lines = sum(1 for _ in open(DATASET.KG, encoding="utf-8"))
tf = TriplesFactory.from_path(str(DATASET.KG))
print(f"{DATASET.KG.name}: {n_lines} lines -> {tf.num_triples} triples, "
      f"{tf.num_entities} entities, {tf.num_relations} relations")
if tf.num_triples != n_lines:
    print(f"  NOTE: from_path deduplicated {n_lines - tf.num_triples} rows")

print(f"\ntraining {args.model} dim={args.dim} epochs={args.epochs} "
      f"seed={args.seed} device={device}")
print("the injected anomalies ARE in this training set, by design")

# ---- train ----------------------------------------------------------------
# pipeline() requires a testing factory, so it gets the training one. The
# metrics it emits are therefore TRAIN-SET numbers and must never be quoted as
# evaluation -- the real evaluation happens in the detectors.
result = pipeline(
    training=tf, testing=tf,
    model=args.model,
    model_kwargs=dict(embedding_dim=args.dim),
    training_kwargs=dict(num_epochs=args.epochs, use_tqdm=False),
    evaluation_kwargs=dict(use_tqdm=False),
    random_seed=args.seed,
    device=device,
)
result.save_to_directory(str(save_to))
print(f"\nsaved to {save_to}")

# ---- sanity: is the model degenerate? -------------------------------------
scores = predict_triples(model=result.model, triples=tf).process(factory=tf).df["score"]
print(f"\nscore spread over all {len(scores)} triples:")
print(f"  min {scores.min():.3f}   median {scores.median():.3f}   "
      f"max {scores.max():.3f}   std {scores.std():.4f}")
if scores.std() < 1e-4:
    print("  COLLAPSED: every triple scores the same. Nothing downstream can work.")

# A few real queries, top 3 answers each. Heads are taken from the data rather
# than hardcoded, so this works on any dataset. Read them: an obviously wrong
# kind of entity in the top 3 means the model has not learned the slot's type.
id2e = {v: k for k, v in tf.entity_to_id.items()}
id2r = {v: k for k, v in tf.relation_to_id.items()}
seen, probes = set(), []
for h, r, _ in tf.mapped_triples.tolist():
    if (h, r) in seen:
        continue
    seen.add((h, r))
    probes.append((id2e[h], id2r[r]))
    if len(probes) == 4:
        break

for head, rel in probes:
    top = predict_target(model=result.model, head=head, relation=rel,
                         triples_factory=tf).df.head(3)
    print(f"\n({head}, {rel}, ?)")
    for _, row in top.iterrows():
        print(f"   {row['tail_label']:<26} {row['score']:.3f}")
