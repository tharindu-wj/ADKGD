"""Step 2: train a KG embedding model on the CONTAMINATED graph.

Training on the errors is deliberate. It is what ADKGD does (dataset.py merges
every split, injects anomalies, and puts them in the POSITIVE half of the loss),
and it is what real auditing looks like: you have one dirty KG, not a clean one.

Train on clean data and inject afterwards and you measure memorisation instead --
every real triple was seen, every fake was not, and the detector separates
seen from unseen rather than true from false.
"""
import os
import sys

# Windows only: this machine crashes inside MKL/oneDNN without these, but on a
# Linux cluster pinning to one thread would cripple a CPU run for no reason.
if sys.platform == "win32":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import torch
from pykeen.pipeline import pipeline
from pykeen.predict import predict_target, predict_triples
from pykeen.triples import TriplesFactory

HERE = Path(__file__).resolve().parent
KG = HERE / "data" / "contaminated_kg.tsv"
SAVE_TO = HERE / "model"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="DistMult",
                choices=["DistMult", "TransE", "ComplEx", "RotatE"])
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--epochs", type=int, default=1000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--device", default="auto", help="auto, cpu, or cuda")
args = ap.parse_args()

device = args.device
if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
if device.startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit("--device cuda asked for, but torch reports no CUDA device")

if not KG.exists():
    raise SystemExit(f"No {KG}. Run 1_contaminate.py first.")

n_lines = sum(1 for _ in open(KG, encoding="utf-8"))
tf = TriplesFactory.from_path(str(KG))
print(f"{KG.relative_to(HERE)}: {n_lines} lines -> {tf.num_triples} triples, "
      f"{tf.num_entities} entities, {tf.num_relations} relations")
if tf.num_triples != n_lines:
    print(f"  NOTE: from_path deduplicated {n_lines - tf.num_triples} rows")

print(f"\ntraining {args.model} dim={args.dim} epochs={args.epochs} "
      f"seed={args.seed} device={device}")
print("the injected anomalies ARE in this training set, by design")

# pipeline() requires a testing factory, so it gets the training one. The
# metrics below are therefore TRAIN-SET numbers and must never be quoted as
# evaluation -- the real evaluation is steps 3 and 4.
result = pipeline(
    training=tf, testing=tf,
    model=args.model,
    model_kwargs=dict(embedding_dim=args.dim),
    training_kwargs=dict(num_epochs=args.epochs, use_tqdm=False),
    evaluation_kwargs=dict(use_tqdm=False),
    random_seed=args.seed,
    device=device,
)
result.save_to_directory(str(SAVE_TO))
print(f"\nsaved to {SAVE_TO.relative_to(HERE)}")

# --- sanity: is the model degenerate? -------------------------------------
scores = predict_triples(model=result.model, triples=tf).process(factory=tf).df["score"]
print(f"\nscore spread over all {len(scores)} triples:")
print(f"  min {scores.min():.3f}   median {scores.median():.3f}   "
      f"max {scores.max():.3f}   std {scores.std():.4f}")
if scores.std() < 1e-4:
    print("  COLLAPSED: every triple scores the same. Nothing downstream can work.")

# Does it know that locatedin takes a region? Read these, do not just skim them.
# A country in the top 3 means the model has not learned the type of the slot.
for h in ("france", "chad", "brazil"):
    if h not in tf.entity_to_id:
        continue
    top = predict_target(model=result.model, head=h, relation="locatedin",
                         triples_factory=tf).df.head(3)
    print(f"\n({h}, locatedin, ?)")
    for _, row in top.iterrows():
        print(f"   {row['tail_label']:<24} {row['score']:.3f}")
