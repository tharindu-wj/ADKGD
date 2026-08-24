"""Test the plausibility scorer: rank every triple, flag the worst 10%.

    python scripts/check_scorer_plausibility.py
    python scripts/check_scorer_plausibility.py --budget 0.05
"""
import sys
from pathlib import Path

# Scripts live in scripts/, so Python puts THAT on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import pandas as pd

from loaders import graph
from loaders.active import DATASET
from tools.scorers import plausibility as scorer
from utils import detect, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=float, default=0.10, help="fraction to flag")
ap.add_argument("--show", type=int, default=12, help="rows to print")
ap.add_argument("--model", default="distmult", help="folder under models/<name>/")
ap.add_argument("--device", default="auto", help="auto, cpu, or cuda")
args = ap.parse_args()

MODEL = DATASET.MODELS / args.model
for p in (DATASET.KG, DATASET.TRUTH):
    if not p.exists():
        raise SystemExit(f"missing {p}. Run scripts/1_inject_anomalies.py first.")
if not (MODEL / "trained_model.pkl").exists():
    raise SystemExit(f"missing {MODEL}. Run scripts/2_train_plausibility_scorer.py first.")

device = args.device
if device == "auto":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

triples = graph.load_triples(DATASET.KG)
values = scorer.score(triples, model_dir=MODEL, kg_path=DATASET.KG, device=device)
print(f"{scorer.NAME}: scored {len(values)} triples using {MODEL.name} on {device}")

table = pd.DataFrame(triples, columns=["head", "relation", "tail"])
table[scorer.NAME] = values

# Labels attached only AFTER scoring, so they cannot influence a score.
table = evaluate.attach_truth(table, DATASET.TRUTH)

flagged, n = detect.flag_worst(values, scorer.DIRECTION, args.budget)
kinds = ", ".join(f"{int((table.kind == k).sum())} {k}" for k in evaluate.KINDS)
print(f"the graph holds {int((table.label == 1).sum())} anomalies ({kinds})\n")
print(evaluate.render(evaluate.score(table, flagged),
                      f"--- {scorer.NAME}: flag the worst {n} ---"))

if args.show:
    print(f"\n{args.show} most implausible:")
    print(table[flagged].nsmallest(args.show, scorer.NAME)[
        ["head", "relation", "tail", scorer.NAME, "kind"]].to_string(index=False))
