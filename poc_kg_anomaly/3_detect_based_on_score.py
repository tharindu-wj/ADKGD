"""Detector A: rank every triple by what the trained MODEL thinks of it.

The evidence is the embedding: a fact that contradicts the rest of the graph
cannot be fitted as well as one the graph supports.

  python 3_detect_based_on_score.py
  python 3_detect_based_on_score.py --budget 0.05
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

import pandas as pd

from utils import evaluate, viewpoints

VIEW = "score"

HERE = Path(__file__).resolve().parent
KG = HERE / "data" / "contaminated_kg.tsv"
GT = HERE / "data" / "ground_truth.tsv"
MODEL = HERE / "model"

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=float, default=0.10, help="fraction of the graph to flag")
ap.add_argument("--show", type=int, default=15, help="how many flagged rows to print")
ap.add_argument("--device", default="auto", help="auto, cpu, or cuda")
args = ap.parse_args()

for p in (KG, GT, MODEL / "trained_model.pkl"):
    if not p.exists():
        raise SystemExit(f"Missing {p}. Run 1_contaminate.py then 2_train.py.")

device = args.device
if device == "auto":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Same function 4_detect_multi.py calls, so the numbers cannot drift apart.
ctx = viewpoints.build_context(KG, MODEL, device)
values = viewpoints.VIEWPOINTS[VIEW](ctx)
print(f"scored {len(values)} triples on device {device}")

table = pd.DataFrame(ctx["triples"], columns=["head", "relation", "tail"])
table[VIEW] = values

# Labels are attached only AFTER scoring, so they cannot influence a score.
table = evaluate.attach_truth(table, GT)

rank = viewpoints.to_rank(values, viewpoints.DIRECTION[VIEW])
n_flag = int(args.budget * len(table))
flagged = table.index.isin(rank.nsmallest(n_flag).index)

kinds = ", ".join(f"{int((table.kind == k).sum())} {k}" for k in evaluate.KINDS)
print(f"the graph holds {int((table.label == 1).sum())} anomalies ({kinds})")
print()
print(evaluate.render(evaluate.evaluate(table, flagged),
                      f"--- flag the {n_flag} least plausible triples ---"))

print(f"\n{args.show} most anomalous:")
print(table.assign(r=rank).sort_values("r").head(args.show)[
    ["head", "relation", "tail", VIEW, "kind"]].to_string(index=False))
