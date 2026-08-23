"""Detector B: rank every triple by whether its endpoints share connections.

The evidence is the graph itself, not a model. If a fact is true, the two things
it connects should already have connections in common.

  chad locatedin africa   8 of chad's 9 links also touch africa  -> 0.889
  chad locatedin europe   0 of chad's 9 links touch europe       -> 0.000

No training, no torch. Pure counting.

  python 3_detect_based_on_neighbourhood.py
  python 3_detect_based_on_neighbourhood.py --budget 0.05
"""
import argparse
from pathlib import Path

import pandas as pd

from utils import evaluate, viewpoints

VIEW = "neighbourhood"

HERE = Path(__file__).resolve().parent
KG = HERE / "data" / "contaminated_kg.tsv"
GT = HERE / "data" / "ground_truth.tsv"

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=float, default=0.10, help="fraction of the graph to flag")
ap.add_argument("--show", type=int, default=15, help="how many flagged rows to print")
args = ap.parse_args()

for p in (KG, GT):
    if not p.exists():
        raise SystemExit(f"Missing {p}. Run 1_contaminate.py first.")

# No model_dir: this viewpoint needs no model, so torch is never imported.
ctx = viewpoints.build_context(KG)
values = viewpoints.VIEWPOINTS[VIEW](ctx)
print(f"scored {len(values)} triples (no model needed)")

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
                      f"--- flag the {n_flag} least supported triples ---"))

print(f"\n{args.show} least supported:")
print(table.assign(r=rank).sort_values("r").head(args.show)[
    ["head", "relation", "tail", VIEW, "kind"]].to_string(index=False))
