"""Detector C: combine every viewpoint, flag the worst 10%.

Runs each viewpoint over the same triples, keeps the results in memory, and
combines them two ways:

  rank-average  average each triple's rank across viewpoints.
  intersection  flag only what EVERY viewpoint dislikes.

Read the intersection row carefully: it flags FEWER triples than the budget, so
its precision is not comparable to the rows above it. Rank-average cut to the
same number of flags beats it. The row is reported for the set relationship it
shows, not as a better operating point.

Both single-viewpoint detectors are reported alongside, from the same code, so
the comparison is like for like.

  python 4_detect_multi.py
  python 4_detect_multi.py --budget 0.05
  python 4_detect_multi.py --views score neighbourhood
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

HERE = Path(__file__).resolve().parent
KG = HERE / "data" / "contaminated_kg.tsv"
GT = HERE / "data" / "ground_truth.tsv"
MODEL = HERE / "model"

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=float, default=0.10, help="fraction of the graph to flag")
ap.add_argument("--views", nargs="+", default=list(viewpoints.VIEWPOINTS),
                choices=list(viewpoints.VIEWPOINTS))
ap.add_argument("--device", default="auto", help="auto, cpu, or cuda")
ap.add_argument("--show", type=int, default=12, help="how many flagged rows to print")
args = ap.parse_args()

for p in (KG, GT):
    if not p.exists():
        raise SystemExit(f"Missing {p}. Run 1_contaminate.py first.")

device = args.device
if device == "auto":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model only if a chosen viewpoint actually needs one.
needs_model = bool(set(args.views) & viewpoints.NEEDS_MODEL)
if needs_model and not (MODEL / "trained_model.pkl").exists():
    raise SystemExit(f"Missing {MODEL}. Run 2_train.py first.")

ctx = viewpoints.build_context(KG, MODEL if needs_model else None, device)
print(f"{len(ctx['triples'])} triples   viewpoints: {', '.join(args.views)}")

# Every viewpoint scores every triple. Results stay in memory, aligned.
results = viewpoints.run(ctx, args.views)

table = pd.DataFrame(ctx["triples"], columns=["head", "relation", "tail"])
for name, values in results.items():
    table[name] = values

# Labels attached only AFTER every score exists, so they cannot influence one.
table = evaluate.attach_truth(table, GT)

n_flag = int(args.budget * len(table))
kinds = ", ".join(f"{int((table.kind == k).sum())} {k}" for k in evaluate.KINDS)
print(f"the graph holds {int((table.label == 1).sum())} anomalies ({kinds})")
print(f"budget: {n_flag} flags\n")


def top(series, k):
    """Boolean mask over `table` for the k most anomalous rows of `series`."""
    return table.index.isin(series.nsmallest(k).index)


ranks = {n: viewpoints.to_rank(v, viewpoints.DIRECTION[n]) for n, v in results.items()}

# --- each viewpoint alone, at the full budget ---------------------------
for name in args.views:
    flagged = top(ranks[name], n_flag)
    print(evaluate.render(evaluate.evaluate(table, flagged), f"--- {name} alone ---"))
    print()

# --- rank-average: the headline combination -----------------------------
avg = viewpoints.rank_average(results)
avg_flag = top(avg, n_flag)
print(evaluate.render(evaluate.evaluate(table, avg_flag), "--- rank-average ---"))

# --- intersection: flagged only if EVERY viewpoint dislikes it ----------
inter = pd.Series(True, index=table.index)
for name in args.views:
    inter &= top(ranks[name], n_flag)
print()
print(evaluate.render(evaluate.evaluate(table, inter), "--- intersection (all views) ---"))

# Rank-average cut to the SAME number of flags, so the row above can be read
# honestly. Fewer flags buys precision for free; only a like-for-like
# comparison says whether the intersection is actually a better operating point.
n_inter = int(inter.sum())
if n_inter and n_inter != n_flag:
    print()
    print(evaluate.render(evaluate.evaluate(table, top(avg, n_inter)),
                          f"--- rank-average cut to {n_inter} flags (like for like) ---"))

# --- who caught what ----------------------------------------------------
# The number that decides whether combining is worth anything: if no anomaly
# is found by exactly one viewpoint, the views are redundant and any
# combination is just the best one with a handicap.
anom = table.index[table.label == 1]
caught = {n: set(table.index[top(ranks[n], n_flag)]) & set(anom) for n in args.views}
print("\n--- complementarity ---")
for name in args.views:
    others = set().union(*(caught[o] for o in args.views if o != name)) if len(args.views) > 1 else set()
    print(f"  found ONLY by {name:16} {len(caught[name] - others)}")
print(f"  found by every viewpoint      {len(set.intersection(*caught.values()))}")
print(f"  missed by all                 {len(set(anom) - set().union(*caught.values()))}")

print(f"\n{args.show} most anomalous by rank-average:")
cols = ["head", "relation", "tail"] + args.views + ["kind"]
print(table.assign(rank_avg=avg).sort_values("rank_avg").head(args.show)[cols].to_string(index=False))
