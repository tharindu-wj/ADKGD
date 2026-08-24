"""Inject two classes of fake triple into the clean graph.

Reads   data/<name>/{train,valid,test}.txt   the clean graph, checked in
Writes  data/<name>/contaminated_kg.tsv      h, r, t              -> detectors
        data/<name>/ground_truth.tsv         h, r, t, label, kind -> evaluator

type_invalid  tail drawn from the OTHER relation's pool   (belgium locatedin japan)
type_valid    tail drawn from the SAME pool, wrong value  (belgium locatedin africa)

Named for how they are BUILT, not for how hard they are.

    python scripts/1_inject_anomalies.py
    python scripts/1_inject_anomalies.py --ratio 0.15 --invalid-frac 0.3 --seed 7
"""
import sys
from pathlib import Path

# Scripts live in scripts/, so Python puts THAT on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np

from loaders.active import DATASET

ap = argparse.ArgumentParser()
ap.add_argument("--ratio", type=float, default=0.10,
                help="anomalies as a fraction of the real triples")
ap.add_argument("--invalid-frac", type=float, default=0.5,
                help="share of the anomalies that are type_invalid")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

rng = np.random.default_rng(args.seed)

# ---- read the clean graph -------------------------------------------------
rows = []
for name in DATASET.SOURCE:
    path = DATASET.DIR / name
    if not path.exists():
        raise SystemExit(f"missing {path}")
    with open(path, encoding="utf-8") as f:
        rows += [tuple(line.rstrip("\n").split("\t")) for line in f]

# sorted() everywhere: Python randomises string hashing per process, so raw set
# iteration would break determinism across runs with no visible error.
real = sorted(set(rows))
known = set(real)
print(f"read {len(rows)} triples, {len(real)} unique")

# ---- tail pools, from the CLEAN graph -------------------------------------
# Legitimate here because we are GENERATING. A detector must never do this: on
# the dirty graph every injected fake adds its own tail to the pool, so the
# pool would validate the very triples it exists to catch.
tails = {}
for h, r, t in real:
    tails.setdefault(r, set()).add(t)
all_tails = set().union(*tails.values())
own = {r: sorted(s) for r, s in tails.items()}
other = {r: sorted(all_tails - s) for r, s in tails.items()}

for r in sorted(own):
    print(f"  {r}: {len(own[r])} own tails, {len(other[r])} belonging to other relations")

# ---- corrupt --------------------------------------------------------------
made = set()


def corrupt(src, pool):
    """Swap the tail for a random entity from pool. None if no valid swap."""
    h, r, t = src
    for _ in range(200):
        new_tail = pool[int(rng.integers(len(pool)))]
        if new_tail == t or new_tail == h:
            continue                        # must change, and no self-loops
        if (h, r, new_tail) in known:
            continue                        # never label a true fact an anomaly
        if (h, r, new_tail) in made:
            continue                        # no duplicate fakes
        made.add((h, r, new_tail))
        return (h, r, new_tail)
    return None


n_anom = int(args.ratio * len(real))
n_invalid = int(args.invalid_frac * n_anom)

# Distinct source triples, so one real fact is not corrupted twice.
picked = rng.permutation(len(real))[:n_anom]
invalid = [c for c in (corrupt(real[i], other[real[i][1]]) for i in picked[:n_invalid]) if c]
valid = [c for c in (corrupt(real[i], own[real[i][1]]) for i in picked[n_invalid:]) if c]

print(f"\nrequested {n_anom} anomalies ({n_invalid} type_invalid / {n_anom - n_invalid} type_valid)")
print(f"realised  {len(invalid) + len(valid)} ({len(invalid)} type_invalid / {len(valid)} type_valid)")

# ---- write ----------------------------------------------------------------
labelled = ([(t, 0, "real") for t in real]
            + [(t, 1, "type_invalid") for t in invalid]
            + [(t, 1, "type_valid") for t in valid])
labelled = [labelled[i] for i in rng.permutation(len(labelled))]

with open(DATASET.KG, "w", encoding="utf-8") as f:
    for (h, r, t), _, _ in labelled:
        f.write(f"{h}\t{r}\t{t}\n")

with open(DATASET.TRUTH, "w", encoding="utf-8") as f:
    for (h, r, t), label, kind in labelled:
        f.write(f"{h}\t{r}\t{t}\t{label}\t{kind}\n")

n_bad = sum(1 for t, lab, _ in labelled if lab == 1 and t in known)
n_loop = sum(1 for (h, _, t), lab, _ in labelled if lab == 1 and h == t)
n_dup = len(labelled) - len({t for t, _, _ in labelled})

print(f"\nwrote {len(labelled)} rows to {DATASET.KG.name} and {DATASET.TRUTH.name}")
print(f"GUARDS  fake-but-actually-true {n_bad}   self-loops {n_loop}   "
      f"duplicates {n_dup}   (all must be 0)")

print("\nsample TYPE_INVALID (wrong kind of entity in the slot):")
for t in invalid[:4]:
    print("   ", "\t".join(t))
print("\nsample TYPE_VALID (right kind of entity, wrong one):")
for t in valid[:4]:
    print("   ", "\t".join(t))
