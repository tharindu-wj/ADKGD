"""Step 1: merge Countries, inject two classes of fake, write two files.

  data/contaminated_kg.tsv   h, r, t              <- what the detectors read
  data/ground_truth.tsv      h, r, t, label, kind <- what the evaluator reads

type_invalid = tail from the WRONG relation's pool   (belgium locatedin japan)
type_valid   = tail from the RIGHT pool, wrong value (belgium locatedin africa)

Named for how they are BUILT, not for how hard they are. Measured on this
fixture the model catches both at about the same rate, so calling them
easy/hard would assert a difficulty gap the data does not show.
"""
import os
import sys

# Must be set before importing torch/matplotlib -- both link OpenMP and the
# duplicate runtime aborts the process with OMP Error #15.
if sys.platform == "win32":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np
from pykeen.datasets import Countries

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--ratio", type=float, default=0.10, help="anomalies as a fraction of real triples")
ap.add_argument("--invalid-frac", type=float, default=0.5,
                help="share of anomalies that are type_invalid")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--ego", type=int, default=0,
                help="draw an ego network per anomaly: N per kind, -1 for all, 0 to skip")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)

# Merge all three splits. Countries ships 1110/24/24; we audit the whole graph.
ds = Countries()
id2e = {v: k for k, v in ds.training.entity_to_id.items()}
id2r = {v: k for k, v in ds.training.relation_to_id.items()}

rows = []
for f in (ds.training, ds.validation, ds.testing):
    for h, r, t in f.mapped_triples.numpy().tolist():
        rows.append((id2e[h], id2r[r], id2e[t]))

# sorted() everywhere: Python randomises string hashing per process, so raw set
# iteration would break determinism across runs without any visible error.
real = sorted(set(rows))
known = set(real)
print(f"merged {len(rows)} triples, {len(real)} unique")

# Tail pools, built from the CLEAN graph. Legitimate here because we are
# GENERATING. A DETECTOR must never do this: on the dirty graph every injected
# fake adds its own tail to the pool, so the pool would validate the very
# triples it exists to catch.
tails = {}
for h, r, t in real:
    tails.setdefault(r, set()).add(t)
all_tails = set().union(*tails.values())
own = {r: sorted(s) for r, s in tails.items()}
other = {r: sorted(all_tails - s) for r, s in tails.items()}

for r in sorted(own):
    print(f"  {r}: {len(own[r])} own tails, {len(other[r])} tails belonging to other relations")

made = set()


def corrupt(src, pool):
    """Swap the tail for a random entity from pool. None if no valid swap found."""
    h, r, t = src
    for _ in range(200):
        new_tail = pool[int(rng.integers(len(pool)))]
        if new_tail == t or new_tail == h:
            continue                        # must change, and no self-loops
        if (h, r, new_tail) in known:
            continue                        # never label a true fact as an anomaly
        if (h, r, new_tail) in made:
            continue                        # no duplicate fakes
        made.add((h, r, new_tail))
        return (h, r, new_tail)
    return None


n_anom = int(args.ratio * len(real))
n_invalid = int(args.invalid_frac * n_anom)
n_valid = n_anom - n_invalid

# Draw distinct source triples so one real fact is not corrupted twice.
src_idx = rng.permutation(len(real))[:n_anom]
invalid_src = [real[i] for i in src_idx[:n_invalid]]
valid_src = [real[i] for i in src_idx[n_invalid:]]

# other[r] = tails that belong to the OTHER relation -> breaks the slot's type
# own[r]   = tails that legitimately fill this slot   -> type survives, fact does not
invalid = [c for c in (corrupt(s, other[s[1]]) for s in invalid_src) if c]
valid = [c for c in (corrupt(s, own[s[1]]) for s in valid_src) if c]

print(f"\nrequested {n_anom} anomalies ({n_invalid} type_invalid / {n_valid} type_valid)")
print(f"realised  {len(invalid) + len(valid)} "
      f"({len(invalid)} type_invalid / {len(valid)} type_valid)")

labelled = ([(t, 0, "real") for t in real]
            + [(t, 1, "type_invalid") for t in invalid]
            + [(t, 1, "type_valid") for t in valid])
order = rng.permutation(len(labelled))
labelled = [labelled[i] for i in order]

DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
kg_path = DATA / "contaminated_kg.tsv"
gt_path = DATA / "ground_truth.tsv"

with open(kg_path, "w", encoding="utf-8") as f:
    for (h, r, t), _, _ in labelled:
        f.write(f"{h}\t{r}\t{t}\n")

with open(gt_path, "w", encoding="utf-8") as f:
    for (h, r, t), label, kind in labelled:
        f.write(f"{h}\t{r}\t{t}\t{label}\t{kind}\n")

n_bad = sum(1 for t, lab, _ in labelled if lab == 1 and t in known)
n_loop = sum(1 for (h, _, t), lab, _ in labelled if lab == 1 and h == t)
n_dup = len(labelled) - len({t for t, _, _ in labelled})

print(f"\nwrote {len(labelled)} rows")
print(f"  {kg_path.relative_to(HERE)}   3 columns, no labels")
print(f"  {gt_path.relative_to(HERE)}   5 columns, same row order")
print(f"\nGUARDS  fake-but-actually-true {n_bad}   self-loops {n_loop}   duplicates {n_dup}   (all must be 0)")

print("\nsample TYPE_INVALID fakes (wrong kind of entity in the slot):")
for t in invalid[:5]:
    print("   ", "\t".join(t))
print("\nsample TYPE_VALID fakes (right kind of entity, wrong one):")
for t in valid[:5]:
    print("   ", "\t".join(t))

# One ego network per anomaly, so the injected set can be audited by eye
# instead of taken on trust.
if args.ego != 0:
    # Imported late so matplotlib is only loaded when a picture is actually wanted.
    from utils import generate_ego as egomod

    ctx = egomod.build_context(real)
    ego_dir = HERE / "ego"
    for old in ego_dir.glob("*"):
        old.unlink()                      # stale images from an earlier seed would mislead

    take = (lambda xs: xs) if args.ego < 0 else (lambda xs: xs[:args.ego])
    todo = ([("type_invalid", t) for t in take(invalid)]
            + [("type_valid", t) for t in take(valid)])

    print(f"\ndrawing {len(todo)} ego networks into {ego_dir.name}/ ...")
    entries = []
    for i, (kind, (h, r, t)) in enumerate(todo, 1):
        score, note = egomod.evidence(h, r, t, ctx)
        png = ego_dir / f"{kind}_{i:03d}_{h}_{r}_{t}.png"
        egomod.draw_ego(h, r, t, kind, ctx, png, title_note=note)
        entries.append((kind, h, r, t, png, note))
        if i % 10 == 0:
            print(f"  {i}/{len(todo)}")

    index = egomod.write_index(entries, ego_dir / "index.html")
    print(f"open {index} to review them all in one page")
