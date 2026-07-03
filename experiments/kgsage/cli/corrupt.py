"""Corrupt YOUR OWN triples with a trained KGSAGE checkpoint (hands-on check).

Feed a file of real (head, relation, tail) triples -- tab-separated, the
dataset's own string IDs -- and see the single-slot corruption the generator
makes for each, through the DEPLOYED decode path (exactly what ADKGD receives).
Optionally score each corruption with the frozen LP (is it actually false?) and
resolve opaque IDs to readable names.

Pick guaranteed-valid triples straight from the data:
    head -8 data/FB15K-237/train.txt > my_triples.tsv
    # or a relation you can reason about:
    grep place_of_birth data/FB15K-237/train.txt | head -8 > my_triples.tsv

Then corrupt them (repo root, PYTHONPATH=experiments):
    python -m kgsage.cli.corrupt \
        --ckpt experiments/kgsage/outputs/checkpoints/kgsage_aii_fb15k237_s0.pt \
        --triples my_triples.tsv \
        --lp_ckpt experiments/kgsage/outputs/lp/fb15k-237-complex.pt \
        --lp_ids  experiments/kgsage/outputs/lp/fb15k-237

Reading the output per triple:
    slot          which of head/tail the generator changed (never the relation)
    gap = s_f(true) - s_f(neg)   POSITIVE = the LP judges the corruption less
                  plausible than the true fact (good: it is false). NEGATIVE
                  ("above true") = the LP finds it MORE plausible -> possible
                  false negative (a true-but-unobserved fact). Needs --lp_ckpt.
    KEPT-ORIGINAL every masked redraw failed (degenerate row) -> no corruption.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

from kgsage.inference import (load_checkpoint, generate_negatives,  # noqa: E402
                              render_stats)


def _read_labels(path):
    labels = {}
    if path:
        with open(path, encoding="utf-8-sig") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    labels[parts[0]] = parts[1]
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="trained KGSAGE checkpoint (.pt)")
    ap.add_argument("--triples", required=True,
                    help="TSV file of your own (head, relation, tail) triples "
                         "('-' reads from stdin)")
    ap.add_argument("--lp_ckpt", default=None, help="frozen LibKGE ComplEx (optional; enables the gap)")
    ap.add_argument("--lp_ids", default=None, help="LibKGE archive dir for --lp_ckpt ids")
    ap.add_argument("--labels", default=None, help="optional TSV: id<TAB>readable name")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    labels = _read_labels(args.labels)

    def nice(tok):
        return f"{labels[tok]} [{tok}]" if tok in labels else tok

    payload = load_checkpoint(args.ckpt)
    e2g, r2g = payload["ent2id"], payload["rel2id"]
    id2e, id2r = payload["id2ent"], payload["id2rel"]

    # read + validate the user's triples against the checkpoint vocabulary
    src = sys.stdin if args.triples == "-" else open(args.triples, encoding="utf-8-sig")
    triples, skipped = [], 0
    with src as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            if h not in e2g or t not in e2g or r not in r2g:
                miss = [x for x in (h, r, t) if x not in {**e2g, **r2g}]
                print(f"!! skipping (not in checkpoint vocab): {parts}  missing={miss}",
                      file=sys.stderr)
                skipped += 1
                continue
            triples.append((h, r, t))
    if not triples:
        print("no usable triples (all skipped?)", file=sys.stderr)
        return 1

    pos_ids = [(e2g[h], r2g[r], e2g[t]) for h, r, t in triples]
    maps = {"id2ent": id2e, "id2rel": id2r, "ent2id": e2g, "rel2id": r2g}
    negs, stats = generate_negatives(pos_ids, payload, maps,
                                     rng=np.random.default_rng(args.seed))

    # optional frozen-LP scoring (direction-consistent; reciprocal-aware)
    scorer = None
    if args.lp_ckpt and args.lp_ids:
        import torch
        from kgsage.lp_scorer import ComplExScorer
        scorer = ComplExScorer.from_libkge(args.lp_ckpt, args.lp_ids)

    print(f"\n{'=' * 74}\ncorrupting {len(triples)} triple(s) with {Path(args.ckpt).name}\n{'=' * 74}")
    above_true = 0
    for (h, r, t), (nh, nr, nt) in zip(pos_ids, negs):
        if nh == h and nt == t:
            print(f"\n  ({nice(id2e[h])}, {id2r[r]}, {nice(id2e[t])})")
            print("    KEPT-ORIGINAL (no valid corruption found for this triple)")
            continue
        slot = "tail" if nt != t else "head"
        moved = (f"{nice(id2e[t])} -> {nice(id2e[nt])}" if slot == "tail"
                 else f"{nice(id2e[h])} -> {nice(id2e[nh])}")
        print(f"\n  ({nice(id2e[h])}, {id2r[r]}, {nice(id2e[t])})")
        print(f"    {slot}: {moved}")
        if scorer is not None:
            import torch
            e2r, r2b = scorer.ent2row, scorer.rel2base
            hs, rs, ts = id2e[h], id2r[r], id2e[t]
            if hs in e2r and ts in e2r and rs in r2b:
                rr = torch.tensor([r2b[rs]])
                with torch.no_grad():
                    if slot == "tail":
                        row = scorer.score_tails_all(torch.tensor([e2r[hs]]), rr)[0]
                        s_true, s_neg = float(row[e2r[ts]]), float(row[e2r[id2e[nt]]])
                    else:
                        row = scorer.score_heads_all(rr, torch.tensor([e2r[ts]]))[0]
                        s_true, s_neg = float(row[e2r[hs]]), float(row[e2r[id2e[nh]]])
                gap = s_true - s_neg
                above_true += int(gap < 0)
                flag = "  <-- ABOVE TRUE (possible false negative)" if gap < 0 else ""
                print(f"    s_f(true)={s_true:7.3f}  s_f(neg)={s_neg:7.3f}  gap={gap:6.3f}{flag}")

    print(f"\n{'-' * 74}")
    print('[deployed decode] ' + render_stats(stats))
    if scorer is not None:
        print(f"above-true (false-negative risk): {above_true}/{len(triples)}")
    if skipped:
        print(f"({skipped} input line(s) skipped: symbols not in the checkpoint vocab)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
