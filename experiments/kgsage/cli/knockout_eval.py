# -*- coding: utf-8 -*-
"""Anchor-knockout evaluation for candidate_v2 checkpoints.

Tail-slot protocol (anchor = head): score the relation's FULL tail pool the
same way the decode path does, take the top-10, then re-score with E'(anchor)
AND the anchor's membership sketch replaced by dataset means. Jaccard between
the two lists measures how much the ranking depends on the anchor's
neighbourhood:
  knockout J@10 ~ 1.0  -> the anchor is ignored, i.e. collapse
  low knockout J@10    -> the weights read the anchor's neighbourhood, i.e.
                          the generator is anchor-specific (what we want).
Cross-head J@10 (pairwise between different heads, same relation) is the
anchor-invariance companion metric; top-1 dominance shows collapse.

This is the SNAPSHOT SELECTION criterion: run it on every .epNN.pt snapshot
and promote the one with the LOWEST mean knockout J@10 (Investigation Round 6
showed the final epoch is not the best generator). Default relations are
FB15K-237; for WN18RR pass e.g. --relations _hypernym
_derivationally_related_form _member_meronym _has_part.

Run from repo root (any env with torch):
  PYTHONPATH=experiments python experiments/kgsage/cli/knockout_eval.py \
      --ckpt <checkpoint.pt> --data data/FB15K-237 [--per_rel 12]
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "experiments")
from kgsage.corruption_generation import load_checkpoint  # noqa: E402

# Each eval script writes into its own subfolder under outputs/eval/, resolved
# relative to this file so the location is correct regardless of cwd.
_EVAL_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "eval"


def _text(p):
    m = {}
    p = Path(p)
    if p.is_file():
        for line in open(p, encoding="utf-8-sig"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                m[parts[0]] = parts[1]
    return m


def jac(a, b):
    A, B = set(a), set(b)
    return len(A & B) / max(len(A | B), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--relations", nargs="*", default=[
        "/people/person/place_of_birth",
        "/people/person/nationality",
        "/people/person/profession",
        "/film/film/genre",
        "/film/film/language",
        "/music/artist/origin",
    ])
    ap.add_argument("--per_rel", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", default=None,
                    help="JSON report path; default: "
                         "outputs/eval/knockout/<ckpt-stem>_knockout.json")
    ap.add_argument("--no_save", action="store_true",
                    help="print the table only, write no JSON")
    args = ap.parse_args()

    P = load_checkpoint(args.ckpt, device=torch.device("cpu"))
    # "candidate_v2" is the frozen architecture string for the
    # dual-discriminator architecture; it is compared as a literal here and in
    # corruption_generation.py, so it must not be renamed.
    if P.get("arch") != "candidate_v2":
        raise SystemExit("not a candidate_v2 checkpoint")
    generator = P["generator"]
    context_table = P["context_table"]      # E': the frozen per-entity context table
    membership_sketches = P["sketches"]     # frozen key; sketches = the Bloom membership sketches
    e2g, r2g, id2e = P["ent2id"], P["rel2id"], P["id2ent"]
    ent_txt = _text(Path(args.data) / "entity2text.txt")
    nm = lambda gid: ent_txt.get(id2e[gid], id2e[gid])[:26]

    # Dataset means stand in for the anchor once it is knocked out.
    mean_context = context_table.mean(dim=0)
    mean_sketch = membership_sketches.mean(dim=0, keepdim=True)

    rng = np.random.default_rng(args.seed)
    by_rel = defaultdict(list)
    for line in open(Path(args.data) / "train.txt", encoding="utf-8-sig"):
        h, r, t = line.rstrip("\n").split("\t")
        if r in args.relations and h in e2g and t in e2g:
            by_rel[r].append((e2g[h], r2g[r], e2g[t]))

    def score_full(h, r, t, context_use, sketch_row):
        """Mirror of the decode path: full tail pool -> n_ent vector + masks."""
        pool_row = P["pool_masks"][1, r]   # frozen key; pool_masks = the per-relation type pools
        pool_ids = torch.nonzero(pool_row).flatten()
        if len(pool_ids) == 0:
            pool_ids = torch.arange(P["n_ent"])
        with torch.no_grad():
            logits = generator(
                torch.tensor([h]), torch.tensor([r]), torch.tensor([t]),
                context_use, sketch_row, pool_ids.unsqueeze(0),
                torch.zeros(1, len(pool_ids)), 2)[0]
        full = torch.full((P["n_ent"],), float("-inf"))
        full[pool_ids] = logits
        for b in P["true_tails"].get((h, r), []):
            full[b] = float("-inf")
        full[t] = float("-inf")
        full[h] = float("-inf")
        return full

    k = args.topk
    print(f"checkpoint: {args.ckpt}")
    print(f"{'relation':<34} {'n':>3} {'cross-head J@10':>15} {'same top1':>10} "
          f"{'knockout J@10':>14}   dominant top-1 pick")
    print("-" * 108)

    rel_means = {}
    per_rel = []
    for r_str in args.relations:
        rows = by_rel.get(r_str, [])
        if len(rows) < 4:
            print(f"{r_str[:34]:<34}  -- too few triples, skipped")
            continue
        pick = rng.choice(len(rows), size=min(args.per_rel, len(rows)),
                          replace=False)
        sample = [rows[i] for i in pick]

        tops, top1s, knockouts = [], [], []
        for h, r, t in sample:
            full = score_full(h, r, t, context_table, membership_sketches[[h]])
            top = torch.topk(full, k).indices.tolist()
            context_knockout = context_table.clone()
            context_knockout[h] = mean_context
            full_ko = score_full(h, r, t, context_knockout, mean_sketch)
            top_ko = torch.topk(full_ko, k).indices.tolist()
            tops.append(top)
            top1s.append(top[0])
            knockouts.append(jac(top, top_ko))

        pair_j = [jac(tops[i], tops[j])
                  for i in range(len(tops)) for j in range(i + 1, len(tops))]
        dom, domn = Counter(top1s).most_common(1)[0]
        rel_means[r_str] = float(np.mean(knockouts))
        per_rel.append({
            "relation": r_str, "n": len(sample),
            # JSON field name kept as xhead_j10 so reports already collected stay
            # comparable; in prose the metric is "cross-head J@10".
            "xhead_j10": round(float(np.mean(pair_j)), 4),
            "same_top1_pct": round(100 * sum(1 for x in top1s if x == dom) / len(top1s), 1),
            "knockout_j10": round(float(np.mean(knockouts)), 4),
            "dominant_pick": nm(dom), "dominant_count": domn,
        })
        print(f"{r_str[:34]:<34} {len(sample):>3} {np.mean(pair_j):>15.3f} "
              f"{sum(1 for x in top1s if x == dom)/len(top1s):>9.0%} "
              f"{np.mean(knockouts):>14.3f}   {nm(dom)} ({domn}/{len(sample)})")

    mean_j = float(np.mean(list(rel_means.values()))) if rel_means else None
    if rel_means:
        print("-" * 108)
        print(f"MEAN knockout J@10 over {len(rel_means)} relations: {mean_j:.3f}")
    print()
    print("READING: knockout J@10 ~1 = deleting the anchor's neighbourhood does not")
    print("change the list (anchor ignored, so not anchor-specific); cross-head J@10")
    print("~1 = one shared ranking for every head (popularity/universal-alien collapse).")

    if not args.no_save and per_rel:
        if args.out:
            out = Path(args.out)
        else:
            stem = Path(args.ckpt).stem
            out = _EVAL_ROOT / "knockout" / f"{stem}_knockout.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "checkpoint": args.ckpt, "data": args.data, "seed": args.seed,
            "per_rel": args.per_rel, "topk": k,
            "mean_knockout_j10": round(mean_j, 4) if mean_j is not None else None,
            "relations": per_rel,
        }, indent=2), encoding="utf-8")
        print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
