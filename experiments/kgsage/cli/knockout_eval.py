# -*- coding: utf-8 -*-
"""Anchor-knockout evaluation for candidate_v2 checkpoints.

Tail-slot protocol (anchor = head): score the relation's FULL tail pool the
same way the decode path does, take the top-10, then re-score with E'(anchor)
AND sketch(anchor) replaced by dataset means. Jaccard between the two lists
measures how much the ranking depends on the anchor's neighbourhood:
  knockout J@10 ~ 1.0  -> anchor ignored (the v1 failure mode)
  low knockout J@10    -> the weights read the anchor's neighbourhood.
Cross-head J@10 (pairwise between different heads, same relation) is the
anchor-invariance companion metric; top-1 dominance shows collapse.

This is the SNAPSHOT SELECTION criterion: run it on every .epNN.pt snapshot
and promote the one with the LOWEST mean knockout J@10 (P6 showed the final
epoch is not the best generator). Default relations are FB15K-237; for
WN18RR pass e.g. --relations _hypernym _derivationally_related_form
_member_meronym _has_part.

Run from repo root (any env with torch):
  PYTHONPATH=experiments python experiments/kgsage/cli/knockout_eval.py \
      --ckpt <v2.pt> --data data/FB15K-237 [--per_rel 12]
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "experiments")
from kgsage.inference import load_checkpoint  # noqa: E402


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
    args = ap.parse_args()

    P = load_checkpoint(args.ckpt, device=torch.device("cpu"))
    if P.get("arch") != "candidate_v2":
        raise SystemExit("not a candidate_v2 checkpoint -- use head_dependence.py for v1")
    G, ctx, sk = P["generator"], P["entity_context"], P["sketches"]
    e2g, r2g, id2e = P["ent2id"], P["rel2id"], P["id2ent"]
    ent_txt = _text(Path(args.data) / "entity2text.txt")
    nm = lambda gid: ent_txt.get(id2e[gid], id2e[gid])[:26]

    mean_ctx = ctx.mean(dim=0)
    mean_sk = sk.mean(dim=0, keepdim=True)

    rng = np.random.default_rng(args.seed)
    by_rel = defaultdict(list)
    for line in open(Path(args.data) / "train.txt", encoding="utf-8-sig"):
        h, r, t = line.rstrip("\n").split("\t")
        if r in args.relations and h in e2g and t in e2g:
            by_rel[r].append((e2g[h], r2g[r], e2g[t]))

    def score_full(h, r, t, ctx_use, sk_row):
        """Mirror of the decode path: full tail pool -> n_ent vector + masks."""
        pool_row = P["pool_masks"][1, r]
        pool_ids = torch.nonzero(pool_row).flatten()
        if len(pool_ids) == 0:
            pool_ids = torch.arange(P["n_ent"])
        with torch.no_grad():
            lg = G(torch.tensor([h]), torch.tensor([r]), torch.tensor([t]),
                   ctx_use, sk_row, pool_ids.unsqueeze(0),
                   torch.zeros(1, len(pool_ids)), 2)[0]
        full = torch.full((P["n_ent"],), float("-inf"))
        full[pool_ids] = lg
        for b in P["true_tails"].get((h, r), []):
            full[b] = float("-inf")
        full[t] = float("-inf")
        full[h] = float("-inf")
        return full

    k = args.topk
    print(f"checkpoint: {args.ckpt}")
    print(f"{'relation':<34} {'n':>3} {'x-head J@10':>12} {'same top1':>10} "
          f"{'knockout J@10':>14}   dominant top-1 pick")
    print("-" * 108)

    rel_means = {}
    for r_str in args.relations:
        rows = by_rel.get(r_str, [])
        if len(rows) < 4:
            print(f"{r_str[:34]:<34}  -- too few triples, skipped")
            continue
        pick = rng.choice(len(rows), size=min(args.per_rel, len(rows)),
                          replace=False)
        sample = [rows[i] for i in pick]

        tops, top1s, kos = [], [], []
        for h, r, t in sample:
            full = score_full(h, r, t, ctx, sk[[h]])
            top = torch.topk(full, k).indices.tolist()
            ctx_ko = ctx.clone()
            ctx_ko[h] = mean_ctx
            full_ko = score_full(h, r, t, ctx_ko, mean_sk)
            top_ko = torch.topk(full_ko, k).indices.tolist()
            tops.append(top)
            top1s.append(top[0])
            kos.append(jac(top, top_ko))

        pair_j = [jac(tops[i], tops[j])
                  for i in range(len(tops)) for j in range(i + 1, len(tops))]
        dom, domn = Counter(top1s).most_common(1)[0]
        rel_means[r_str] = float(np.mean(kos))
        print(f"{r_str[:34]:<34} {len(sample):>3} {np.mean(pair_j):>12.3f} "
              f"{sum(1 for x in top1s if x == dom)/len(top1s):>9.0%} "
              f"{np.mean(kos):>14.3f}   {nm(dom)} ({domn}/{len(sample)})")

    if rel_means:
        print("-" * 108)
        print(f"MEAN knockout J@10 over {len(rel_means)} relations: "
              f"{np.mean(list(rel_means.values())):.3f}")
    print()
    print("READING: knockout J@10 ~1 = deleting the anchor's neighbourhood does not")
    print("change the list (anchor ignored); x-head J@10 ~1 = one shared ranking for")
    print("every head (popularity/universal-alien collapse).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
