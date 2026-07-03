"""Eyeball + aggregate diagnostics for the Option-B band sampler (B4).

Samples N positives from a split, corrupts them with BandSampler, and prints
per-sample scoring detail (s(true), s(neg), gap, rank-in-pool, pool size)
plus the aggregate stats line. Optional --labels TSV (entity<TAB>name) makes
FB15K MIDs / WN18RR synset offsets readable.

Usage (repo root, pytorch env):
  python -m kgsage.cli.inspect_band --data data/FB15K-237 \
      --lp_ckpt experiments/kgsage/outputs/lp/fb15k-237-complex.pt \
      --lp_ids  experiments/kgsage/outputs/lp/fb15k-237 [--n 30] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

from kgsage.lp_scorer import ComplExScorer            # noqa: E402
from kgsage.band_sampler import (BandSampler, _read_triples,  # noqa: E402
                                 render_stats, HEAD, TAIL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="ADKGD dataset dir (train/valid/test.txt)")
    ap.add_argument("--lp_ckpt", required=True)
    ap.add_argument("--lp_ids", required=True)
    ap.add_argument("--split", default="train", choices=["train", "valid", "test"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--band_k", type=int, default=10)
    ap.add_argument("--band_temp", type=float, default=0.5)
    ap.add_argument("--labels", default=None, help="optional TSV: entity<TAB>readable name")
    args = ap.parse_args()

    labels = {}
    if args.labels:
        with open(args.labels, encoding="utf-8-sig") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    labels[parts[0]] = parts[1]

    def nice(tok: str) -> str:
        return f"{tok} ({labels[tok]})" if tok in labels else tok

    scorer = ComplExScorer.from_libkge(args.lp_ckpt, args.lp_ids)
    sampler = BandSampler(scorer, args.data, band_k=args.band_k,
                          band_temp=args.band_temp)

    triples = _read_triples(Path(args.data) / f"{args.split}.txt")
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(triples), size=min(args.n, len(triples)), replace=False)
    positives = [triples[i] for i in picks]
    negatives, stats = sampler.corrupt_strings(positives, seed=args.seed + 1)

    e2r, r2b = scorer.ent2row, scorer.rel2base
    gaps, ranks = [], []
    print(f"\n{'=' * 76}\nBand-sampler inspection: {args.data} [{args.split}] "
          f"n={len(positives)} seed={args.seed}\n{'=' * 76}")
    with torch.no_grad():
        for (h, r, t), (nh, nr, nt) in zip(positives, negatives):
            slot = TAIL if nt != t else HEAD
            rr = r2b[r]
            hs = torch.tensor([e2r[h]]); ts = torch.tensor([e2r[t]])
            rt = torch.tensor([rr])
            # all comparisons in ONE scoring direction (reciprocal models score
            # heads in the inverse direction; mixing directions is meaningless)
            if slot == TAIL:
                all_scores = scorer.score_tails_all(hs, rt)[0]
                s_true, s_neg = float(all_scores[e2r[t]]), float(all_scores[e2r[nt]])
                pool = sampler.tail_pool.get(rr, torch.empty(0, dtype=torch.long))
                banned = sampler.true_tails.get((e2r[h], rr), set())
                changed = f"tail: {nice(t)} -> {nice(nt)}"
            else:
                all_scores = scorer.score_heads_all(rt, ts)[0]
                s_true, s_neg = float(all_scores[e2r[h]]), float(all_scores[e2r[nh]])
                pool = sampler.head_pool.get(rr, torch.empty(0, dtype=torch.long))
                banned = sampler.true_heads.get((rr, e2r[t]), set())
                changed = f"head: {nice(h)} -> {nice(nh)}"
            masked = [int(p) for p in pool if int(p) not in banned]
            pool_scores = all_scores[masked] if masked else torch.empty(0)
            rank = int((pool_scores > s_neg).sum().item()) + 1 if masked else -1
            gap = s_true - s_neg
            gaps.append(gap); ranks.append(rank)
            print(f"\n  ({nice(h)}, {r}, {nice(t)})")
            print(f"    {changed}")
            print(f"    s(true)={s_true:7.3f}  s(neg)={s_neg:7.3f}  gap={gap:6.3f}  "
                  f"rank-in-pool={rank}/{len(masked)}  banned={len(banned)}")

    print(f"\n{'-' * 76}")
    print(render_stats(stats))
    gaps_np, ranks_np = np.array(gaps), np.array(ranks)
    print(f"gap  s(true)-s(neg): median={np.median(gaps_np):.3f} "
          f"p10={np.percentile(gaps_np, 10):.3f} p90={np.percentile(gaps_np, 90):.3f} "
          f"negatives-above-true={(gaps_np < 0).sum()}/{len(gaps_np)}")
    print(f"rank-in-pool:        median={np.median(ranks_np[ranks_np > 0]):.0f} "
          f"p90={np.percentile(ranks_np[ranks_np > 0], 90):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
