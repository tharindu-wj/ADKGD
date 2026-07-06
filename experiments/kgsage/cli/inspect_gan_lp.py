"""Eyeball + aggregate diagnostics for GAN negatives under the frozen LP
(stage A6). The counterpart of cli/inspect_band.py: corrupts N sampled
positives through the DEPLOYED decode path (kgsage.inference, i.e. exactly
what a downstream detector receives) and scores the emissions with the frozen ComplEx.

Usage (repo root, pytorch env):
  PYTHONPATH=experiments python -m kgsage.cli.inspect_gan_lp \
      --ckpt experiments/kgsage/outputs/checkpoints/kgsage_mini.pt \
      --data data/FB15K-mini \
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

from kgsage.lp_scorer import ComplExScorer                      # noqa: E402
from kgsage.band_sampler import _read_triples                   # noqa: E402
from kgsage.inference import (load_checkpoint, generate_negatives,  # noqa: E402
                              render_stats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="KGSAGE GAN checkpoint (.pt)")
    ap.add_argument("--data", required=True, help="dataset dir the ckpt was trained on")
    ap.add_argument("--lp_ckpt", required=True)
    ap.add_argument("--lp_ids", required=True)
    ap.add_argument("--split", default="train", choices=["train", "valid", "test"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    payload = load_checkpoint(args.ckpt)
    scorer = ComplExScorer.from_libkge(args.lp_ckpt, args.lp_ids)

    triples = _read_triples(Path(args.data) / f"{args.split}.txt")
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(triples), size=min(args.n, len(triples)), replace=False)
    positives_str = [triples[i] for i in picks]

    # the GAN payload's own maps ARE the identity id_maps here
    e2g, r2g = payload["ent2id"], payload["rel2id"]
    pos_ids = [(e2g[h], r2g[r], e2g[t]) for h, r, t in positives_str]
    id_maps = {"id2ent": payload["id2ent"], "id2rel": payload["id2rel"],
                  "ent2id": e2g, "rel2id": r2g}
    negatives, stats = generate_negatives(pos_ids, payload, id_maps,
                                          rng=np.random.default_rng(args.seed + 1))

    e2r, r2b = scorer.ent2row, scorer.rel2base
    id2e, id2r = payload["id2ent"], payload["id2rel"]
    gaps, above = [], 0
    print(f"\n{'=' * 76}\nGAN-arm inspection: {args.ckpt}\n"
          f"arm ckpt on {args.data} [{args.split}] n={len(pos_ids)}\n{'=' * 76}")
    with torch.no_grad():
        for (h, r, t), (nh, nr, nt) in zip(pos_ids, negatives):
            hs, rs, ts = id2e[h], id2r[r], id2e[t]
            slot = "tail" if nt != t else "head"
            rr = torch.tensor([r2b[rs]])
            if slot == "tail":
                row = scorer.score_tails_all(torch.tensor([e2r[hs]]), rr)[0]
                s_true, s_neg = float(row[e2r[ts]]), float(row[e2r[id2e[nt]]])
                changed = f"tail: {ts} -> {id2e[nt]}"
            else:
                row = scorer.score_heads_all(rr, torch.tensor([e2r[ts]]))[0]
                s_true, s_neg = float(row[e2r[hs]]), float(row[e2r[id2e[nh]]])
                changed = f"head: {hs} -> {id2e[nh]}"
            gap = s_true - s_neg
            gaps.append(gap)
            above += int(gap < 0)
            print(f"\n  ({hs}, {rs}, {ts})")
            print(f"    {changed}")
            print(f"    s_f(true)={s_true:7.3f}  s_f(neg)={s_neg:7.3f}  gap={gap:6.3f}")

    print(f"\n{'-' * 76}")
    print('[deployed decode] ' + render_stats(stats))
    g = np.array(gaps)
    print(f"gap s_f(true)-s_f(neg): median={np.median(g):.3f} "
          f"p10={np.percentile(g, 10):.3f} p90={np.percentile(g, 90):.3f} "
          f"above-true={above}/{len(g)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
