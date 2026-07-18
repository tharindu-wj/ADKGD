"""P2 KILL-SWITCH for KGSAGE-2: can D_match learn corroboration at all?

Trains the D_match critic ALONE (no GAN) on pairs built purely from data:
  positive  (anchor h, its true slot filler t)            -> 1
  negative  (anchor h, SAME-RELATION filler of another h') -> 0   [strict form]
with the direct h--candidate edge excluded from the neighbour sample.

Reports held-out AUC. GO if AUC >= 0.75 (the untrained max-cos proxy already
reaches 0.80 on the LOOSE mismatch form; the same-relation form here is
strictly harder, which is why it is the honest gate). NO-GO means the critic
architecture cannot read the neighbourhood and the adversarial build must not
proceed.

Usage (repo root, pytorch env; CPU is fine):
  PYTHONPATH=experiments python -m kgsage.cli.dmatch_gate \
      --ckpt experiments/kgsage/outputs/checkpoints/kgsage_fb15k237_s0.pt \
      [--pairs 40000] [--epochs 3] [--n_nbr 32] [--seed 0]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

from kgsage.inference import load_checkpoint            # noqa: E402
from kgsage.gan.d_match import DMatch                   # noqa: E402


def build_pairs(triples, adj, by_rel, n_pairs, rng):
    """[(anchor, candidate, label)] with strict same-relation mismatches."""
    pairs = []
    sample = rng.sample(triples, min(n_pairs // 2, len(triples)))
    for h, r, t in sample:
        pairs.append((h, t, 1))
        others = by_rel[r]
        for _ in range(8):                       # find a genuine mismatch
            h2, _, t2 = others[rng.randrange(len(others))]
            if t2 != t and t2 not in adj.get(h, set()):
                pairs.append((h, t2, 0))
                break
    rng.shuffle(pairs)
    return pairs


def batches(pairs, adj, ctx, n_nbr, bs, rng, device):
    for s in range(0, len(pairs), bs):
        chunk = pairs[s:s + bs]
        B = len(chunk)
        cand = torch.tensor([c for _, c, _ in chunk], dtype=torch.long)
        y = torch.tensor([l for _, _, l in chunk], dtype=torch.float32)
        nbr = torch.zeros(B, n_nbr, dtype=torch.long)
        mask = torch.zeros(B, n_nbr, dtype=torch.bool)
        for i, (a, c, _) in enumerate(chunk):
            ns = [n for n in adj.get(a, ()) if n != c]     # EXCLUDE direct edge
            if not ns:
                continue
            if len(ns) > n_nbr:
                ns = rng.sample(ns, n_nbr)
            nbr[i, :len(ns)] = torch.tensor(ns)
            mask[i, :len(ns)] = True
        yield (ctx[cand].to(device), ctx[nbr].to(device), mask.to(device),
               y.to(device))


def auc_of(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(len(scores))
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 - 1) / 2) / max(n1 * n0, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n_nbr", type=int, default=32)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", type=float, default=0.75)
    args = ap.parse_args()

    device = torch.device("cpu")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    P = load_checkpoint(args.ckpt, device=device)
    ctx = P["entity_context"]
    triples = list(P["real_triple_set"])
    adj: dict[int, set] = {}
    by_rel = defaultdict(list)
    for h, r, t in triples:
        adj.setdefault(h, set()).add(t)
        adj.setdefault(t, set()).add(h)
        by_rel[r].append((h, r, t))

    pairs = build_pairs(triples, adj, by_rel, args.pairs, rng)
    n_test = max(2000, len(pairs) // 10)
    test, train = pairs[:n_test], pairs[n_test:]
    print(f"pairs: train={len(train)}  held-out={n_test}  "
          f"(strict same-relation mismatches)")

    D = DMatch(dim=ctx.shape[1]).to(device)
    opt = torch.optim.AdamW(D.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()

    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        rng.shuffle(train)
        tot = nb = 0
        for ce, ne, mk, y in batches(train, adj, ctx, args.n_nbr, args.batch,
                                     rng, device):
            loss = bce(D(ce, ne, mk), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        # held-out AUC
        D.eval()
        scores, labels = [], []
        with torch.no_grad():
            for ce, ne, mk, y in batches(test, adj, ctx, args.n_nbr,
                                         args.batch, rng, device):
                scores.append(D(ce, ne, mk).numpy()); labels.append(y.numpy())
        D.train()
        auc = auc_of(np.concatenate(scores), np.concatenate(labels))
        print(f"  epoch {ep}/{args.epochs}  bce={tot/nb:.4f}  "
              f"held-out AUC={auc:.4f}  ({time.perf_counter()-t0:.0f}s)")

    verdict = "GO" if auc >= args.gate else "NO-GO"
    print(f"\nGATE: AUC={auc:.4f} vs threshold {args.gate}  ->  {verdict}")
    return 0 if auc >= args.gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
