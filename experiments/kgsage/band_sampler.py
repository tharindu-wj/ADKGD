"""Close-but-false negative sampler over a frozen link-predictor (Option B, B2).

For a true triple (h, r, t) and a chosen entity slot, sample a corruption that
is (a) TYPE-VALID: drawn from the relation's observed slot fillers in the
train split; (b) FALSE by the strongest available oracle: never equal to any
known-true filler of the query across train+valid+test (1-N safe) and never a
self-loop; (c) PLAUSIBLE-BUT-NOT-THE-TRUTH: candidates scoring ABOVE the true
value under the frozen scorer are dropped (counted -- the false-negative
proxy), and the sample is drawn from the top `band_k` remaining candidates BY
RANK (raw KGE scores are not calibrated across relations) with a softmax at
`band_temp`.

Fallback ladder (all counted, never a null / used_original):
    band empty after masking+above-true drop
      -> uniform over the masked pool (still type-valid, still false)
      -> pool exhausted: uniform over all entities minus known-true minus self
         (type validity surrendered but counted; occurs only on degenerate
         relations).

The sampler lives in the scorer's row space; the public API is string-keyed
(kgsage stays detector-agnostic -- id translation is the bridge's job). Scorer =
any object with ent2row/rel2base dicts and score_tails_all / score_heads_all /
score_hrt (see lp_scorer.ComplExScorer; the GAN warmup KGE implements the
same protocol later).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

HEAD, TAIL = 0, 2


def _read_triples(path: Path) -> list[tuple[str, str, str]]:
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                out.append(tuple(parts))
    return out


class BandSampler:
    def __init__(self, scorer, dataset_dir: str | Path,
                 band_k: int = 10, band_temp: float = 0.5):
        self.scorer = scorer
        self.band_k = band_k
        self.band_temp = band_temp
        dataset_dir = Path(dataset_dir)

        splits = {s: _read_triples(dataset_dir / f"{s}.txt")
                  for s in ("train", "valid", "test")}
        if not splits["train"]:
            raise ValueError(f"no train.txt under {dataset_dir}")

        # every string must be scoreable; identical standard splits guarantee it
        missing = {tok for tris in splits.values() for h, r, t in tris
                   for tok in (h, t) if tok not in scorer.ent2row}
        missing |= {r for tris in splits.values() for _, r, _ in tris
                    if r not in scorer.rel2base}
        if missing:
            raise ValueError(
                f"{len(missing)} dataset symbols missing from the scorer map "
                f"(e.g. {sorted(missing)[:3]}) -- wrong scorer for this dataset?")

        e2r, r2b = scorer.ent2row, scorer.rel2base
        # type pools: observed slot fillers per relation, TRAIN split only
        tail_pool: dict[int, set[int]] = {}
        head_pool: dict[int, set[int]] = {}
        for h, r, t in splits["train"]:
            rr = r2b[r]
            head_pool.setdefault(rr, set()).add(e2r[h])
            tail_pool.setdefault(rr, set()).add(e2r[t])
        self.tail_pool = {r: torch.tensor(sorted(v), dtype=torch.long)
                          for r, v in tail_pool.items()}
        self.head_pool = {r: torch.tensor(sorted(v), dtype=torch.long)
                          for r, v in head_pool.items()}

        # falseness oracle: every known-true filler, ALL splits
        self.true_tails: dict[tuple[int, int], set[int]] = {}
        self.true_heads: dict[tuple[int, int], set[int]] = {}
        for tris in splits.values():
            for h, r, t in tris:
                hr, rr, tr = e2r[h], r2b[r], e2r[t]
                self.true_tails.setdefault((hr, rr), set()).add(tr)
                self.true_heads.setdefault((rr, tr), set()).add(hr)

        self.n_ent = scorer.ent_emb.shape[0]
        self.row2ent = {v: k for k, v in e2r.items()}
        # the consumer dataset's entity universe: the global fallback must
        # never leave it (a scorer may cover a superset, e.g. full FB15K-237
        # scoring the FB15K-mini smoke subset)
        universe = {e2r[tok] for tris in splits.values()
                    for h, _, t in tris for tok in (h, t)}
        self.universe = sorted(universe)

    # ------------------------------------------------------------------

    def _pick(self, pool: torch.Tensor, scores: torch.Tensor, s_true: float,
              banned: set[int], self_row: int, rng: np.random.Generator,
              stats: dict) -> int:
        """One corruption pick in row space. `scores` aligns with `pool`."""
        keep = [i for i in range(len(pool))
                if int(pool[i]) not in banned and int(pool[i]) != self_row]
        if keep:
            kept_scores = scores[keep].numpy()
            below = [k for k, s in zip(keep, kept_scores) if s < s_true]
            stats["fn_dropped"] += len(keep) - len(below)
            if below:
                below_scores = scores[below].numpy()
                order = np.argsort(-below_scores)[: self.band_k]
                cand_idx = [below[o] for o in order]
                logits = below_scores[order] / max(self.band_temp, 1e-6)
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                stats["band_used"] += 1
                return int(pool[int(rng.choice(cand_idx, p=probs))])
            stats["fallback_pool_uniform"] += 1
            return int(pool[int(rng.choice(keep))])
        # degenerate relation: whole pool is true/self -- leave type validity
        # but stay inside the consumer dataset's entity universe
        stats["fallback_global"] += 1
        while True:
            cand = self.universe[int(rng.integers(0, len(self.universe)))]
            if cand not in banned and cand != self_row:
                return cand

    def corrupt_strings(self, triples: list[tuple[str, str, str]],
                        rng: np.random.Generator | None = None,
                        seed: int = 0) -> tuple[list[tuple[str, str, str]], dict]:
        """One single-slot (head/tail) close-but-false corruption per input."""
        if rng is None:
            rng = np.random.default_rng(seed)
        e2r, r2b = self.scorer.ent2row, self.scorer.rel2base

        rows = [(e2r[h], r2b[r], e2r[t]) for h, r, t in triples]
        slots = np.where(rng.random(len(rows)) < 0.5, HEAD, TAIL)

        stats = {"n": len(rows), "head_slots": int((slots == HEAD).sum()),
                 "tail_slots": int((slots == TAIL).sum()), "band_used": 0,
                 "fn_dropped": 0, "fallback_pool_uniform": 0,
                 "fallback_global": 0}

        # group by (slot, relation) for batched scoring
        groups: dict[tuple[int, int], list[int]] = {}
        for i, ((_, rr, _), slot) in enumerate(zip(rows, slots)):
            groups.setdefault((int(slot), rr), []).append(i)

        out: list[tuple[str, str, str] | None] = [None] * len(rows)
        with torch.no_grad():
            for (slot, rr), idxs in groups.items():
                pool = (self.tail_pool if slot == TAIL else self.head_pool).get(
                    rr, torch.empty(0, dtype=torch.long))
                hs = torch.tensor([rows[i][0] for i in idxs])
                ts = torch.tensor([rows[i][2] for i in idxs])
                rt = torch.full((len(idxs),), rr, dtype=torch.long)
                if slot == TAIL:
                    all_scores = self.scorer.score_tails_all(hs, rt)
                    # direction-consistent s(true): same score row as candidates
                    s_true = all_scores[torch.arange(len(idxs)), ts]
                else:
                    all_scores = self.scorer.score_heads_all(rt, ts)
                    # reciprocal models score heads in the inverse direction;
                    # comparing candidates against a forward-direction
                    # score_hrt(h,r,t) would be inconsistent -- read the true
                    # head's score from the same row instead
                    s_true = all_scores[torch.arange(len(idxs)), hs]
                pool_scores = (all_scores[:, pool] if len(pool) else
                               torch.empty(len(idxs), 0))
                for j, i in enumerate(idxs):
                    hrow, _, trow = rows[i]
                    if slot == TAIL:
                        banned = self.true_tails.get((hrow, rr), set())
                        pick = self._pick(pool, pool_scores[j], float(s_true[j]),
                                          banned, hrow, rng, stats)
                        out[i] = (triples[i][0], triples[i][1], self.row2ent[pick])
                    else:
                        banned = self.true_heads.get((rr, trow), set())
                        pick = self._pick(pool, pool_scores[j], float(s_true[j]),
                                          banned, trow, rng, stats)
                        out[i] = (self.row2ent[pick], triples[i][1], triples[i][2])

        stats["band_rate"] = stats["band_used"] / max(stats["n"], 1)
        stats["fn_dropped_per_triple"] = stats["fn_dropped"] / max(stats["n"], 1)
        return out, stats  # type: ignore[return-value]


def render_stats(stats: dict) -> str:
    return (f"[lp_band] n={stats['n']} slots h/t={stats['head_slots']}/{stats['tail_slots']} "
            f"band={stats['band_used']} ({stats['band_rate']:.1%}) "
            f"fn_dropped/triple={stats['fn_dropped_per_triple']:.2f} "
            f"fallback pool-uniform={stats['fallback_pool_uniform']} "
            f"global={stats['fallback_global']}")
