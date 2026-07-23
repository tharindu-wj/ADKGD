# -*- coding: utf-8 -*-
"""Shared stage-1 for the LLM (7.3) and ego-graph (7.4) evaluations.

Generate N corruptions from a trained generator on a chosen split and write ONE
rich CSV. Two consumers read it:
  - ego_from_csv.py   -> renders an ego graph per row (7.4)
  - blind_from_csv.py -> makes the blind, control-mixed upload file (7.3)

The CSV carries both the raw string ids (for ego rendering) and natural-language
statements (for the LLM), so neither consumer needs the checkpoint again.

Run from repo root (pytorch env):
  PYTHONPATH=experiments python experiments/kgsage/cli/gen_corruptions_csv.py \
      --ckpt experiments/kgsage/outputs/checkpoints/v2_fb_dyn5.pt \
      --data data/FB15K-237 --split test --per_rel 4 --seed 7 \
      --out experiments/kgsage/outputs/eval/fb_corruptions.csv
"""
from __future__ import annotations
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "experiments")
from kgsage.inference import load_checkpoint, generate_negatives  # noqa: E402

# Each eval script writes into its own subfolder under outputs/eval/, resolved
# relative to this file so the location is correct regardless of cwd.
_EVAL_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "eval"

# Human-phraseable relations -> a natural-language template. Extend for YAGO.
TEMPLATES = {
    "/people/person/place_of_birth": "{h} was born in {t}.",
    "/people/person/nationality": "{h} is a citizen of {t}.",
    "/people/person/profession": "{h}'s profession is {t}.",
    "/film/film/genre": "The film {h} belongs to the genre {t}.",
    "/film/film/language": "The film {h} is in {t}.",
    "/film/film/country": "The film {h} was produced in {t}.",
    "/music/artist/origin": "The musical artist {h} originates from {t}.",
}


def _clean(name: str) -> str:
    """Trim FB15K-237 entity2text artifacts for readable statements."""
    name = re.sub(r"-GB\b", "", name)               # Actor-GB -> Actor
    name = re.sub(r"\bLanguage\b", "", name).strip()  # English Language -> English
    return re.sub(r"\s{2,}", " ", name)


def _text_map(p: Path) -> dict:
    m = {}
    if p.is_file():
        for line in open(p, encoding="utf-8-sig"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                m[parts[0]] = parts[1]
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test", choices=["test", "valid", "train"])
    ap.add_argument("--per_rel", type=int, default=4,
                    help="corruptions per phraseable relation (balances the set)")
    ap.add_argument("--relations", nargs="*", default=list(TEMPLATES),
                    help="restrict to these relations (default: all templated)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None,
                    help="output CSV path; default: "
                         "outputs/eval/gen_corruptions/<dataset>_<split>_corruptions.csv")
    args = ap.parse_args()

    P = load_checkpoint(args.ckpt, device=torch.device("cpu"))
    e2g, r2g = P["ent2id"], P["rel2id"]
    id2e, id2r = P["id2ent"], P["id2rel"]
    maps = {"id2ent": id2e, "id2rel": id2r, "ent2id": e2g, "rel2id": r2g}
    ent_txt = _text_map(Path(args.data) / "entity2text.txt")
    nm = lambda gid: _clean(ent_txt.get(id2e[gid], id2e[gid]))
    raw_name = lambda gid: ent_txt.get(id2e[gid], id2e[gid])

    # undirected adjacency (all splits, from the checkpoint) for the shared-nbr check
    adj = {}
    for a, _, b in P["real_triple_set"]:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    # read the requested split, keep name-resolvable triples of templated relations
    import random as pyrand
    rng_py = pyrand.Random(args.seed)
    by_rel = defaultdict(list)
    split_file = Path(args.data) / f"{args.split}.txt"
    for line in open(split_file, encoding="utf-8-sig"):
        h, r, t = line.rstrip("\n").split("\t")
        if (r in args.relations and r in TEMPLATES
                and h in e2g and t in e2g and h in ent_txt and t in ent_txt):
            by_rel[r].append((h, r, t))

    rows = []
    for r_str in args.relations:
        pool = by_rel.get(r_str, [])
        if not pool:
            continue
        sample = rng_py.sample(pool, min(args.per_rel, len(pool)))
        pos = [(e2g[h], r2g[r], e2g[t]) for h, r, t in sample]
        negs, _ = generate_negatives(pos, P, maps,
                                     rng=np.random.default_rng(args.seed))
        for (h, r, t), (nh, nr, nt) in zip(pos, negs):
            if (nh, nt) == (h, t):
                continue
            slot = "tail" if nt != t else "head"
            anchor = h if slot == "tail" else t
            filler = nt if slot == "tail" else nh
            tpl = TEMPLATES[r_str]
            rows.append({
                "idx": len(rows),
                "relation": r_str,
                "slot": slot,
                "orig_h_id": id2e[h], "orig_r_id": r_str, "orig_t_id": id2e[t],
                "corr_h_id": id2e[nh], "corr_r_id": r_str, "corr_t_id": id2e[nt],
                "orig_h_name": nm(h), "orig_t_name": nm(t),
                "corr_h_name": nm(nh), "corr_t_name": nm(nt),
                "orig_statement": tpl.format(h=nm(h), t=nm(t)),
                "corr_statement": tpl.format(h=nm(nh), t=nm(nt)),
                "shared_neighbours": len(adj.get(anchor, set()) & adj.get(filler, set())),
                "direct_neighbour": int(filler in adj.get(anchor, set())),
                "anchor_degree": len(adj.get(anchor, set())),
            })

    if args.out:
        out_path = Path(args.out)
    else:
        ds = Path(args.data).name
        out_path = _EVAL_ROOT / "gen_corruptions" / f"{ds}_{args.split}_corruptions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} corruptions -> {out_path}")
    print(f"  split={args.split}  relations={len(by_rel)}  "
          f"tail-slot={sum(1 for r in rows if r['slot']=='tail')}  "
          f"0-shared={sum(1 for r in rows if r['shared_neighbours']==0 and not r['direct_neighbour'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
