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
from kgsage.corruption_generation import load_checkpoint, generate_negatives  # noqa: E402

# Each eval script writes into its own subfolder under outputs/eval/, resolved
# relative to this file so the location is correct regardless of cwd.
_EVAL_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "eval"

# Human-phraseable relations -> a natural-language template. Extend for YAGO.
# Relation names are dataset-unique (FB uses /paths/, WN uses _names), so one
# dict serves every dataset: only the relations present in the chosen split are
# ever used. A relation MUST appear here to be sampled (see the filter below).
TEMPLATES = {
    # --- FB15K-237 ---
    "/people/person/place_of_birth": "{h} was born in {t}.",
    "/people/person/nationality": "{h} is a citizen of {t}.",
    "/people/person/profession": "{h}'s profession is {t}.",
    "/film/film/genre": "The film {h} belongs to the genre {t}.",
    "/film/film/language": "The film {h} is in {t}.",
    "/film/film/country": "The film {h} was produced in {t}.",
    "/music/artist/origin": "The musical artist {h} originates from {t}.",
    # --- WN18RR (directions verified against train.txt) ---
    "_hypernym": "{h} is a kind of {t}.",
    "_instance_hypernym": "{h} is an instance of {t}.",
    "_member_meronym": "{t} is a member of {h}.",
    "_has_part": "{h} has part {t}.",
    "_derivationally_related_form": "{h} is derivationally related to {t}.",
    "_synset_domain_topic_of": "{h} belongs to the topic domain of {t}.",
    "_member_of_domain_region": "{t} is a term used in the region {h}.",
    "_member_of_domain_usage": "{t} is a term of the usage type {h}.",
    "_also_see": "{h} is semantically related to {t}.",
    "_verb_group": "{h} is in the same verb group as {t}.",
    "_similar_to": "{h} is similar to {t}.",
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
    # Readable labels where a mapping file exists; otherwise fall back to the
    # raw id (entities) or the last path segment (relations). Datasets without
    # these files (e.g. YAGO, whose ids are already human-readable) just use ids.
    ent_txt = _text_map(Path(args.data) / "entity2text.txt")
    rel_txt = _text_map(Path(args.data) / "relation2text.txt")
    nm = lambda gid: _clean(ent_txt.get(id2e[gid], id2e[gid]))
    rel_label = lambda r_str: rel_txt.get(r_str) or r_str.rstrip("/").split("/")[-1]

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

    # The CSV is deliberately simple: the two triples, head/relation/tail each,
    # in readable labels. ego_from_csv re-derives ids and metrics by matching
    # these names against the graph, so nothing else needs to be stored here.
    rows = []
    n_tail = n_zero = 0
    for r_str in args.relations:
        pool = by_rel.get(r_str, [])
        if not pool:
            continue
        sample = rng_py.sample(pool, min(args.per_rel, len(pool)))
        pos = [(e2g[h], r2g[r], e2g[t]) for h, r, t in sample]
        negs, _ = generate_negatives(pos, P, maps,
                                     rng=np.random.default_rng(args.seed))
        rl = rel_label(r_str)
        for (h, r, t), (nh, nr, nt) in zip(pos, negs):
            if (nh, nt) == (h, t):
                continue
            slot = "tail" if nt != t else "head"
            anchor, filler = (h, nt) if slot == "tail" else (t, nh)
            rows.append({
                "orig_head": nm(h), "orig_relation": rl, "orig_tail": nm(t),
                "corr_head": nm(nh), "corr_relation": rl, "corr_tail": nm(nt),
            })
            n_tail += (slot == "tail")
            n_zero += (len(adj.get(anchor, set()) & adj.get(filler, set())) == 0
                       and filler not in adj.get(anchor, set()))

    if args.out:
        out_path = Path(args.out)
    else:
        ds = Path(args.data).name
        out_path = _EVAL_ROOT / "gen_corruptions" / f"{ds}_{args.split}_corruptions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["orig_head", "orig_relation", "orig_tail",
              "corr_head", "corr_relation", "corr_tail"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} corruptions -> {out_path}")
    print(f"  split={args.split}  relations={len(by_rel)}  "
          f"tail-slot={n_tail}  0-shared={n_zero}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
