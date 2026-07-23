# -*- coding: utf-8 -*-
"""7.4 (semantic lens): neighbourhood-context blocks for the LLM contradiction
evaluation.

For each corruption in the CSV, gather the ANCHOR entity's local facts from the
graph, render them readable, and emit a case block: the facts + one triple to
assess. Paste the blocks into an LLM with the 3-way prompt (SUPPORTED / NEUTRAL
/ CONTRADICTED) -- KGSAGE corruptions should come back not-supported, while the
true tails (the control) should come back supported.

The anchor is resolved by matching the ORIGINAL (real) triple against the graph,
which pins the ids and disambiguates repeated display names. When assessing the
true triple (--which control), that exact edge is EXCLUDED from the facts so the
judgement is by corroboration, not tautology.

Run from repo root (pytorch env not needed -- pure graph/text):
  PYTHONPATH=experiments python experiments/kgsage/cli/gen_neighbourhood_context.py \
      --csv experiments/kgsage/outputs/eval/gen_corruptions/FB15K-237_test_corruptions.csv \
      --data data/FB15K-237
"""
from __future__ import annotations
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

_EVAL_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "eval"

PREDICATE = {
    "people person place of birth": "place of birth",
    "people person nationality": "nationality",
    "people person profession": "profession",
    "film film genre": "genre",
    "film film language": "language",
    "film film country": "country of production",
    "music artist origin": "origin",
}


def _clean(name: str) -> str:
    name = re.sub(r"-GB\b", "", name)
    name = re.sub(r"\bLanguage\b", "", name).strip()
    return re.sub(r"\s{2,}", " ", name)


def _text_map(p: Path) -> dict:
    m = {}
    if p.is_file():
        for line in open(p, encoding="utf-8-sig"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                m[parts[0]] = parts[1]
    return m


def _load_triples(data: Path):
    out = []
    for split in ("train", "valid", "test"):
        p = data / f"{split}.txt"
        if p.is_file():
            for line in open(p, encoding="utf-8-sig"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    out.append(tuple(parts))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--which", default="corrupted",
                    choices=["corrupted", "control", "both"])
    ap.add_argument("--slot", default="tail", choices=["tail", "head", "both"],
                    help="which corruptions to include. tail (default) is the "
                         "meaningful case: anchor = the subject, whose own facts "
                         "support or contradict the new object. head corruptions "
                         "anchor on a hub object (a place/genre), where an absent "
                         "value reads as merely neutral, not contradicted.")
    ap.add_argument("--max_facts", type=int, default=20)
    ap.add_argument("--out", default=None,
                    help="default: outputs/eval/neighbourhood/<csv-stem>_<which>.txt")
    args = ap.parse_args()

    data = Path(args.data)
    ent_txt = _text_map(data / "entity2text.txt")
    rel_txt = _text_map(data / "relation2text.txt")
    triples = _load_triples(data)
    triple_set = set(triples)

    name = lambda e: _clean(ent_txt.get(e, e))
    def pred(rp):
        lab = rel_txt.get(rp, rp.rstrip("/").split("/")[-1])
        return PREDICATE.get(lab, lab)

    # name -> ids (raw + cleaned); relation label -> paths; per-entity fact index
    name2ids = defaultdict(set)
    for e, txt in ent_txt.items():
        name2ids[txt].add(e); name2ids[_clean(txt)].add(e)
    label2rels = defaultdict(set)
    facts = defaultdict(list)                       # entity -> [(dir, rel, other)]
    all_rels = set()
    for h, r, t in triples:
        all_rels.add(r)
        facts[h].append(("out", r, t))
        facts[t].append(("in", r, h))
    for e in facts:
        name2ids.setdefault(e, set()).add(e)
    for rp in all_rels:
        label2rels[rp].add(rp)
        label2rels[rp.rstrip("/").split("/")[-1]].add(rp)
        if rp in rel_txt:
            label2rels[rel_txt[rp]].add(rp)

    def resolve_orig(oh, orel, ot):
        for r in label2rels.get(orel, set()):
            for h in name2ids.get(oh, ()):
                for t in name2ids.get(ot, ()):
                    if (h, r, t) in triple_set:
                        return h, r, t
        return None

    def fact_lines(anchor, rel_priority=None, exclude=None):
        # Order: the anchor's OWN value(s) for the corrupted relation first (so
        # a functional contradiction is visible), then other simple relations,
        # then FB's reified CVT relations (a "." in the path -- mostly noise).
        seen, prio, simple, compound = set(), [], [], []
        for d, r, other in facts.get(anchor, []):
            if exclude is not None and (
                    (d == "out" and (anchor, r, other) == exclude) or
                    (d == "in" and (other, r, anchor) == exclude)):
                continue
            s = (f"  - {name(anchor)}, {pred(r)}, {name(other)}" if d == "out"
                 else f"  - {name(other)}, {pred(r)}, {name(anchor)}")
            if s in seen:
                continue
            seen.add(s)
            bucket = prio if r == rel_priority else (compound if "." in r else simple)
            bucket.append(s)
        return (prio + simple + compound)[:args.max_facts]

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))
    modes = (["corrupted", "control"] if args.which == "both" else [args.which])

    def build(mode):
        blocks, n = [], 0
        for r in rows:
            got = resolve_orig(r["orig_head"], r["orig_relation"], r["orig_tail"])
            if got is None:
                continue
            h, rp, t = got
            slot = "head" if r["corr_head"] != r["orig_head"] else "tail"
            if args.slot != "both" and slot != args.slot:
                continue
            anchor = h if slot == "tail" else t
            if mode == "corrupted":                       # assess the fake
                triple_str = f"({r['corr_head']}, {pred(rp)}, {r['corr_tail']})"
                lines = fact_lines(anchor, rel_priority=rp)   # keep the real edge (it contradicts)
            else:                                          # assess the true triple
                triple_str = f"({r['orig_head']}, {pred(rp)}, {r['orig_tail']})"
                lines = fact_lines(anchor, rel_priority=rp, exclude=(h, rp, t))  # drop its own edge
            if not lines:
                continue
            n += 1
            blocks.append(
                f"CASE {n}\nKnown facts about {name(anchor)}:\n"
                + "\n".join(lines)
                + f"\nAssess: {triple_str}\n")
        return blocks, n

    for mode in modes:
        blocks, n = build(mode)
        if args.out and len(modes) == 1:
            out = Path(args.out)
        else:
            stem = Path(args.csv).stem
            out = _EVAL_ROOT / "neighbourhood" / f"{stem}_{mode}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(blocks), encoding="utf-8")
        print(f"wrote {n} {mode} cases -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
