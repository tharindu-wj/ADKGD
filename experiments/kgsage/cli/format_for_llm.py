# -*- coding: utf-8 -*-
"""7.3 helper: turn a corruptions CSV into paste-ready triple blocks for the
blind LLM real-world evaluation.

Prints two blocks with clean relation predicates:
  CORRUPTED  -- the corruptions to judge (the main run)
  CONTROL    -- the matching true triples, for the control run (validates the
                judges; the control arm should come back mostly True, the
                corruptions mostly False)

The control arm judges the true triple each corruption was derived from, so a
judge that calls both blocks False has simply not read the facts.

Run from repo root:
  PYTHONPATH=experiments python experiments/kgsage/cli/format_for_llm.py \
      --csv experiments/kgsage/outputs/eval/gen_corruptions/FB15K-237_test_corruptions.csv
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

# clean predicate for the templated relations; falls back to the readable phrase
PREDICATE = {
    # --- FB15K-237 ---
    "people person place of birth": "place of birth",
    "people person nationality": "nationality",
    "people person profession": "profession",
    "film film genre": "genre",
    "film film language": "language",
    "film film country": "country of production",
    "music artist origin": "origin",
    # --- WN18RR (keys are the relation2text.txt labels) ---
    "hypernym": "is a kind of",
    "instance hypernym": "is an instance of",
    "member meronym": "has member",
    "has part": "has part",
    "derivationally related form": "is derivationally related to",
    "synset domain topic of": "belongs to the topic domain of",
    "member of domain region": "is the region domain of",
    "member of domain usage": "is the usage domain of",
    "also see": "is semantically related to",
    "verb group": "is in the same verb group as",
    "similar to": "is similar to",
}


def _pred(rel: str) -> str:
    return PREDICATE.get(rel.strip(), rel.strip())


def _lemma(name: str) -> str:
    """'spalacidae, mole rats' -> 'spalacidae'.

    WordNet labels are 'lemma, gloss' and the gloss itself contains commas, so
    printing them whole makes an (h, r, t) triple unreadable. Keeping the lemma
    restores the triple shape.

    WN18RR ONLY. Do not use on FB15K-237: ~1.2% of its labels legitimately
    contain a comma (e.g. 'University of California, Irvine'), and truncating
    those yields a DIFFERENT real entity.
    """
    return name.split(",")[0].strip()


def _block(rows, prefix, lemma_only=False):
    """Render one paste-ready block of (h, r, t) lines.

    `prefix` selects the frozen CSV column family: "corr" (here corr_ means
    CORRUPTED -- in the trainer log corr-pick instead means "corroborated") or
    "orig", the true triple the corruption was derived from.
    """
    out = []
    for r in rows:
        h, rel, t = r[f"{prefix}_head"], _pred(r[f"{prefix}_relation"]), r[f"{prefix}_tail"]
        if lemma_only:
            h, t = _lemma(h), _lemma(t)
        out.append(f"({h}, {rel}, {t})")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--which", default="both",
                    choices=["corrupted", "control", "both"])
    ap.add_argument("--lemma_only", action="store_true",
                    help="WN18RR ONLY: print just the lemma of each entity, "
                         "dropping the gloss after the first comma (WordNet "
                         "glosses contain commas and break the triple layout). "
                         "Do NOT use on FB15K-237 -- ~1.2%% of its labels "
                         "contain a real comma and would be truncated to a "
                         "different entity.")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))

    if args.which in ("corrupted", "both"):
        print("=" * 70)
        print(f"CORRUPTED  ({len(rows)} triples -- the main run)")
        print("=" * 70)
        print(_block(rows, "corr", args.lemma_only))
    if args.which in ("control", "both"):
        print()
        print("=" * 70)
        print(f"CONTROL  ({len(rows)} true triples -- the control run)")
        print("=" * 70)
        print(_block(rows, "orig", args.lemma_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
