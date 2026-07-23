# -*- coding: utf-8 -*-
"""7.3 helper: turn a corruptions CSV into paste-ready triple blocks for the
blind LLM real-world evaluation.

Prints two blocks with clean relation predicates:
  CORRUPTED  -- the triples to judge (the main run)
  ORIGINALS  -- the true triples, for the CONTROL run (validates the judges;
                originals should come back mostly True, corruptions mostly False)

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
    "people person place of birth": "place of birth",
    "people person nationality": "nationality",
    "people person profession": "profession",
    "film film genre": "genre",
    "film film language": "language",
    "film film country": "country of production",
    "music artist origin": "origin",
}


def _pred(rel: str) -> str:
    return PREDICATE.get(rel.strip(), rel.strip())


def _block(rows, prefix):
    out = []
    for r in rows:
        h, rel, t = r[f"{prefix}_head"], _pred(r[f"{prefix}_relation"]), r[f"{prefix}_tail"]
        out.append(f"({h}, {rel}, {t})")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--which", default="both",
                    choices=["corrupted", "original", "both"])
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))

    if args.which in ("corrupted", "both"):
        print("=" * 70)
        print(f"CORRUPTED  ({len(rows)} triples -- the main run)")
        print("=" * 70)
        print(_block(rows, "corr"))
    if args.which in ("original", "both"):
        print()
        print("=" * 70)
        print(f"ORIGINALS  ({len(rows)} triples -- the CONTROL run)")
        print("=" * 70)
        print(_block(rows, "orig"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
