# -*- coding: utf-8 -*-
"""7.4 consumer: render one ego graph per corruption in a CSV.

Reads a CSV produced by gen_corruptions_csv.py and calls ego_viz.py for each
row, so the ego figures come from the SAME corruption set the LLM judges (7.3).

By default renders the best exemplars first (tail-slot, zero shared neighbours,
moderate anchor degree so the graph is legible), capped by --limit.

Run from repo root (pytorch env):
  PYTHONPATH=experiments python experiments/kgsage/cli/ego_from_csv.py \
      --csv experiments/kgsage/outputs/eval/fb_corruptions.csv \
      --data data/FB15K-237 --out_dir experiments/kgsage/outputs/eval/ego --limit 6
"""
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=6,
                    help="max ego graphs to render (0 = all rows)")
    ap.add_argument("--all_rows", action="store_true",
                    help="render every row in file order instead of ranking exemplars")
    ap.add_argument("--deg_min", type=int, default=6)
    ap.add_argument("--deg_max", type=int, default=45)
    ap.add_argument("--ext", default="png", choices=["png", "pdf"])
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))
    if not args.all_rows:
        # rank: clearest contradiction first (tail slot, 0 shared, legible degree)
        def key(r):
            zero = (int(r["shared_neighbours"]) == 0 and r["direct_neighbour"] == "0")
            deg = int(r["anchor_degree"])
            legible = args.deg_min <= deg <= args.deg_max
            return (r["slot"] != "tail", not zero, not legible, deg)
        rows = sorted(rows, key=key)
    if args.limit:
        rows = rows[:args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ego_viz = "experiments/kgsage/cli/ego_viz.py"
    e2t = str(Path(args.data) / "entity2text.txt")

    ok = 0
    for r in rows:
        stem = f"ego_{r['idx']}_{r['relation'].split('/')[-1]}"
        out = str(out_dir / f"{stem}.{args.ext}")
        cmd = [sys.executable, ego_viz, "--data", args.data,
               "--orig", r["orig_h_id"], r["orig_r_id"], r["orig_t_id"],
               "--corr", r["corr_h_id"], r["corr_r_id"], r["corr_t_id"],
               "--out", out, "--edge-labels", "--short-relations",
               "--entity2text", e2t]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            ok += 1
            print(f"OK  {out}   {r['corr_statement']}")
        else:
            print(f"FAIL {stem}: {res.stderr.strip().splitlines()[-1:]}", file=sys.stderr)

    print(f"\nrendered {ok}/{len(rows)} ego graphs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
