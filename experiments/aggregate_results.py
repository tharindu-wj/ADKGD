"""Aggregate per-run JSON records into mean±std tables, one per matrix cell.

run_experiment.py writes one `<model>_<dataset>_run.json` per run under
checkpoints/<dataset>/. This script groups them by
(dataset, neg_source, test_anomaly_source, anomaly_ratio) and reports
mean±std over seeds for AUC, AUPRC and the five P@K / R@K cutoffs.

A cell is named "train-neg x test-anom": the four cells are random x random
(baseline), random x gan, gan x random and gan x gan. `neg_source` /
`test_anomaly_source` and the values random|gan are FROZEN JSON keys copied
straight from the detector CLI contract, so they are read verbatim here;
'gan' means "corruptions produced by the trained KGSAGE generator", which the
detector then consumes as training negatives (neg_source) or as injected eval
anomalies (test_anomaly_source).

Usage (repo root):
    python experiments/aggregate_results.py [--root checkpoints] [--dataset X]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def fmt(vals: list[float | None]) -> str:
    xs = [v for v in vals if v is not None]
    if not xs:
        return "   --   "
    if len(xs) == 1:
        return f"{xs[0]:.4f}   "
    return f"{np.mean(xs):.4f}±{np.std(xs):.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="checkpoints")
    ap.add_argument("--dataset", default=None, help="filter to one dataset")
    args = ap.parse_args()

    records = []
    for path in sorted(Path(args.root).glob("**/*_run.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"!! skipping {path}: {exc}")
            continue
        if args.dataset and rec.get("dataset") != args.dataset:
            continue
        records.append(rec)
    if not records:
        print("no *_run.json records found")
        return 1

    cells: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        key = (rec["dataset"], rec["neg_source"], rec["test_anomaly_source"],
               rec["anomaly_ratio"])
        cells[key].append(rec)

    for (ds, neg, test, ratio), recs in sorted(cells.items()):
        seeds = sorted(r["seed"] for r in recs)
        ks = sorted(recs[0]["precision_at"], key=float)
        print(f"\n=== {ds} | train-neg={neg} | test-anom={test} | "
              f"ratio={ratio} | seeds={seeds} ===")
        print(f"  AUC   {fmt([r.get('auc') for r in recs])}    "
              f"AUPRC {fmt([r.get('auprc') for r in recs])}")
        print(f"  {'K':>8}  {'Precision@K':>16}  {'Recall@K':>16}")
        for k in ks:
            p = fmt([r["precision_at"].get(k) for r in recs])
            r_ = fmt([r["recall_at"].get(k) for r in recs])
            print(f"  {float(k) * 100:>7.1f}%  {p:>16}  {r_:>16}")
    print(f"\n{len(records)} runs in {len(cells)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
