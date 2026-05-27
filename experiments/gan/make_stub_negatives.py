"""Produce a placeholder gan_negatives_stub.tsv using ADKGD's own random corrupter.

This is a *stub* — it's not actually GAN-generated; it just reuses
generate_anomalous_triples_2 (the fully-random corruption baseline) so we can
prove the load_gan_negatives loader and downstream plumbing works end-to-end
BEFORE the real GAN exists.

Usage from the repo root:
    python experiments/gan/make_stub_negatives.py                              # defaults: --dataset dummy_kg --count 200
    python experiments/gan/make_stub_negatives.py --dataset FB15K --count 400000

Writes:
    data/<dataset>/gan_negatives_stub.tsv

Once the real GAN exists, you'll point --gan_neg_path at its output instead;
this stub script becomes a curiosity, not a runtime dependency.
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="dummy_kg",
                    help="dataset folder under data/ (default: dummy_kg — the tiny 18-fact KG for fast iteration)")
    ap.add_argument("--count", type=int, default=200,
                    help="how many stub anomalous triples to emit (200 is plenty for dummy_kg; bump to ~400000 for full FB15K)")
    ap.add_argument("--out", default=None,
                    help="output path; defaults to data/<dataset>/gan_negatives_stub.tsv")
    args = ap.parse_args()

    # Reader lives at the repo root; this script lives at experiments/gan/...
    repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo_root))

    from dataset import Reader  # noqa: E402  (intentional after sys.path tweak)

    data_dir = repo_root / "data" / args.dataset
    if not data_dir.exists():
        print("ERROR: dataset folder not found: %s" % data_dir, file=sys.stderr)
        return 2

    # Reader needs an anomaly_ratio (for inject_anomaly) but doesn't care about
    # the value -- we're not going to use anything that depends on it.
    fake_args = Namespace(anomaly_ratio=0.15)
    print("Reading %s ..." % data_dir)
    r = Reader(fake_args, str(data_dir) + "/")

    print("Generating %d random anomalous triples (stub stand-in for real GAN output)..." % args.count)
    triples = r.generate_anomalous_triples_2(args.count)

    out_path = Path(args.out) if args.out else (data_dir / "gan_negatives_stub.tsv")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for h, rel, t in triples:
            f.write("%s\t%s\t%s\n" % (r.id2ent[h], r.id2rel[rel], r.id2ent[t]))

    print("Wrote %d triples to %s" % (args.count, out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
