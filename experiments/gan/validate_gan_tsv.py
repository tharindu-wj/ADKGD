"""Offline pre-flight validator for the GAN-generated negatives TSV.

Run this on the login node BEFORE submitting an HPC job, so vocabulary drift,
encoding issues, real-triple leakage, and pool-too-small problems are caught
in ~5 seconds at zero compute cost (instead of 15 minutes into a slurm run).

Mirrors the runtime logic in Reader.load_gan_negatives (dataset.py) but does
not import any heavy deps (no torch, no model code) -- works in any env that
has Python 3 and can read the data/ directory.

Usage from the repo root:
    python experiments/gan/validate_gan_tsv.py \
        --dataset FB15K \
        --tsv data/FB15K/gan_negatives.tsv

Exits 0 if "OK to use", 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Same constant set used by ADKGD's Reader: train + valid + test merged.
DATA_FILES = ("train.txt", "valid.txt", "test.txt")

# We need to support runs at up to this anomaly ratio without running out of
# usable pool entries (the slurm scripts target 0.05 today but the plan keeps
# the door open for 0.10 / 0.15).
MAX_ANOMALY_RATIO = 0.15

# Hard-coded vocabulary-drop tolerance: above this, the GAN is producing
# noticeable amounts of out-of-vocab triples and should be fixed.
MAX_UNKNOWN_VOCAB_FRACTION = 0.05


def _read_real_graph(data_dir: Path):
    """Return (ent_set, rel_set, triple_set) from train + valid + test merged.

    Mirrors Reader.read_triples() (dataset.py) closely enough for validation:
    we only need to know *what surface names exist* and *which (h,r,t) tuples
    are real*. We don't need ent2id ordering here.
    """
    ents = set()
    rels = set()
    triples = set()
    for fn in DATA_FILES:
        path = data_dir / fn
        if not path.exists():
            print("ERROR: missing dataset file: %s" % path, file=sys.stderr)
            sys.exit(2)
        with open(path, encoding='utf-8') as f:
            for raw in f:
                parts = raw.rstrip('\r\n').split('\t')
                if len(parts) != 3:
                    continue
                h, r, t = parts
                ents.add(h)
                ents.add(t)
                rels.add(r)
                triples.add((h, r, t))
    return ents, rels, triples


def validate(dataset: str, tsv_path: Path, data_root: Path) -> int:
    data_dir = data_root / dataset
    print("Validating %s against vocab from %s/" % (tsv_path, data_dir))
    ents, rels, real_triples = _read_real_graph(data_dir)
    num_original = len(real_triples)
    print("  real graph: %d entities, %d relations, %d triples\n" % (len(ents), len(rels), num_original))

    parsed = 0
    bad_format = 0
    skipped_unknown = 0
    skipped_original = 0
    unique = set()

    if not tsv_path.exists():
        print("ERROR: TSV not found: %s" % tsv_path, file=sys.stderr)
        return 2

    with open(tsv_path, encoding='utf-8') as f:
        for raw in f:
            parts = raw.rstrip('\r\n').split('\t')
            if len(parts) != 3:
                bad_format += 1
                continue
            parsed += 1
            h, r, t = parts
            if h not in ents or r not in rels or t not in ents:
                skipped_unknown += 1
                continue
            triple = (h, r, t)
            if triple in real_triples:
                skipped_original += 1
                continue
            unique.add(triple)

    valid = len(unique)
    total = parsed + bad_format

    needed = int(num_original * MAX_ANOMALY_RATIO) + 1
    pool_ok = valid >= needed
    unknown_ok = (skipped_unknown / parsed) <= MAX_UNKNOWN_VOCAB_FRACTION if parsed > 0 else False
    unique_ok = parsed == 0 or (valid / max(parsed - skipped_unknown - skipped_original, 1)) >= 0.95

    print("TSV: %d lines parsed (+ %d bad-format)" % (parsed, bad_format))
    print("   %d unique valid anomalous triples after filtering" % valid)
    print("   %d dropped (unknown vocab)         %s"
          % (skipped_unknown, "" if unknown_ok else "  <-- HIGH: %.1f%% of parsed lines" % (100 * skipped_unknown / max(parsed, 1))))
    print("   %d dropped (collide with real graph)" % skipped_original)
    print("   Pool sufficient for anomaly_ratio up to %.2f (need %d): %s"
          % (MAX_ANOMALY_RATIO, needed, "YES" if pool_ok else "NO"))
    print("   Unique-rate among in-vocab triples: %.1f%%   %s"
          % (100 * valid / max(parsed - skipped_unknown, 1), "OK" if unique_ok else "LOW (mode collapse?)"))

    all_ok = pool_ok and unknown_ok and unique_ok and bad_format == 0
    if all_ok:
        print("\n   Verdict: OK to use.")
        return 0
    else:
        print("\n   Verdict: NOT READY. Fix the GAN pipeline before sbatch:")
        if not pool_ok:
            print("     - pool too small (have %d valid, need >= %d for ratio %.2f)"
                  % (valid, needed, MAX_ANOMALY_RATIO))
        if not unknown_ok:
            print("     - too many out-of-vocab strings (likely whitespace/case/encoding drift)")
        if not unique_ok:
            print("     - too many duplicate triples (mode collapse in your GAN)")
        if bad_format > 0:
            print("     - %d lines did not have exactly 3 tab-separated fields" % bad_format)
        return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="FB15K",
                    help="dataset folder under data/ (default: FB15K)")
    ap.add_argument("--tsv", required=True,
                    help="path to the GAN-generated negatives TSV file")
    ap.add_argument("--data_root", default="data",
                    help="root data folder; default 'data' resolves relative to cwd")
    args = ap.parse_args()
    # Resolve relative to the repo root (this script lives at experiments/gan/...,
    # repo root is two levels up).
    repo_root = Path(__file__).resolve().parent.parent.parent
    tsv = Path(args.tsv)
    if not tsv.is_absolute():
        tsv = repo_root / tsv
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    return validate(args.dataset, tsv, data_root)


if __name__ == "__main__":
    raise SystemExit(main())
