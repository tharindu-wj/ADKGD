"""Spot-check (positive, negative) pairs extracted from a SLURM log.

Run after a B2 ADKGD run to verify the KGSAGE-generated negatives satisfy
the Category-5 contract:

  STEP 1: The negative is NOT in the real KG (no collision).
  STEP 2: The slot that changed matches what the [moved: ...] tag says.
  STEP 3: The new entity has at least one type in common with the original
          (type-coherence check via entity_types.tsv).
  STEP 4: The negative != positive (no self-equality).

Usage:
  python -m experiments.scripts.verify_negatives \\
      --log_file ~/ADKGD/adkgd_with_kgsage_fb15k237-NNNNNN.out.txt \\
      --dataset FB15K-237

Output: a small report. Failing pairs are listed for inspection.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# Regex to grab "pos: (h, r, t)" and "neg: (h, r, t) [moved: ...]" pairs.
_POS_RE = re.compile(r"pos:\s*\(([^,]+),\s*([^,]+),\s*([^)]+)\)")
_NEG_RE = re.compile(r"neg:\s*\(([^,]+),\s*([^,]+),\s*([^)]+)\)\s*\[moved:\s*([^\]]+)\]")


def parse_pairs(log_path):
    """Yield (positive, negative, moved_str) tuples from the SLURM log."""
    pos = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m_pos = _POS_RE.search(line)
            if m_pos:
                pos = (m_pos.group(1).strip(),
                       m_pos.group(2).strip(),
                       m_pos.group(3).strip())
                continue
            m_neg = _NEG_RE.search(line)
            if m_neg and pos is not None:
                neg = (m_neg.group(1).strip(),
                       m_neg.group(2).strip(),
                       m_neg.group(3).strip())
                moved = m_neg.group(4).strip()
                yield pos, neg, moved
                pos = None


def load_entity_types(types_tsv_path):
    """Return {entity: set of types} parsed from entity_types.tsv."""
    types = defaultdict(set)
    with open(types_tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                ent, _pred, tp = parts
                types[ent].add(tp)
    return dict(types)


def load_real_triples(*paths):
    """Return a set of (h, r, t) string-triples present in any split."""
    real = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    real.add(tuple(parts))
    return real


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log_file", required=True, help="SLURM .out.txt file")
    ap.add_argument("--dataset", default="FB15K-237")
    ap.add_argument("--limit", type=int, default=0,
                    help="only check the first N pairs (0 = all)")
    args = ap.parse_args()

    data_dir = _PROJECT_ROOT / "data" / args.dataset
    types_tsv = data_dir / "entity_types.tsv"
    if not types_tsv.exists():
        print(f"!! entity_types.tsv not found at {types_tsv}", file=sys.stderr)
        print(f"   Run Phase 1 first.", file=sys.stderr)
        return 1

    print(f"Loading entity types from {types_tsv} ...", flush=True)
    entity_types = load_entity_types(types_tsv)

    print(f"Loading real triples from {data_dir}/{{train,valid,test}}.txt ...",
          flush=True)
    real = load_real_triples(
        str(data_dir / "train.txt"),
        str(data_dir / "valid.txt"),
        str(data_dir / "test.txt"),
    )
    print(f"  {len(real):,} real triples loaded")

    print(f"Parsing pairs from {args.log_file} ...", flush=True)
    pairs = list(parse_pairs(args.log_file))
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"  {len(pairs)} (pos, neg) pairs to verify")
    print()

    counts = {
        "total": 0,
        "collision_with_real": 0,
        "self_equal": 0,
        "wrong_slot_tag": 0,
        "type_incoherent": 0,
        "ok": 0,
    }
    failures = []

    for pos, neg, moved in pairs:
        counts["total"] += 1

        # CHECK 1: neg not in real KG.
        if neg in real:
            counts["collision_with_real"] += 1
            failures.append(("REAL", pos, neg, moved))
            continue

        # CHECK 2: neg != pos.
        if neg == pos:
            counts["self_equal"] += 1
            failures.append(("SELF_EQUAL", pos, neg, moved))
            continue

        # CHECK 3: the slot tagged as moved actually differs.
        diffs = []
        if pos[0] != neg[0]:
            diffs.append("head")
        if pos[1] != neg[1]:
            diffs.append("relation")
        if pos[2] != neg[2]:
            diffs.append("tail")
        moved_slots = [s.strip() for s in moved.split(",") if s.strip()]
        if set(diffs) != set(moved_slots):
            counts["wrong_slot_tag"] += 1
            failures.append(("SLOT_TAG", pos, neg, moved))
            continue

        # CHECK 4: type-coherence on the moved slot.
        if "head" in moved_slots and pos[0] != neg[0]:
            old_types = entity_types.get(pos[0], set())
            new_types = entity_types.get(neg[0], set())
            if old_types and new_types and not (old_types & new_types):
                counts["type_incoherent"] += 1
                failures.append(("TYPE_HEAD", pos, neg, moved))
                continue
        if "tail" in moved_slots and pos[2] != neg[2]:
            old_types = entity_types.get(pos[2], set())
            new_types = entity_types.get(neg[2], set())
            if old_types and new_types and not (old_types & new_types):
                counts["type_incoherent"] += 1
                failures.append(("TYPE_TAIL", pos, neg, moved))
                continue

        counts["ok"] += 1

    # Report.
    print("=" * 60)
    print(f"  Verification Report  ({args.dataset})")
    print("=" * 60)
    total = counts["total"]
    if total == 0:
        print("  No pairs parsed - check the log format.")
        return 1
    print(f"  Total checked:        {total}")
    print(f"  PASS:                 {counts['ok']} ({counts['ok']/total*100:.1f}%)")
    print(f"  Collision with real:  {counts['collision_with_real']}")
    print(f"  Self-equal:           {counts['self_equal']}")
    print(f"  Wrong slot tag:       {counts['wrong_slot_tag']}")
    print(f"  Type-incoherent:      {counts['type_incoherent']}")
    print()

    if failures:
        print(f"First {min(10, len(failures))} failures for inspection:")
        for reason, pos, neg, moved in failures[:10]:
            print(f"  [{reason}]")
            print(f"    pos: {pos}")
            print(f"    neg: {neg}  [moved: {moved}]")

    return 0 if counts["ok"] == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
