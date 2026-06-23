"""Test 1.3 — anti-symmetric predicate pair density audit on FB15K-237.

This is the FIRST decision gate of Phase 1, and the cheapest one to run.
It tells us whether FB15K-237 actually has the anti-symmetric structure
KGSAGE needs to learn from.

WHY THIS MATTERS:
  FB15K-237 was deliberately constructed by Toutanova-Chen 2015 to remove
  inverse relations from FB15K, preventing test leakage. That removes the
  most obvious anti-symmetric supervision signal — direct inverse pairs.

  But the AnyBURL paper (Meilicke 2019 IJCAI) showed that 2-cycle and
  inverse-like REGULARITIES survive in the training set even though the
  test set is constructed to be hard for them. KGSAGE depends on this:
  we need enough mutually-exclusive predicate pairs (r, r') such that
  r(a, b) AND r'(b, a) almost never co-occur — these are the contradiction
  pair templates the KGSAGE Generator will learn.

WHAT WE COMPUTE:
  For every ordered relation pair (r, r'):
    support(r)       = |{(h, t) : (h, r, t) ∈ train}|
    coocur(r, r')    = |{(h, t) : (h, r, t) ∈ train AND (t, r', h) ∈ train}|
    ratio(r, r')     = coocur(r, r') / support(r)

  An "anti-symmetric pair" is one with:
    support(r) ≥ MIN_SUPPORT  (we need enough data to learn from)
    ratio(r, r') < ANTISYM_RATIO  (the swap almost never co-occurs)

  A "symmetric pair" is one with:
    support(r) ≥ MIN_SUPPORT
    ratio(r, r') > SYM_RATIO  (the swap almost always co-occurs)

DECISION GATE:
  PASS if we find ≥ 30 anti-symmetric pairs with support(r) ≥ 100.
  FAIL if we find < 10 — the data is too sparse for learned generation;
       pivot to NELL-995 or rule-mining.

Usage:
    python -m experiments.kgsage.audit_density --data data/FB15K-237
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from .data import load_fb15k237


# Thresholds — tweak in CONTRADICTION_SPEC.md, change here too.
MIN_SUPPORT = 100      # Need at least this many anchor triples per relation.
ANTISYM_RATIO = 0.01   # ratio below this = mutually exclusive on role-swap
SYM_RATIO = 0.5        # ratio above this  = symmetric, exclude as partner
PASS_COUNT = 30        # ≥ this many anti-sym pairs → green light
FAIL_COUNT = 10        # < this many anti-sym pairs → red flag, pivot


def audit(kg):
    """Run the density audit and return a results dict.

    Returns:
        dict with:
          relation_support     : {r_id: count} — total anchor triples per relation
          coocurrence_counts   : {(r1, r2): count} — sparse, only non-zero entries
          antisym_pairs        : list of {r1, r2, support, coocur, ratio}, sorted by ratio
          sym_pairs            : list of same shape, sorted by ratio descending
          decision              : "PASS" / "MARGINAL" / "FAIL"
          decision_reason      : human-readable explanation
    """
    triples_train = kg["triples_train"]
    n_rel = kg["n_rel"]

    # ─── Step 1: count support per relation ─────────────────────────────
    # support(r) = number of (h, t) pairs where (h, r, t) is in train.
    relation_support = defaultdict(int)
    for h, r, t in triples_train:
        relation_support[r] += 1

    # ─── Step 2: build (h, t) → set of relations that go h→t ────────────
    # We need this so we can answer "given (h, r, t), what relations exist
    # going from t back to h?". Direct dict lookup beats O(n_triples) scan.
    edges_by_pair = defaultdict(set)  # (h, t) -> {r1, r2, ...}
    for h, r, t in triples_train:
        edges_by_pair[(h, t)].add(r)

    # ─── Step 3: count co-occurrence per ordered pair (r, r') ────────────
    # For each anchor triple (h, r, t), look up what relations go from
    # t back to h. Each such r' contributes to coocurrence_counts[(r, r')].
    coocurrence_counts = defaultdict(int)  # (r, r') -> count
    for h, r, t in triples_train:
        # Look up relations going t -> h (the role-swap direction).
        reverse_relations = edges_by_pair.get((t, h), set())
        for r_prime in reverse_relations:
            coocurrence_counts[(r, r_prime)] += 1

    # ─── Step 4: classify pairs as anti-symmetric / symmetric ────────────
    # Only consider pairs where the anchor relation r has enough support.
    # Self-pairs (r == r') are interesting too: a self-anti-symmetric
    # relation is one where r(a,b) implies NOT r(b,a) — e.g. hasChild.
    antisym = []
    sym = []
    for (r, r_prime), coocur in coocurrence_counts.items():
        support_r = relation_support[r]
        if support_r < MIN_SUPPORT:
            continue
        ratio = coocur / support_r
        record = {
            "r": r,
            "r_prime": r_prime,
            "support_r": support_r,
            "coocur": coocur,
            "ratio": ratio,
        }
        if ratio < ANTISYM_RATIO:
            antisym.append(record)
        elif ratio > SYM_RATIO:
            sym.append(record)

    # Also include relations that NEVER appear in a 2-cycle (pure anti-sym).
    # These are pairs (r, r') with coocur = 0 — they don't appear in the
    # dict above because of the sparse storage. We treat the most common
    # case explicitly: for each relation r with enough support, every r'
    # NOT in coocurrence_counts[(r, *)] is a "fully anti-symmetric" partner.
    # We skip enumerating all such pairs (would be O(n_rel^2)) and just
    # tag this as a note in the decision_reason.

    # ─── Step 5: sort outputs for easy human inspection ──────────────────
    antisym.sort(key=lambda x: x["ratio"])  # lowest ratio first
    sym.sort(key=lambda x: x["ratio"], reverse=True)  # highest ratio first

    # ─── Step 6: decision gate ──────────────────────────────────────────
    n_antisym = len(antisym)
    if n_antisym >= PASS_COUNT:
        decision = "PASS"
        reason = (
            f"Found {n_antisym} anti-symmetric relation pairs with support ≥ "
            f"{MIN_SUPPORT} and ratio < {ANTISYM_RATIO}. "
            f"FB15K-237 has enough signal; proceed to encoder training."
        )
    elif n_antisym >= FAIL_COUNT:
        decision = "MARGINAL"
        reason = (
            f"Found {n_antisym} anti-symmetric pairs (need ≥ {PASS_COUNT} for clean PASS, "
            f"have ≥ {FAIL_COUNT}). Proceed with caution; KGSAGE may have low "
            f"diversity. Consider adding NELL-995 or relaxing thresholds."
        )
    else:
        decision = "FAIL"
        reason = (
            f"Only {n_antisym} anti-symmetric pairs found (need ≥ {FAIL_COUNT}). "
            f"FB15K-237 inverse removal was too aggressive. PIVOT: try NELL-995, "
            f"add WN18RR, or switch thesis to rule-mining without KGSAGE."
        )

    return {
        "relation_support": dict(relation_support),
        "antisym_pairs": antisym,
        "sym_pairs": sym,
        "n_antisym": n_antisym,
        "n_sym": len(sym),
        "decision": decision,
        "decision_reason": reason,
        "thresholds": {
            "min_support": MIN_SUPPORT,
            "antisym_ratio": ANTISYM_RATIO,
            "sym_ratio": SYM_RATIO,
            "pass_count": PASS_COUNT,
            "fail_count": FAIL_COUNT,
        },
    }


def print_report(kg, results, top_k=20):
    """Pretty-print the audit results for human inspection."""
    id2rel = kg["id2rel"]

    print("=" * 70)
    print(f"  KGSAGE Phase 1 Test 1.3 — Anti-Symmetric Predicate Pair Density Audit")
    print("=" * 70)
    print()
    print(f"  Training triples : {len(kg['triples_train']):>8,}")
    print(f"  Relations        : {kg['n_rel']:>8,}")
    print(f"  Entities         : {kg['n_ent']:>8,}")
    print()
    print(f"  Thresholds")
    print(f"    min support per relation        : {MIN_SUPPORT}")
    print(f"    anti-symmetric ratio (max)      : {ANTISYM_RATIO}")
    print(f"    symmetric ratio (min)           : {SYM_RATIO}")
    print()
    print(f"  Found")
    print(f"    anti-symmetric pairs            : {results['n_antisym']:>4}")
    print(f"    symmetric pairs                 : {results['n_sym']:>4}")
    print()
    print(f"  ── Top {top_k} anti-symmetric pairs (lowest ratio first) ──")
    for record in results["antisym_pairs"][:top_k]:
        r_str = _short(id2rel[record["r"]])
        rp_str = _short(id2rel[record["r_prime"]])
        print(f"    {record['ratio']:.4f}  "
              f"sup={record['support_r']:>5}  "
              f"coocur={record['coocur']:>4}  "
              f"r={r_str}  r'={rp_str}")
    print()
    print(f"  ── Top {top_k} symmetric pairs (highest ratio first) ──")
    for record in results["sym_pairs"][:top_k]:
        r_str = _short(id2rel[record["r"]])
        rp_str = _short(id2rel[record["r_prime"]])
        print(f"    {record['ratio']:.4f}  "
              f"sup={record['support_r']:>5}  "
              f"coocur={record['coocur']:>4}  "
              f"r={r_str}  r'={rp_str}")
    print()
    print("=" * 70)
    print(f"  DECISION: {results['decision']}")
    print("=" * 70)
    print(f"  {results['decision_reason']}")
    print()


def _short(rel_string, max_len=50):
    """Truncate long relation strings for tidy printing."""
    if len(rel_string) <= max_len:
        return rel_string
    return rel_string[: max_len - 3] + "..."


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/FB15K-237",
                    help="dataset folder under data/")
    ap.add_argument("--out", default="experiments/kgsage/outputs/density_audit.json",
                    help="where to save the audit results JSON")
    ap.add_argument("--top_k", type=int, default=20,
                    help="how many pairs to show in the printed report")
    args = ap.parse_args()

    print(f"Loading {args.data}/...", flush=True)
    kg = load_fb15k237(args.data)
    print(f"  loaded {len(kg['triples_train']):,} train, "
          f"{len(kg['triples_valid']):,} valid, "
          f"{len(kg['triples_test']):,} test triples", flush=True)
    print()

    results = audit(kg)
    print_report(kg, results, top_k=args.top_k)

    # ─── Save results to JSON for Test 1.2 to consume ────────────────────
    # Test 1.2 (anti-symmetric signal in relation embeddings) uses the
    # output of this audit to choose which relation pairs to test.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # We add the relation STRINGS to the saved records so Test 1.2 can
    # look them up without re-running the audit.
    id2rel = kg["id2rel"]
    save_payload = {
        "decision": results["decision"],
        "decision_reason": results["decision_reason"],
        "n_antisym": results["n_antisym"],
        "n_sym": results["n_sym"],
        "thresholds": results["thresholds"],
        "antisym_pairs": [
            {**rec, "r_str": id2rel[rec["r"]], "r_prime_str": id2rel[rec["r_prime"]]}
            for rec in results["antisym_pairs"]
        ],
        "sym_pairs": [
            {**rec, "r_str": id2rel[rec["r"]], "r_prime_str": id2rel[rec["r_prime"]]}
            for rec in results["sym_pairs"]
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)
    print(f"Saved audit to {out_path}")

    return 0 if results["decision"] in ("PASS", "MARGINAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
