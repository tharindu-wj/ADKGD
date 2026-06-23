"""Test 1.3 — dataset-level anti-symmetric predicate pair density audit.

This is the FIRST decision gate of Phase 1, and the cheapest one to run.
It tells us whether the dataset actually has the anti-symmetric structure
KGSAGE needs to learn from — *before* spending GPU time on encoder training.

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

DECISION GATE (overridable per dataset via kgsage.data.datasets):
  PASS if we find ≥ `antisym_min_pairs` anti-symmetric pairs with support ≥ MIN_SUPPORT.
  FAIL if we find very few — the dataset is too sparse for learned generation;
       pivot to a different dataset or to rule-mining.

Usage:
    python -m kgsage.cli.audit_dataset --dataset fb15k237
    python -m kgsage.cli.audit_dataset --dataset wn18rr
    python -m kgsage.cli.audit_dataset --dataset /path/to/custom_kg
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from kgsage.data.loaders import load_kg
from kgsage.data.datasets import resolve_dataset


# Thresholds — tweak in CONTRADICTION_SPEC.md, change here too.
MIN_SUPPORT = 100      # Need at least this many anchor triples per relation.
ANTISYM_RATIO = 0.01   # ratio below this = mutually exclusive on role-swap
SYM_RATIO = 0.5        # ratio above this  = symmetric, exclude as partner
DEFAULT_PASS_COUNT = 30   # default if dataset doesn't specify
DEFAULT_FAIL_COUNT = 10   # default if dataset doesn't specify


def audit(kg, dataset_name=None, pass_count=None, fail_count=None):
    """Run the density audit and return a results dict.

    Inputs:
      kg            : dict from load_kg() — the loaded knowledge graph
      dataset_name  : str — used only for the decision-reason text
      pass_count    : int — overrides DEFAULT_PASS_COUNT (resolve_dataset supplies it)
      fail_count    : int — overrides DEFAULT_FAIL_COUNT (resolve_dataset supplies it)

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
    pass_count = pass_count if pass_count is not None else DEFAULT_PASS_COUNT
    fail_count = fail_count if fail_count is not None else DEFAULT_FAIL_COUNT
    dataset_label = dataset_name or "this dataset"

    n_antisym = len(antisym)
    if n_antisym >= pass_count:
        decision = "PASS"
        reason = (
            f"Found {n_antisym} anti-symmetric relation pairs with support ≥ "
            f"{MIN_SUPPORT} and ratio < {ANTISYM_RATIO}. "
            f"{dataset_label} has enough signal; proceed to encoder training."
        )
    elif n_antisym >= fail_count:
        decision = "MARGINAL"
        reason = (
            f"Found {n_antisym} anti-symmetric pairs (need ≥ {pass_count} for clean PASS, "
            f"have ≥ {fail_count}). Proceed with caution; KGSAGE may have low "
            f"diversity. Consider trying a different dataset or relaxing thresholds."
        )
    else:
        decision = "FAIL"
        reason = (
            f"Only {n_antisym} anti-symmetric pairs found (need ≥ {fail_count}). "
            f"{dataset_label} is too sparse for learned generation. PIVOT: try a "
            f"different dataset (NELL-995, YAGO-4.5), or switch thesis to rule-mining."
        )

    return {
        "dataset_name": dataset_label,
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
            "pass_count": pass_count,
            "fail_count": fail_count,
        },
    }


def print_report(kg, results, top_k=20):
    """Pretty-print the audit results for human inspection."""
    id2rel = kg["id2rel"]
    thresholds = results["thresholds"]

    print("=" * 70)
    print(f"  KGSAGE Phase 1 Test 1.3 — Dataset Anti-Symmetric Pair Density Audit")
    print("=" * 70)
    print()
    print(f"  Dataset          : {results['dataset_name']}")
    print(f"  Training triples : {len(kg['triples_train']):>8,}")
    print(f"  Relations        : {kg['n_rel']:>8,}")
    print(f"  Entities         : {kg['n_ent']:>8,}")
    print()
    print(f"  Thresholds")
    print(f"    min support per relation        : {thresholds['min_support']}")
    print(f"    anti-symmetric ratio (max)      : {thresholds['antisym_ratio']}")
    print(f"    symmetric ratio (min)           : {thresholds['sym_ratio']}")
    print(f"    pass count (anti-sym pairs)     : {thresholds['pass_count']}")
    print(f"    fail count (anti-sym pairs)     : {thresholds['fail_count']}")
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="known short name (fb15k237, wn18rr, nell995, dummy_kg) "
                         "or a filesystem path to a directory containing "
                         "train.txt / valid.txt / test.txt")
    ap.add_argument("--out", default=None,
                    help="where to save the audit JSON (default: "
                         "experiments/kgsage/outputs/<dataset>_density_audit.json)")
    ap.add_argument("--top_k", type=int, default=20,
                    help="how many pairs to show in the printed report")
    args = ap.parse_args()

    # Resolve the dataset spec — accepts a short name or a path.
    config = resolve_dataset(args.dataset)

    # Default output path includes dataset name so multiple datasets can coexist.
    if args.out is None:
        args.out = f"experiments/kgsage/outputs/{config['name']}_density_audit.json"

    print(f"Loading {config['name']} from {config['path']}/...", flush=True)
    kg = load_kg(config["path"])
    print(f"  loaded {len(kg['triples_train']):,} train, "
          f"{len(kg['triples_valid']):,} valid, "
          f"{len(kg['triples_test']):,} test triples", flush=True)
    print()

    # Pass the dataset's per-dataset thresholds into the audit (resolve_dataset
    # gives None for unknown paths → audit uses the global DEFAULT_*_COUNTs).
    results = audit(
        kg,
        dataset_name=config["name"],
        pass_count=config.get("antisym_min_pairs"),
        fail_count=None,  # we keep the same FAIL threshold globally
    )
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
        "dataset_name": results["dataset_name"],
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
