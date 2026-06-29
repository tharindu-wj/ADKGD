"""v2-A regression test / decision gate.

Confirms that the rule-mined partner-template miner
(gan.partner_templates.mine_partner_templates -- KG-agnostic, pure Python)
reproduces the trusted Test 1.3 anti-symmetric classification
(data.audit_dataset.audit -- observed 2-cycles, already PASSED on FB15K-237).

This is the v2-A analogue of Test 1.3: run it on any KG BEFORE building Phase 2
on the rule-mined positives. If it passes, the rule miner is a sound drop-in for
the observed miner and Phase 2 can train on either source identically.

    python -m kgsage.cli.verify_templates --dataset fb15k237
    python -m kgsage.cli.verify_templates --dataset wn18rr

PASS criteria (exit 0):
    anti-symmetry recall  >= 95%   (recovers the audit's anti-symmetric relations)
    symmetry exclusion    >= 95%   (no false-positive contradictions)
    inverse-pair recall   >= 80%   (excludes the audit's real-inverse traps)
"""
import argparse

from kgsage.data.loaders import load_kg
from kgsage.data.datasets import resolve_dataset
from kgsage.data.audit_dataset import audit
from kgsage.gan.partner_templates import (
    mine_partner_templates,
    _count_cooccurrence,
    MIN_SUPPORT,
    SYM_RATIO,
)


# Decision-gate thresholds for this test (distinct from the audit's thresholds).
PASS_RECALL = 0.95          # anti-symmetry recall
PASS_EXCLUSION = 0.95       # symmetry exclusion
PASS_INVERSE = 0.80         # inverse-pair recall


def compare(kg, dataset_name=None):
    """Run audit + miner and return a metrics dict."""
    # ── ground truth: the trusted observed-2-cycle audit ──
    a = audit(kg, dataset_name=dataset_name)
    audit_antisym = a["antisym_pairs"]
    audit_sym = a["sym_pairs"]

    audit_self_antisym = {p["r"] for p in audit_antisym if p["r"] == p["r_prime"]}
    audit_self_sym = {p["r"] for p in audit_sym if p["r"] == p["r_prime"]}
    audit_cross_sym = {(p["r"], p["r_prime"]) for p in audit_sym if p["r"] != p["r_prime"]}

    # ── candidate: the rule-mined templates ──
    templates = mine_partner_templates(kg)
    rel_with_template = set(templates.keys())

    # recompute the miner's inverse-pair set for the inverse check
    support, coocur = _count_cooccurrence(kg)
    miner_inverse = {
        (r, rp) for (r, rp), c in coocur.items()
        if r != rp and support[r] >= MIN_SUPPORT and c / support[r] >= SYM_RATIO
    }

    # ── metrics ──
    recovered = audit_self_antisym & rel_with_template
    recall = len(recovered) / len(audit_self_antisym) if audit_self_antisym else 1.0

    correctly_excluded = audit_self_sym - rel_with_template
    exclusion = len(correctly_excluded) / len(audit_self_sym) if audit_self_sym else 1.0
    wrongly_kept = audit_self_sym & rel_with_template

    caught_inv = audit_cross_sym & miner_inverse
    inv_recall = len(caught_inv) / len(audit_cross_sym) if audit_cross_sym else 1.0

    perfectly_antisym = {r for r in rel_with_template if r not in support or r not in
                         {rr for (rr, rp) in coocur if rr == rp}}
    # simpler: relations with a template that have no self-co-occurrence entry
    self_keys = {rr for (rr, rp) in coocur if rr == rp}
    perfectly_antisym = {r for r in rel_with_template if r not in self_keys}

    return {
        "audit": a,
        "audit_self_antisym": audit_self_antisym,
        "audit_self_sym": audit_self_sym,
        "audit_cross_sym": audit_cross_sym,
        "templates": templates,
        "rel_with_template": rel_with_template,
        "miner_inverse": miner_inverse,
        "recall": recall,
        "recovered": recovered,
        "exclusion": exclusion,
        "wrongly_kept": wrongly_kept,
        "inv_recall": inv_recall,
        "perfectly_antisym": perfectly_antisym,
    }


def print_report(kg, m):
    id2rel = kg["id2rel"]
    a = m["audit"]

    def sec(t):
        print("\n" + "=" * 72 + f"\n  {t}\n" + "=" * 72)

    sec("v1 audit (observed 2-cycles -- the trusted Test 1.3)")
    print(f"  decision              : {a['decision']}")
    print(f"  anti-symmetric pairs  : {a['n_antisym']}")
    print(f"  symmetric pairs       : {a['n_sym']}")
    print(f"    self anti-sym (r==r')  : {len(m['audit_self_antisym'])} relations")
    print(f"    self symmetric (r==r') : {len(m['audit_self_sym'])} relations")
    print(f"    cross symmetric (r!=r'): {len(m['audit_cross_sym'])} pairs (real inverses)")

    sec("v2-A rule miner")
    print(f"  relations emitting a template : {len(m['rel_with_template'])}")
    print(f"  real-inverse pairs excluded   : {len(m['miner_inverse'])}")

    sec("STEP 1 -- anti-symmetry recall")
    print(f"  audit self-anti-symmetric : {len(m['audit_self_antisym'])}")
    print(f"  recovered by miner        : {len(m['recovered'])}")
    print(f"  RECALL                    : {m['recall']:.1%}")

    sec("STEP 2 -- symmetry exclusion (trap avoidance)")
    print(f"  audit self-symmetric      : {len(m['audit_self_sym'])}")
    print(f"  wrongly kept              : {len(m['wrongly_kept'])}")
    print(f"  EXCLUSION RATE            : {m['exclusion']:.1%}")
    for r in list(m["wrongly_kept"])[:10]:
        print(f"    ! {id2rel[r]}")

    sec("STEP 3 -- inverse-pair exclusion")
    print(f"  audit real-inverse pairs  : {len(m['audit_cross_sym'])}")
    print(f"  flagged by miner          : {len(m['audit_cross_sym'] & m['miner_inverse'])}")
    print(f"  INVERSE-PAIR RECALL       : {m['inv_recall']:.1%}")

    sec("STEP 4 -- extra coverage beyond the observed audit")
    print(f"  relations with a template : {len(m['rel_with_template'])}")
    print(f"    also flagged by audit      : {len(m['audit_self_antisym'] & m['rel_with_template'])}")
    print(f"    NEVER co-occur (audit blind): {len(m['perfectly_antisym'])}")
    print(f"  -> the miner covers {len(m['perfectly_antisym'])} relations the observed")
    print(f"     audit cannot enumerate (zero reverse edges). This is the")
    print(f"     mechanism that makes inverse-removed KGs (YAGO) tractable.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="known short name (fb15k237, wn18rr, ...) or a path")
    args = ap.parse_args()

    config = resolve_dataset(args.dataset)
    print(f"Loading {config['name']} from {config['path']}/ ...", flush=True)
    kg = load_kg(config["path"])
    print(f"  {kg['n_ent']:,} entities, {kg['n_rel']} relations, "
          f"{len(kg['triples_train']):,} train triples")

    m = compare(kg, dataset_name=config["name"])
    print_report(kg, m)

    print("\n" + "=" * 72 + "\n  VERDICT\n" + "=" * 72)
    pr = m["recall"] >= PASS_RECALL
    pe = m["exclusion"] >= PASS_EXCLUSION
    pi = m["inv_recall"] >= PASS_INVERSE
    print(f"  anti-symmetry recall  >= {PASS_RECALL:.0%} : {'PASS' if pr else 'FAIL'}  ({m['recall']:.1%})")
    print(f"  symmetry exclusion    >= {PASS_EXCLUSION:.0%} : {'PASS' if pe else 'FAIL'}  ({m['exclusion']:.1%})")
    print(f"  inverse-pair recall   >= {PASS_INVERSE:.0%} : {'PASS' if pi else 'FAIL'}  ({m['inv_recall']:.1%})")
    print()
    if pr and pe and pi:
        print("  PASS -- v2-A rule miner reproduces the trusted Test 1.3 classification.")
        print("          Safe to build Phase 2 on rule-mined positives for this KG.")
        return 0
    print("  FAIL -- rule confidences disagree with observed pairs. Investigate")
    print("          thresholds (or supply symmetric_exclude_names on a")
    print("          one-directional KG) before committing to v2-A here.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
