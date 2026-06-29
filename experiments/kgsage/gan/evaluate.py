"""KGSAGE GAN generation-quality metrics — Phase 3.

Answers the Phase 3 RQ: are KGSAGE's generated contradictions meaningfully
BETTER than rule-only 2-cycle mining? It runs two pipelines on a held-out
anchor set and compares them head-to-head:

  A. KGSAGE    -- generate_kgsage_partners() on the trained checkpoint.
  B. rule-only -- sample the partner relation from the empirical template
                  distribution (mine_partner_templates), with the SAME
                  absent-from-graph rejection. This is the baseline the thesis
                  must beat to justify a learned adversarial generator.

Metrics (computed for BOTH pipelines, then compared):
  precision (Test 3.1)  fraction of partners that are non-trivially wrong --
                        absent from the full graph and not a self-loop.
  diversity (Test 3.3)  Shannon entropy (nats) of the partner-relation distribution.
  recall    (Test 3.2)  Hits@5: for held-out observed anti-symmetric (r, r')
                        templates, does the generator's top-5 partner set for
                        relation r include the true partner r'?
  mode_collapse         top-1 / top-5 partner-relation frequency share.

Head-to-head (the "is the GAN just the rule?" numbers):
  agreement             fraction of anchors where KGSAGE's top-1 partner equals
                        the rule's top-1 partner (1.0 => identical behaviour).
  js_divergence         Jensen-Shannon divergence (nats, in [0, ln2]) between the
                        two global partner-relation distributions. 0 => the GAN
                        reproduces the rule distribution exactly.

Decision gate (plan Phase 3): Test 3.1 AND (Test 3.2 OR Test 3.3) must pass,
AND KGSAGE must not sit statistically on top of the rule baseline (low
js_divergence + high agreement) if it is to claim a contribution beyond
"we automated rule mining".
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import torch

from kgsage.data.loaders import load_kg
from kgsage.gan.partner_templates import (
    mine_partner_templates,
    observed_2cycle_templates,
)
from kgsage.inference import load_kgsage_checkpoint, generate_kgsage_partners

__all__ = ["evaluate_quality", "report", "main"]


# ---------------------------------------------------------------------------
# Small metric helpers.
# ---------------------------------------------------------------------------
def _entropy_nats(rel_seq):
    """Shannon entropy (nats) of a sequence of partner-relation IDs."""
    counts = Counter(rel_seq)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts.values())


def _mode_share(rel_seq, k):
    """Fraction of output taken by the top-k most frequent partner relations."""
    counts = Counter(rel_seq)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    topk = sum(c for _, c in counts.most_common(k))
    return topk / total


def _js_divergence(seq_a, seq_b, n_rel):
    """Jensen-Shannon divergence (nats) between two partner-rel distributions."""
    def dist(seq):
        v = np.zeros(n_rel, dtype=np.float64)
        for r in seq:
            v[r] += 1.0
        s = v.sum()
        return v / s if s > 0 else v
    p, q = dist(seq_a), dist(seq_b)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _weighted_choice(weights, rng):
    """Sample an index proportional to weights (numpy Generator)."""
    total = sum(weights)
    if total <= 0:
        return rng.integers(len(weights))
    u = rng.random() * total
    cum = 0.0
    for i, w in enumerate(weights):
        cum += w
        if u <= cum:
            return i
    return len(weights) - 1


# ---------------------------------------------------------------------------
# Pipeline B: the rule-only baseline (sample partner from template distribution).
# ---------------------------------------------------------------------------
def rule_only_partners(anchors, templates, real_set, n_rel, rng, max_retries=10):
    """For each anchor (h, r, t), sample a partner r' from templates[r] by
    confidence and emit the role-swap (t, r', h), rejecting in-graph collisions.
    Anchors whose relation has no template fall back to a uniform absent r'.
    Returns (partners_list, stats) mirroring generate_kgsage_partners.
    """
    out = []
    stats = {"generated": 0, "fallbacks": 0, "skipped_selfloop": 0, "self_swap": 0}
    for (h, r, t) in anchors:
        if h == t:
            stats["skipped_selfloop"] += 1
            continue
        parts = templates.get(r)
        partner = None
        if parts:
            rels = [rp for rp, _ in parts]
            weights = [max(c, 1e-6) for _, c in parts]
            for _ in range(max_retries):
                rp = rels[_weighted_choice(weights, rng)]
                if (t, rp, h) not in real_set:
                    partner = (t, rp, h)
                    break
        if partner is None:
            for _ in range(200):
                rp = int(rng.integers(n_rel))
                if (t, rp, h) not in real_set:
                    partner = (t, rp, h)
                    break
            stats["fallbacks"] += 1
        if partner is None:
            continue
        if partner[1] == r:
            stats["self_swap"] += 1
        stats["generated"] += 1
        out.append(partner)
    return out, stats


# ---------------------------------------------------------------------------
# Top-k partner relations (for the Hits@5 recall test).
# ---------------------------------------------------------------------------
def _gan_topk(G, anchors, device, k=5):
    """Return, per anchor, the top-k partner relation IDs from the generator."""
    if not anchors:
        return []
    h = torch.tensor([a[0] for a in anchors], dtype=torch.long, device=device)
    r = torch.tensor([a[1] for a in anchors], dtype=torch.long, device=device)
    t = torch.tensor([a[2] for a in anchors], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = G(h, r, t)
    return logits.topk(min(k, logits.size(1)), dim=-1).indices.cpu().tolist()


def _gan_top1(G, anchors, device):
    """Return, per anchor, the argmax partner relation ID (deterministic)."""
    return [row[0] for row in _gan_topk(G, anchors, device, k=1)]


def _hits_at_k_recall(G, kg, held_out_pairs, device, rng, k=5, anchors_per_pair=20):
    """Test 3.2 for the GAN: for each held-out (r, r_true), sample anchors of
    relation r and check whether r_true appears in the top-k partner set.
    Returns mean hit-rate over held-out pairs."""
    by_rel = defaultdict(list)
    for (h, r, t) in kg["triples_train"]:
        by_rel[r].append((h, r, t))
    if not held_out_pairs:
        return 0.0
    hit_rates = []
    for (r, r_true) in held_out_pairs:
        pool = by_rel.get(r, [])
        if not pool:
            continue
        idx = rng.choice(len(pool), size=min(anchors_per_pair, len(pool)), replace=False)
        sample = [pool[i] for i in idx]
        topk = _gan_topk(G, sample, device, k=k)
        hits = sum(1 for row in topk if r_true in row)
        hit_rates.append(hits / len(sample))
    return float(np.mean(hit_rates)) if hit_rates else 0.0


def _hits_at_k_rule(templates, held_out_pairs, k=5):
    """Test 3.2 for the rule baseline: r_true in the top-k highest-confidence
    template partners for relation r."""
    if not held_out_pairs:
        return 0.0
    hits = 0
    for (r, r_true) in held_out_pairs:
        parts = sorted(templates.get(r, []), key=lambda x: -x[1])[:k]
        if r_true in {rp for rp, _ in parts}:
            hits += 1
    return hits / len(held_out_pairs)


# ---------------------------------------------------------------------------
# The battery.
# ---------------------------------------------------------------------------
def _pipeline_metrics(partners, anchors_considered, real_set):
    """precision / diversity / mode_collapse / self-swap for one pipeline."""
    rels = [p[1] for p in partners]
    valid = sum(1 for p in partners if p not in real_set and p[0] != p[2])
    self_swap = sum(1 for (a, p) in zip(anchors_considered, partners) if p[1] == a[1])
    n = max(len(partners), 1)
    return {
        "n_generated": len(partners),
        "precision": valid / n,                       # Test 3.1 (non-trivially wrong)
        "diversity_nats": _entropy_nats(rels),        # Test 3.3
        "distinct_partner_rels": len(set(rels)),
        "mode_top1_share": _mode_share(rels, 1),
        "mode_top5_share": _mode_share(rels, 5),
        "self_swap_share": self_swap / n,
        "cross_rel_share": 1.0 - self_swap / n,
    }


def evaluate_quality(checkpoint_path, data_dir, *, n_samples=1000, min_support=100,
                     seed=0, device=None, holdout_frac=0.2):
    """Run the full Phase 3 battery on a trained KGSAGE checkpoint.

    Returns a JSON-serialisable dict: {gan, rule, comparison, tests, meta}.
    """
    payload = load_kgsage_checkpoint(checkpoint_path, device=device)
    G, dev = payload["generator"], payload["device"]
    real_set = payload["real_triple_set"]
    n_rel = payload["n_rel"]

    kg = load_kg(data_dir)
    if kg["n_rel"] != n_rel or kg["n_ent"] != payload["n_ent"]:
        raise SystemExit(
            f"Vocab mismatch: checkpoint has {payload['n_ent']} ent / {n_rel} rel, "
            f"data has {kg['n_ent']} / {kg['n_rel']}. Use the dataset the GAN was trained on.")

    rng = np.random.default_rng(seed)

    # Templates: rule baseline (mine) + recall ground truth (observed anti-sym).
    templates_rule = mine_partner_templates(kg, min_support=min_support)
    templates_obs = observed_2cycle_templates(kg, min_support=min_support)

    # Held-out anchor set: prefer the test split (truly unseen), else train.
    anchor_pool = kg["triples_test"] or kg["triples_train"]
    idx = rng.choice(len(anchor_pool), size=min(n_samples, len(anchor_pool)), replace=False)
    anchors = [anchor_pool[i] for i in idx]
    anchors_ne = [a for a in anchors if a[0] != a[2]]   # non-self-loop anchors

    # Pipeline A: KGSAGE.
    gan_partners, _ = generate_kgsage_partners(anchors, payload, adkgd_maps=None, rng=rng)
    # Pipeline B: rule-only.
    rule_partners, _ = rule_only_partners(anchors, templates_rule, real_set, n_rel, rng)

    gan_m = _pipeline_metrics(gan_partners, anchors_ne, real_set)
    rule_m = _pipeline_metrics(rule_partners, anchors_ne, real_set)

    # Head-to-head: top-1 agreement + JS divergence of partner distributions.
    gan_top1 = _gan_top1(G, anchors_ne, dev)
    rule_top1 = []
    for (h, r, t) in anchors_ne:
        parts = sorted(templates_rule.get(r, []), key=lambda x: -x[1])
        rule_top1.append(parts[0][0] if parts else -1)
    agree = sum(1 for a, b in zip(gan_top1, rule_top1) if a == b and b != -1)
    matched = sum(1 for b in rule_top1 if b != -1)
    agreement = agree / max(matched, 1)
    js = _js_divergence([p[1] for p in gan_partners], [p[1] for p in rule_partners], n_rel)

    # Test 3.2 recall: hold out 20% of observed anti-symmetric (r, r') pairs.
    obs_pairs = [(r, rp) for r, parts in templates_obs.items() for rp, _ in parts]
    rng.shuffle(obs_pairs)
    n_hold = max(1, int(round(holdout_frac * len(obs_pairs)))) if obs_pairs else 0
    held = obs_pairs[:n_hold]
    gan_hits5 = _hits_at_k_recall(G, kg, held, dev, rng, k=5)
    rule_hits5 = _hits_at_k_rule(templates_rule, held, k=5)

    # Decision gates.
    t31 = gan_m["precision"] > 0.60 and gan_m["precision"] >= rule_m["precision"]
    t32 = gan_hits5 > 0.40 and gan_hits5 > rule_hits5
    t33 = gan_m["diversity_nats"] > 1.5 and gan_m["diversity_nats"] > rule_m["diversity_nats"]
    gate = t31 and (t32 or t33)

    return {
        "meta": {
            "checkpoint": checkpoint_path,
            "data_dir": data_dir,
            "n_anchors": len(anchors_ne),
            "n_rel": n_rel,
            "min_support": min_support,
            "held_out_pairs": len(held),
        },
        "gan": gan_m,
        "rule": rule_m,
        "comparison": {
            "agreement_top1": agreement,
            "js_divergence_nats": js,
            "gan_recall_hits5": gan_hits5,
            "rule_recall_hits5": rule_hits5,
        },
        "tests": {
            "3.1_precision_pass": bool(t31),
            "3.2_recall_pass": bool(t32),
            "3.3_diversity_pass": bool(t33),
            "decision_gate_pass": bool(gate),
        },
    }


def report(results):
    """Pretty-print the metrics for a PASS/MARGINAL/FAIL reading."""
    g, r, c, t, m = (results["gan"], results["rule"], results["comparison"],
                     results["tests"], results["meta"])
    line = "=" * 70
    print(line)
    print("  KGSAGE Phase 3 — Generation Quality (KGSAGE vs rule-only)")
    print(line)
    print(f"  checkpoint : {m['checkpoint']}")
    print(f"  anchors    : {m['n_anchors']:,} (held-out)   relations: {m['n_rel']}   "
          f"held-out pairs: {m['held_out_pairs']}")
    print()
    print(f"  {'metric':24s}{'KGSAGE':>12s}{'rule-only':>12s}{'winner':>10s}")
    print("  " + "-" * 56)

    def row(name, gv, rv, higher_better=True, fmt="{:.3f}"):
        if abs(gv - rv) < 1e-9:
            win = "tie"
        elif (gv > rv) == higher_better:
            win = "KGSAGE"
        else:
            win = "rule"
        print(f"  {name:24s}{fmt.format(gv):>12s}{fmt.format(rv):>12s}{win:>10s}")

    row("precision (3.1)", g["precision"], r["precision"])
    row("diversity nats (3.3)", g["diversity_nats"], r["diversity_nats"])
    row("recall Hits@5 (3.2)", c["gan_recall_hits5"], c["rule_recall_hits5"])
    row("distinct partner rels", g["distinct_partner_rels"], r["distinct_partner_rels"],
        fmt="{:.0f}")
    row("mode top-1 share", g["mode_top1_share"], r["mode_top1_share"], higher_better=False)
    row("self-swap share", g["self_swap_share"], r["self_swap_share"], higher_better=False)
    row("cross-rel share", g["cross_rel_share"], r["cross_rel_share"])
    print()
    print("  Head-to-head (is the GAN just the rule?):")
    print(f"    top-1 agreement with rule : {c['agreement_top1']:.1%}  "
          f"(100% => identical partner choices)")
    print(f"    JS divergence vs rule     : {c['js_divergence_nats']:.4f} nats  "
          f"(0 => reproduces the rule distribution)")
    print()
    print(line)
    verdict = "PASS" if t["decision_gate_pass"] else "FAIL"
    print(f"  DECISION GATE (3.1 AND (3.2 OR 3.3)) : {verdict}")
    print(line)
    print(f"    Test 3.1 precision  : {'PASS' if t['3.1_precision_pass'] else 'FAIL'}")
    print(f"    Test 3.2 recall     : {'PASS' if t['3.2_recall_pass'] else 'FAIL'}")
    print(f"    Test 3.3 diversity  : {'PASS' if t['3.3_diversity_pass'] else 'FAIL'}")
    print()
    if c["agreement_top1"] > 0.80 and c["js_divergence_nats"] < 0.05:
        print("  NOTE: KGSAGE is statistically indistinguishable from the rule baseline")
        print("        (high agreement + near-zero JS divergence). The learned generator")
        print("        currently adds little over rule mining — consider the KBGAN-style")
        print("        plausible-AND-absent objective to produce harder cross-relation")
        print("        contradictions, or narrow the thesis claim accordingly.")
    return results


def main():
    ap = argparse.ArgumentParser(description="KGSAGE Phase 3 generation-quality eval.")
    ap.add_argument("--data", required=True, help="Dataset directory (the one the GAN trained on).")
    ap.add_argument("--ckpt", required=True, help="Trained KGSAGE checkpoint (.pt).")
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--min_support", type=int, default=100,
                    help="Template support floor (use 1 for dummy_kg).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="Optional path to dump the metrics JSON.")
    args = ap.parse_args()

    results = evaluate_quality(
        args.ckpt, args.data, n_samples=args.n_samples,
        min_support=args.min_support, seed=args.seed, device=args.device,
    )
    report(results)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote metrics JSON to {args.out}")
    return 0 if results["tests"]["decision_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
