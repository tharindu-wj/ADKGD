"""Test 1.2 — Anti-symmetric signal in the trained relation embeddings.

This is the critical Phase 1 test. Test 1.1 (link prediction MRR) tells
us the encoder LEARNED SOMETHING; Test 1.2 tells us it learned the
RIGHT thing — namely, that the relation embedding space encodes
anti-symmetric structure.

WHY IT MATTERS:
  KGSAGE Phase 2's Generator takes a relation embedding r and produces
  a distribution over partner relations r'. If the embedding space
  doesn't distinguish "symmetric pairs" (e.g. colleagueOf with itself)
  from "anti-symmetric pairs" (e.g. marriedTo + hasChild), the Generator
  can't learn the partner distribution we need — it'll mix them up
  and generate false-positive "contradictions" from symmetric relations.

  Conversely, if there IS a signal, we can verify it before investing
  6 more weeks in Phase 2-5.

WHAT WE TEST:
  Hypothesis (alternative): for two relations r1 and r2 that have very
  different roles (anti-symmetric on role-swap), their embeddings should
  point in DIFFERENT directions — lower cosine similarity.
  For two relations r1 and r2 that play similar roles (symmetric), their
  embeddings should be similar — higher cosine similarity.

  We measure this with a Mann-Whitney U test, comparing the cosine
  similarity distributions of anti-symmetric pairs vs symmetric pairs.

DECISION GATE:
  PASS    : Mann-Whitney U test p < 0.05, anti-symmetric pairs have
            statistically lower cosine similarity than symmetric pairs.
  FAIL    : p ≥ 0.05 OR the direction is wrong.
            ESCALATE: try CompGCN (Plan B), longer training, or pivot.

DATA SOURCE:
  We use the output of audit_dataset.py (Test 1.3) to identify the
  anti-symmetric and symmetric pairs. This avoids hand-curation and
  makes the test data-driven.

Usage:
    python -m kgsage.cli.audit_embeddings --dataset fb15k237 \\
        --ckpt experiments/kgsage/outputs/fb15k237_encoder.pt \\
        --audit experiments/kgsage/outputs/fb15k237_density_audit.json
"""
import argparse
import json

import torch
import torch.nn.functional as F

from kgsage.data.datasets import resolve_dataset
from kgsage.encoder.models import KGSAGELinkPredictor


# Decision-gate threshold from the thesis plan.
ALPHA = 0.05
SAMPLE_SIZE = 20  # use top-20 anti-sym and top-20 sym pairs


def cosine_similarity(a, b):
    """Cosine similarity between two 1-D tensors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()


def mann_whitney_u(group1, group2):
    """Mann-Whitney U test — non-parametric "are these two samples different?"

    Returns:
        U statistic, two-tailed p-value (approximate via normal).

    We do this from scratch (rather than via scipy.stats) to avoid an
    extra dependency. For our sample sizes (n1 = n2 = ~20) the normal
    approximation is fine.

    The test ranks all values together, then sums the ranks of one group.
    Under the null (groups have the same distribution), the rank sum
    follows a known distribution.
    """
    n1 = len(group1)
    n2 = len(group2)
    combined = [(v, 0) for v in group1] + [(v, 1) for v in group2]
    combined.sort(key=lambda x: x[0])

    # Average rank for ties — the standard correction.
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-indexed
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    rank_sum_1 = sum(r for r, (_, g) in zip(ranks, combined) if g == 0)

    # U statistic for group 1.
    u1 = rank_sum_1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u_stat = min(u1, u2)

    # Normal approximation for p-value.
    # mean and std of U under H0:
    mu_u = n1 * n2 / 2.0
    sigma_u = (n1 * n2 * (n1 + n2 + 1) / 12.0) ** 0.5
    if sigma_u == 0:
        return u_stat, 1.0
    z = (u_stat - mu_u) / sigma_u

    # Two-tailed p-value via normal CDF.
    # We use a basic implementation of the CDF (math.erf is in stdlib).
    import math
    cdf_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p_value = 2.0 * min(cdf_z, 1.0 - cdf_z)
    return u_stat, p_value


def main():
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="known short name (fb15k237, wn18rr, nell995, dummy_kg) "
                         "or a filesystem path to a directory")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path (default: "
                         "experiments/kgsage/outputs/<dataset>_encoder.pt)")
    ap.add_argument("--audit", default=None,
                    help="dataset audit JSON path (default: "
                         "experiments/kgsage/outputs/<dataset>_density_audit.json)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample_size", type=int, default=SAMPLE_SIZE,
                    help="how many top pairs from each group to test")
    args = ap.parse_args()

    # Resolve dataset name → config dict (gives us the canonical short name).
    config = resolve_dataset(args.dataset)
    ckpt_path = args.ckpt or f"experiments/kgsage/outputs/{config['name']}_encoder.pt"
    audit_path = args.audit or f"experiments/kgsage/outputs/{config['name']}_density_audit.json"

    if not Path(ckpt_path).exists():
        print(f"!! checkpoint not found: {ckpt_path}", file=sys.stderr)
        print(f"   run `python -m kgsage.cli.train_encoder --dataset {args.dataset}` first.",
              file=sys.stderr)
        return 1

    if not Path(audit_path).exists():
        print(f"!! audit results not found: {audit_path}", file=sys.stderr)
        print(f"   run `python -m kgsage.cli.audit_dataset --dataset {args.dataset}` first.",
              file=sys.stderr)
        return 1

    # ─── Load the trained model ────────────────────────────────────────
    print(f"Loading checkpoint {ckpt_path}...", flush=True)
    device = torch.device(args.device)
    model, ent2id, rel2id = KGSAGELinkPredictor.load_pretrained(
        ckpt_path, device=device,
    )

    # The relation embedding table is the artifact we test.
    rel_emb = model.decoder.rel_emb.weight.detach()
    print(f"  relation embedding shape: {tuple(rel_emb.shape)}", flush=True)

    # ─── Load the audit results ────────────────────────────────────────
    print(f"Loading audit results from {audit_path}...", flush=True)
    with open(audit_path, encoding="utf-8") as f:
        audit = json.load(f)

    antisym_pairs = audit["antisym_pairs"][:args.sample_size]
    sym_pairs = audit["sym_pairs"][:args.sample_size]

    if len(antisym_pairs) < 5 or len(sym_pairs) < 5:
        print(f"!! Audit found too few pairs to test: "
              f"{len(antisym_pairs)} anti-symmetric, {len(sym_pairs)} symmetric.",
              file=sys.stderr)
        print(f"   Re-run `python -m kgsage.cli.audit_dataset --dataset {args.dataset}` "
              f"and check thresholds.", file=sys.stderr)
        return 1

    print(f"  using top {len(antisym_pairs)} anti-symmetric, "
          f"top {len(sym_pairs)} symmetric pairs", flush=True)
    print()

    # ─── Compute cosine similarities ───────────────────────────────────
    # For each pair (r, r'), cosine(rel_emb[r], rel_emb[r']).
    antisym_sims = []
    for rec in antisym_pairs:
        sim = cosine_similarity(rel_emb[rec["r"]], rel_emb[rec["r_prime"]])
        antisym_sims.append(sim)

    sym_sims = []
    for rec in sym_pairs:
        sim = cosine_similarity(rel_emb[rec["r"]], rel_emb[rec["r_prime"]])
        sym_sims.append(sim)

    mean_antisym = sum(antisym_sims) / len(antisym_sims)
    mean_sym = sum(sym_sims) / len(sym_sims)

    # ─── Statistical test ──────────────────────────────────────────────
    u_stat, p_value = mann_whitney_u(antisym_sims, sym_sims)

    # ─── Report ────────────────────────────────────────────────────────
    print("=" * 70)
    print("  KGSAGE Phase 1 Test 1.2 — Anti-Symmetric Signal Extraction")
    print("=" * 70)
    print(f"  Anti-symmetric pairs tested  : {len(antisym_sims)}")
    print(f"  Symmetric pairs tested       : {len(sym_sims)}")
    print()
    print(f"  Mean cosine similarity (anti-symmetric pairs)  : {mean_antisym:+.4f}")
    print(f"  Mean cosine similarity (symmetric pairs)       : {mean_sym:+.4f}")
    print()
    print(f"  Mann-Whitney U statistic                       : {u_stat:.2f}")
    print(f"  Two-tailed p-value                             : {p_value:.6f}")
    print()

    # We need TWO conditions for PASS:
    #   1. p_value < alpha (statistically significant difference)
    #   2. anti-symmetric pairs have LOWER cosine similarity than symmetric
    #      (the direction we expected)
    direction_ok = mean_antisym < mean_sym
    significant = p_value < ALPHA

    if significant and direction_ok:
        verdict = "PASS"
        reason = (f"p = {p_value:.6f} < {ALPHA} AND mean(anti-sym) < mean(sym). "
                  f"Relation embedding space DOES encode anti-symmetric structure. "
                  f"Phase 1 is complete — proceed to Phase 2 (KGSAGE Generator).")
    elif not direction_ok:
        verdict = "FAIL"
        reason = (f"Anti-symmetric pairs have HIGHER cosine similarity than "
                  f"symmetric pairs — direction is REVERSED. The encoder hasn't "
                  f"learned the structure we need. "
                  f"Try: longer training, ConvE decoder, or escalate to CompGCN.")
    else:
        verdict = "FAIL"
        reason = (f"p = {p_value:.6f} ≥ {ALPHA}. No statistically significant "
                  f"difference between anti-symmetric and symmetric cosine "
                  f"similarities. The signal is too weak. "
                  f"Try: longer training, ConvE decoder, or escalate to CompGCN (+2 weeks).")

    print("=" * 70)
    print(f"  DECISION: {verdict}")
    print("=" * 70)
    print(f"  {reason}")

    # ─── Detail dump for forensic analysis ─────────────────────────────
    print()
    print("── Detail: per-pair cosine similarities ──")
    print(f"\n  Anti-symmetric pairs (expected: low similarity)")
    for rec, sim in sorted(zip(antisym_pairs, antisym_sims),
                          key=lambda x: x[1]):
        r_str = _short(rec["r_str"])
        rp_str = _short(rec["r_prime_str"])
        print(f"    cos={sim:+.4f}  r={r_str}   r'={rp_str}")

    print(f"\n  Symmetric pairs (expected: high similarity)")
    for rec, sim in sorted(zip(sym_pairs, sym_sims),
                          key=lambda x: x[1], reverse=True):
        r_str = _short(rec["r_str"])
        rp_str = _short(rec["r_prime_str"])
        print(f"    cos={sim:+.4f}  r={r_str}   r'={rp_str}")

    return 0 if verdict == "PASS" else 1


def _short(rel_string, max_len=45):
    """Truncate long relation strings for tidy printing."""
    if len(rel_string) <= max_len:
        return rel_string
    return rel_string[: max_len - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
