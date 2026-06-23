"""Test 1.1 — Link prediction MRR on the test set.

This is the sanity check after training. If MRR is too low we know
something went wrong (data loading bug, optimisation bug, vanishing
gradients) before we move on to the more subtle Test 1.2.

DECISION GATE (per-dataset, overridable via kgsage.data.datasets):
  PASS    : MRR ≥ `expected_mrr` for the dataset (default: 0.30)
  WARN    : 0.66 × expected_mrr ≤ MRR < expected_mrr
  FAIL    : MRR < 0.66 × expected_mrr (something is wrong; debug before Phase 2)

WHAT MRR MEANS:
  For every test triple (h, r, t), we ask the model:
    "Given (h, r, ?), rank all entities by how plausibly they could be t."
    "Given (?, r, t), rank all entities by how plausibly they could be h."
  We take the rank of the true entity in each case (smaller = better).
  Reciprocal rank = 1 / rank. MRR is the mean across all test triples.

  An MRR of 0.30 on FB15K-237 means on average the true answer is somewhere
  around rank 3 — out of ~14,541 possible entities. That's solid.

FILTERED MRR:
  When ranking, we don't count OTHER known true triples as "ahead of"
  the one we're testing. Example: if (h, r, t1) and (h, r, t2) are both
  in the KG, scoring t1 above t2 shouldn't penalise us. We filter out
  triples in `triple_set_all` (train+valid+test) before computing rank.
  This is standard practice; it gives a more honest evaluation.

Usage:
    python -m kgsage.cli.evaluate_encoder \\
        --dataset fb15k237 \\
        --ckpt experiments/kgsage/outputs/fb15k237_encoder.pt
"""
import argparse
import time

import torch

from kgsage.data.loaders import load_kg
from kgsage.data.datasets import resolve_dataset
from kgsage.encoder.models import KGSAGELinkPredictor


# Global default for datasets that don't specify their own expected_mrr.
DEFAULT_PASS_MRR = 0.30


def compute_filtered_mrr(model, kg, device, max_eval=None):
    """Compute filtered MRR + Hits@K on the test set.

    Returns:
        dict with `mrr`, `hits_at_1`, `hits_at_3`, `hits_at_10`, `n_test`.

    Method:
      For each test triple (h, r, t) we score it twice:
        - tail prediction:  score (h, r, *) for every *, rank t
        - head prediction:  score (*, r, t) for every *, rank h
      "Filtered" means we mask out other known triples before ranking
      so we don't penalise the model for ranking another true fact higher.
    """
    model.eval()
    edge_index = kg["edge_index"].to(device)
    edge_type = kg["edge_type"].to(device)
    test_triples = kg["triples_test"]
    n_ent = kg["n_ent"]
    triple_set_all = kg["triple_set_all"]

    # Pre-compute entity embeddings once for the whole evaluation.
    # The encoder reads the train graph; that's stable across all
    # test queries.
    with torch.no_grad():
        ent_emb_all = model.encoder(edge_index, edge_type)
        rel_emb_all = model.decoder.rel_emb.weight

    if max_eval is not None and max_eval < len(test_triples):
        # Useful for smoke-testing locally — restrict to a small subset.
        import random
        rng = random.Random(0)
        test_triples = rng.sample(test_triples, max_eval)

    n_test = len(test_triples)

    ranks_tail = []  # ranks when predicting the tail
    ranks_head = []  # ranks when predicting the head

    start_time = time.time()
    with torch.no_grad():
        for i, (h, r, t) in enumerate(test_triples):
            # ─── Tail prediction: score (h, r, *) for every * ────────
            h_emb = ent_emb_all[h]
            r_emb = rel_emb_all[r]
            tail_scores = (ent_emb_all * (h_emb * r_emb)).sum(dim=-1)
            # tail_scores has shape (n_ent,); element j is score of (h, r, j)

            true_score = tail_scores[t].item()
            # Mask filter: any other true (h, r, t') in the KG should not
            # count as "ahead of" t. We set their scores to -inf so they
            # never beat the true score.
            for t_prime in range(n_ent):
                if t_prime == t:
                    continue
                if (h, r, t_prime) in triple_set_all:
                    tail_scores[t_prime] = float("-inf")

            rank = 1 + (tail_scores > true_score).sum().item()
            ranks_tail.append(rank)

            # ─── Head prediction: score (*, r, t) for every * ────────
            t_emb = ent_emb_all[t]
            head_scores = (ent_emb_all * (r_emb * t_emb)).sum(dim=-1)

            true_score = head_scores[h].item()
            for h_prime in range(n_ent):
                if h_prime == h:
                    continue
                if (h_prime, r, t) in triple_set_all:
                    head_scores[h_prime] = float("-inf")

            rank = 1 + (head_scores > true_score).sum().item()
            ranks_head.append(rank)

            # Progress every 1000 triples.
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                eta = elapsed * (n_test - i - 1) / (i + 1)
                print(f"  evaluated {i + 1:>6}/{n_test:>6}   "
                      f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                      flush=True)

    all_ranks = ranks_tail + ranks_head

    mrr = sum(1.0 / r for r in all_ranks) / len(all_ranks)
    hits_at_1 = sum(1 for r in all_ranks if r <= 1) / len(all_ranks)
    hits_at_3 = sum(1 for r in all_ranks if r <= 3) / len(all_ranks)
    hits_at_10 = sum(1 for r in all_ranks if r <= 10) / len(all_ranks)

    return {
        "mrr": mrr,
        "hits_at_1": hits_at_1,
        "hits_at_3": hits_at_3,
        "hits_at_10": hits_at_10,
        "n_test": len(all_ranks) // 2,  # number of triples (each scored 2x)
    }


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
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_eval", type=int, default=None,
                    help="evaluate only N test triples (for quick local smoke test)")
    args = ap.parse_args()

    # Resolve dataset name → config dict.
    config = resolve_dataset(args.dataset)
    ckpt_path = args.ckpt or f"experiments/kgsage/outputs/{config['name']}_encoder.pt"

    if not Path(ckpt_path).exists():
        print(f"!! checkpoint not found: {ckpt_path}", file=sys.stderr)
        print(f"   run `python -m kgsage.cli.train_encoder --dataset {args.dataset}` first.",
              file=sys.stderr)
        return 1

    # Determine the pass threshold — use dataset's expected_mrr, fall back to global default.
    pass_mrr = config.get("expected_mrr")
    if pass_mrr is None:
        pass_mrr = DEFAULT_PASS_MRR
    warn_mrr = pass_mrr * 2 / 3  # warn at 66% of pass threshold

    print(f"Loading {config['name']} from {config['path']}/...", flush=True)
    kg = load_kg(config["path"])
    print(f"  {len(kg['triples_test']):,} test triples\n", flush=True)

    print(f"Loading checkpoint {ckpt_path}...", flush=True)
    device = torch.device(args.device)
    model, _, _ = KGSAGELinkPredictor.load_pretrained(ckpt_path, device=device)
    print(f"  loaded on {device}\n", flush=True)

    print("Computing filtered MRR... (this is slow — scoring every entity per query)", flush=True)
    results = compute_filtered_mrr(model, kg, device, max_eval=args.max_eval)
    print()

    # ─── Report ────────────────────────────────────────────────────────
    print("=" * 70)
    print("  KGSAGE Phase 1 Test 1.1 — Link Prediction MRR")
    print("=" * 70)
    print(f"  Dataset                 : {config['name']}")
    print(f"  Pass threshold (MRR)    : {pass_mrr:.2f}")
    print(f"  Test triples evaluated  : {results['n_test']:>6,}")
    print(f"  MRR (filtered)          : {results['mrr']:.4f}")
    print(f"  Hits@1                  : {results['hits_at_1']:.4f}")
    print(f"  Hits@3                  : {results['hits_at_3']:.4f}")
    print(f"  Hits@10                 : {results['hits_at_10']:.4f}")
    print()
    if results["mrr"] >= pass_mrr:
        verdict = "PASS"
        reason = (f"MRR {results['mrr']:.4f} ≥ {pass_mrr:.2f}. "
                  f"Encoder is well-trained. Proceed to Test 1.2.")
    elif results["mrr"] >= warn_mrr:
        verdict = "WARN"
        reason = (f"MRR {results['mrr']:.4f} is below the {pass_mrr:.2f} target "
                  f"but above the {warn_mrr:.2f} debug threshold. "
                  f"Check Test 1.2 result before deciding to retrain.")
    else:
        verdict = "FAIL"
        reason = (f"MRR {results['mrr']:.4f} < {warn_mrr:.2f}. "
                  f"Something is wrong with training. "
                  f"Check: data loader (edge_index shape, edge_type dtype), "
                  f"learning rate (try 1e-4 instead of 1e-3), "
                  f"number of epochs (try doubling), "
                  f"basis count (try a different value).")
    print("=" * 70)
    print(f"  DECISION: {verdict}")
    print("=" * 70)
    print(f"  {reason}")

    return 0 if verdict in ("PASS", "WARN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
