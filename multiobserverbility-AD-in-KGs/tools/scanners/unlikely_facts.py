"""Scanner: facts the trained link predictor finds unlikely.

The one scanner that needs a model. scripts/2_train_scorer.py trains a KGE
model on the (contaminated) graph and scores EVERY triple once, offline, into
prepared/scores.npy -- so at run time this is an array lookup, no torch, no
40-second model load inside an agent's turn, and the scores are identical
across runs by construction.

A false fact tends to break the regularities the embedding learned, so a low
score is a falsehood LEAD -- serving "false in the world" norms. It is only a
lead: rare-but-true facts also score low, which is exactly why a judge reads
the shortlist. Note the model cannot see direction on symmetric relations --
that blindness is why one_way_links exists as a separate scanner.

The manifest check refuses a score file computed for a different graph: a
stale file would silently score triples that no longer exist.
"""
import hashlib
import json

import numpy as np

from loaders.active import DATASET

NAME = "unlikely_facts"


def find(scope_ids, ctx):
    """All in-scope triples, least plausible first, with their percentile."""
    if not DATASET.SCORES.exists():
        raise RuntimeError(
            "no score file. Run scripts/2_train_scorer.py first -- "
            "this scanner reads its scores precomputed.")

    manifest = json.loads(DATASET.SCORES_MANIFEST.read_text(encoding="utf-8"))
    kg_hash = hashlib.sha256(DATASET.KG.read_bytes()).hexdigest()
    if manifest["kg_sha256"] != kg_hash:
        raise RuntimeError(
            f"STALE SCORES: {DATASET.SCORES.name} was computed for a different "
            f"graph than {DATASET.KG.name}. Re-run scripts/2_train_scorer.py.")

    scores = np.load(DATASET.SCORES)
    if len(scores) != len(ctx.triples):
        raise RuntimeError("score file length does not match the graph.")

    # Percentile over the WHOLE graph -- the scope selects candidates, but
    # "bottom 0.4%" must mean the same thing whatever the scope is.
    order = np.argsort(scores, kind="stable")
    rank_of = np.empty(len(scores), dtype=int)
    rank_of[order] = np.arange(len(scores))

    candidates = []
    for position in order:
        triple = ctx.triples[position]
        if triple[1] not in scope_ids:
            continue
        percentile = 100.0 * rank_of[position] / len(scores)
        note = (f"plausibility score {scores[position]:.3f} "
                f"(bottom {max(percentile, 0.1):.1f}% of the graph)")
        candidates.append((triple, note))
    return candidates
