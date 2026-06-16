"""Classify each relation as 1-1, 1-N, N-1, or N-N.

Follows Bordes et al. (2013) convention with threshold = 1.5:

  For each relation r, compute:
    avg_tails_per_head(r) = average over heads h of |{t: (h,r,t)}|
    avg_heads_per_tail(r) = average over tails t of |{h: (h,r,t)}|

  head_side = "1" if avg_heads_per_tail(r) < threshold else "N"
  tail_side = "1" if avg_tails_per_head(r) < threshold else "N"
  cardinality(r) = head_side + "-" + tail_side

Examples (intuition check):
  /people/person/nationality
    many citizens point to one country -> avg_heads_per_tail = high -> "N"
    each person has ~1 nationality      -> avg_tails_per_head = ~1  -> "1"
    -> classified as "N-1"

  /location/country/capital
    one capital per country            -> avg_tails_per_head ~ 1    -> "1"
    one country per capital            -> avg_heads_per_tail ~ 1    -> "1"
    -> classified as "1-1"

These classifications inform Phase 2's cardinality-weighted sampling
(following CGSP Eqs. 5-6, Tong et al. 2026): tail corruption of 1-N
relations and head corruption of N-1 relations get DOWN-WEIGHTED to
avoid generating false negatives (triples that ARE true but happen to
be absent from training).
"""
from collections import defaultdict


# Bordes 2013 convention. Don't change without justifying in the thesis.
DEFAULT_THRESHOLD = 1.5


def classify_cardinality(triples_path, relation_to_id, threshold=DEFAULT_THRESHOLD):
    """Classify every relation in the vocabulary.

    Args:
      triples_path:    path to train.txt (TSV format).
      relation_to_id:  vocabulary built by concept_pools.build_vocab().
      threshold:       Bordes convention; 1.5 is standard.

    Returns:
      cardinality:       {r_id: "1-1" | "1-N" | "N-1" | "N-N"}
      cardinality_stats: {r_id: {"avg_tails_per_head": float,
                                  "avg_heads_per_tail": float}}
    """
    # Group by relation, then by head (for tails-per-head) and tail (for heads-per-tail).
    tails_per_head = defaultdict(lambda: defaultdict(set))   # r_id -> h -> {tails}
    heads_per_tail = defaultdict(lambda: defaultdict(set))   # r_id -> t -> {heads}

    with open(triples_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            if r not in relation_to_id:
                continue
            r_id = relation_to_id[r]
            tails_per_head[r_id][h].add(t)
            heads_per_tail[r_id][t].add(h)

    # Compute averages and classify.
    cardinality = {}
    stats = {}
    for r_id in range(len(relation_to_id)):
        tph_values = [len(s) for s in tails_per_head[r_id].values()]
        hpt_values = [len(s) for s in heads_per_tail[r_id].values()]
        avg_tph = sum(tph_values) / len(tph_values) if tph_values else 0.0
        avg_hpt = sum(hpt_values) / len(hpt_values) if hpt_values else 0.0
        head_side = "1" if avg_hpt < threshold else "N"
        tail_side = "1" if avg_tph < threshold else "N"
        cardinality[r_id] = f"{head_side}-{tail_side}"
        stats[r_id] = {
            "avg_tails_per_head": avg_tph,
            "avg_heads_per_tail": avg_hpt,
        }
    return cardinality, stats
