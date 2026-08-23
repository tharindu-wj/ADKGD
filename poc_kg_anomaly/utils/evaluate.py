"""Score a set of flagged triples against ground truth.

Only this module reads ground_truth.tsv. The views never see it, so a label
cannot leak into a score even by accident.
"""
import pandas as pd

# The anomaly classes, named for how they are built:
#   type_invalid -- wrong kind of entity in the slot  (belgium locatedin japan)
#   type_valid   -- right kind, wrong one             (belgium locatedin africa)
KINDS = ["type_invalid", "type_valid"]


def load_truth(gt_path):
    """The answer key: head, relation, tail, label (0/1), kind (real/type_invalid/type_valid)."""
    return pd.read_csv(gt_path, sep="\t", header=None,
                       names=["head", "relation", "tail", "label", "kind"])


def attach_truth(table, gt_path):
    """Join labels onto a finished view table. Call this AFTER scoring."""
    gt = load_truth(gt_path)
    out = table.merge(gt, on=["head", "relation", "tail"], how="left")

    # A duplicated (h,r,t) in the answer key makes this merge FAN OUT: the table
    # grows while the rank Series does not, and .isin() on index values then
    # re-pairs scores to the wrong triples. Measured with one duplicated row:
    # caught fell 105 -> 13 and precision 82.7% -> 10.2%, silently. 1_contaminate
    # only PRINTS its duplicate count, so this is the last line of defence.
    if len(out) != len(table):
        raise RuntimeError(
            f"ground truth merge changed the row count ({len(table)} -> {len(out)}). "
            f"{gt_path} has duplicate (head, relation, tail) rows.")

    if out["label"].isna().any():
        raise RuntimeError(f"{int(out['label'].isna().sum())} scored triples "
                           "have no ground-truth row -- the files are out of sync")
    return out


def evaluate(table, flagged):
    """flagged = a boolean Series over `table`. Returns a metrics dict.

    precision = of what we flagged, how much was actually an anomaly
    recall    = of the anomalies present, how many we found
    Per-kind recall is the interesting one: overall recall hides which KIND
    of anomaly is being missed.
    """
    n_flagged = int(flagged.sum())
    hits = table[flagged & (table.label == 1)]
    total_anom = int((table.label == 1).sum())

    m = {
        "flagged": n_flagged,
        "budget_pct": 100.0 * n_flagged / len(table),
        "caught": len(hits),
        "precision": len(hits) / n_flagged if n_flagged else 0.0,
        "recall": len(hits) / total_anom if total_anom else 0.0,
    }
    for kind in KINDS:
        present = int((table.kind == kind).sum())
        found = int((flagged & (table.kind == kind)).sum())
        m[f"recall_{kind}"] = found / present if present else float("nan")
        m[f"found_{kind}"] = found
        m[f"present_{kind}"] = present

    # What a detector flagging the same number of triples at random would get.
    m["chance_precision"] = total_anom / len(table)
    return m


def render(m, title):
    """One readable block per detector."""
    lines = [
        f"{title}",
        f"  flagged    {m['flagged']} triples ({m['budget_pct']:.1f}% of the graph)",
        f"  caught     {m['caught']} anomalies",
        f"  precision  {m['precision']:.1%}   (random would give {m['chance_precision']:.1%})",
        f"  recall     {m['recall']:.1%}",
    ]
    for kind in KINDS:
        lines.append(f"  recall {kind:<13} {m['recall_' + kind]:6.1%}  "
                     f"({m['found_' + kind]}/{m['present_' + kind]})")
    return "\n".join(lines)
