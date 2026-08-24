"""Score a set of flags against ground truth.

This is the ONLY module that reads ground_truth.tsv, and it is called only
after every score already exists.
"""
import pandas as pd

#: the anomaly classes, named for how 1_inject_anomalies.py builds them
KINDS = ["type_invalid", "type_valid"]


def attach_truth(table, truth_path):
    """Join labels onto a finished score table. Call AFTER scoring."""
    gt = pd.read_csv(truth_path, sep="\t", header=None,
                     names=["head", "relation", "tail", "label", "kind"])
    out = table.merge(gt, on=["head", "relation", "tail"], how="left")

    # A duplicated (h,r,t) in the answer key makes this merge FAN OUT: the
    # table grows while the score list does not, and rows silently re-pair.
    # Measured with one duplicated row: caught fell 105 -> 13, no error raised.
    if len(out) != len(table):
        raise RuntimeError(f"ground truth merge changed the row count "
                           f"({len(table)} -> {len(out)}): {truth_path} has "
                           "duplicate (head, relation, tail) rows.")
    if out["label"].isna().any():
        raise RuntimeError(f"{int(out['label'].isna().sum())} scored triples have "
                           "no ground-truth row -- the files are out of sync.")
    return out


def score(table, flagged):
    """precision, recall, and recall split by anomaly class."""
    n_flag = int(flagged.sum())
    hits = table[flagged & (table.label == 1)]
    total = int((table.label == 1).sum())

    m = {"flagged": n_flag,
         "pct": 100.0 * n_flag / len(table),
         "caught": len(hits),
         "precision": len(hits) / n_flag if n_flag else 0.0,
         "recall": len(hits) / total if total else 0.0,
         "chance": total / len(table)}
    for kind in KINDS:
        present = int((table.kind == kind).sum())
        found = int((flagged & (table.kind == kind)).sum())
        m[f"found_{kind}"] = found
        m[f"present_{kind}"] = present
        m[f"recall_{kind}"] = found / present if present else float("nan")
    return m


def render(m, title):
    lines = [title,
             f"  flagged    {m['flagged']} triples ({m['pct']:.1f}% of the graph)",
             f"  caught     {m['caught']} anomalies",
             f"  precision  {m['precision']:.1%}   (random would give {m['chance']:.1%})",
             f"  recall     {m['recall']:.1%}"]
    for kind in KINDS:
        lines.append(f"  recall {kind:<13} {m['recall_' + kind]:6.1%}  "
                     f"({m['found_' + kind]}/{m['present_' + kind]})")
    return "\n".join(lines)
