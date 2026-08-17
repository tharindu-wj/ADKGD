"""Tool 4: which entities did MORE THAN ONE finished viewpoint flag?

run_lof_per_viewpoint answers "what does THIS viewpoint find?", one viewpoint at
a time. This tool runs them all, flags each viewpoint's top FLAG_FRACTION, and
reports the OVERLAP -- the entities more than one viewpoint flagged. Every
viewpoint's percentile is printed for those, so an entity two viewpoints both
rate at the 99th reads differently from one that scraped in at the 90th.

It belongs AFTER derivation: the viewpoints passed in are final. Using it to
choose between candidate column sets would let per-entity results steer
derivation, which is the tuning loop the project avoids.

The ~8 scoring lines are copied from run_lof_per_viewpoint ON PURPOSE -- no tool
imports another tool (tools/registry.py). Change the recipe in one, change it
in both, or the two tools will quietly disagree about the same viewpoint.
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from data.active import DATA

#: Fraction of each viewpoint's scored population counted as "flagged".
#: Measured trade-off on california_housing (entities flagged by 2+ of three
#: baseline viewpoints):
#:      top 1%  ->  25 entities vs 5.8 expected by chance  (4.3x)
#:      top 10% -> 764 entities vs 619 expected by chance  (1.2x)
#: A wider cut finds more overlap, but most of the extra is coincidence.
FLAG_FRACTION = 0.10

#: Maximum rows in the table. The overlap is usually far bigger.
TOP_SHOW = 10

#: Same neighbourhood size as run_lof_per_viewpoint -- keep the two in sync.
NEIGHBOURS_K = 20

#: ONE viewpoint is allowed on purpose. Requiring two used to make an agent
#: INVENT a second observer point just to satisfy this tool.
MAX_VIEWPOINTS = 5
MIN_ROWS = 100


def _filter_signature(row_filter):
    """A row filter reduced to something two filters can be compared by.

    None and {} both mean "every row", so they must compare equal -- otherwise
    two identical viewpoints would look different and the duplicate warning
    below would never fire.
    """
    if not row_filter:
        return None
    return (row_filter.get("column"),
            row_filter.get("min", -np.inf),
            row_filter.get("max", np.inf))


def compare_viewpoints(viewpoints: list[dict]) -> str:
    """Run finished viewpoints and report the entities more than one of them flagged.

    Use this AFTER every viewpoint is final. Each viewpoint flags its top 10%;
    the entities appearing in two or more of those sets are the result.

    One viewpoint is fine -- there is no overlap to find, so you get its
    strongest entities instead. Never add a viewpoint you did not mean just to
    have something to compare.

    Args:
        viewpoints: 1 to 5 finished viewpoints, each a dict like
            {"observer": "<short name for the observer point>",
             "columns": ["<col>", "<col>"],
             "row_filter": null}
            row_filter, when present, has the same shape run_lof_per_viewpoint
            uses: {"column": "<col>", "min": <num>, "max": <num>}.

    Returns a text report: the overlapping entities with every viewpoint's
    percentile rank (higher = more anomalous, * = in that viewpoint's flagged
    top 10%, -- = outside its row filter so it has no verdict), then counts by
    how many viewpoints agreed and what chance alone would have produced. Bad
    input comes back as an ERROR string explaining the fix.
    """
    # -- validate, answering with readable errors ------------------------------
    if not isinstance(viewpoints, list) or not viewpoints:
        return ("ERROR: give your finished viewpoints as a list, each a dict with "
                '"observer", "columns" and optional "row_filter". One viewpoint is '
                "fine -- you will get its flagged entities with no comparison.")
    if len(viewpoints) > MAX_VIEWPOINTS:
        return f"ERROR: compare at most {MAX_VIEWPOINTS} viewpoints; {len(viewpoints)} given."

    names, columns_per_vp, masks = [], [], []
    for i, vp in enumerate(viewpoints):
        if not isinstance(vp, dict):
            return f"ERROR: viewpoint {i + 1} is not a dict."
        cols = vp.get("columns")
        if not isinstance(cols, list) or len(cols) < 2:
            return (f"ERROR: viewpoint {i + 1} needs a 'columns' list of at least 2 "
                    "column names.")
        unknown = [c for c in cols if c not in DATA.columns]
        if unknown:
            return (f"ERROR: viewpoint {i + 1} names unknown column(s) {unknown}. "
                    "Call list_columns for the valid names.")

        # Row filter: same semantics as run_lof_per_viewpoint. The mask records
        # WHO this viewpoint scores -- entities outside it get no verdict ("--").
        mask = np.ones(len(DATA), dtype=bool)
        rf = vp.get("row_filter")
        if rf:
            col = rf.get("column")
            if col not in DATA.columns:
                return f"ERROR: viewpoint {i + 1} row filter column '{col}' does not exist."
            values = DATA[col].to_numpy()
            mask &= (values >= rf.get("min", -np.inf)) & (values <= rf.get("max", np.inf))
            if mask.sum() < MIN_ROWS:
                return (f"ERROR: viewpoint {i + 1}'s row filter keeps only "
                        f"{int(mask.sum())} rows. Widen it.")

        names.append(str(vp.get("observer") or vp.get("name") or f"V{i + 1}")[:16])
        columns_per_vp.append(cols)
        masks.append(mask)

    # -- are any two of these the SAME viewpoint? ------------------------------
    # Two observers can independently land on identical columns and filter. Their
    # overlap is then 100% by arithmetic, not by agreement, and a reader shown
    # only the table would take it for overwhelming corroboration. Say so.
    # Column ORDER does not matter to LOF, so compare as a sorted set.
    signatures = [(tuple(sorted(cols)), _filter_signature(vp.get("row_filter")))
                  for cols, vp in zip(columns_per_vp, viewpoints)]
    duplicates = [(names[i], names[j])
                  for i in range(len(signatures))
                  for j in range(i + 1, len(signatures))
                  if signatures[i] == signatures[j]]

    # -- score each viewpoint alone (same recipe as the other tool; see docstring)
    percentiles, flagged = [], []
    for cols, mask in zip(columns_per_vp, masks):
        frame = DATA.loc[mask]
        X = frame[list(cols)].to_numpy(dtype=float)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        detector = LocalOutlierFactor(n_neighbors=NEIGHBOURS_K)
        detector.fit_predict(X)
        scores = -detector.negative_outlier_factor_          # higher = more anomalous

        n = len(frame)
        ranks = scores.argsort().argsort()                   # 0 .. n-1, ascending
        percentiles.append({b: ranks[j] / (n - 1) * 100 for j, b in enumerate(frame.index)})
        n_flag = max(1, int(round(FLAG_FRACTION * n)))
        flagged.append(set(frame.index[np.argsort(scores)[::-1][:n_flag]].tolist()))

    # -- the answer: entities flagged by more than one viewpoint ---------------
    k = len(viewpoints)
    counts = {b: sum(b in f for f in flagged) for b in set().union(*flagged)}

    # Rank by the WEAKEST verdict any scoring viewpoint gave, so entities every
    # viewpoint rates highly beat ones that scraped into a flagged set by a hair.
    def weakest(b):
        return min((p[b] for p in percentiles if b in p), default=0.0)

    need = 2 if k > 1 else 1                # with one viewpoint there is no overlap
    shared = [b for b, c in counts.items() if c >= need]
    rows = sorted(shared, key=lambda b: (-counts[b], -weakest(b)))[:TOP_SHOW]

    # -- report ----------------------------------------------------------------
    if k == 1:
        headline = (f"One viewpoint, so there is no overlap to find. Its top "
                    f"{FLAG_FRACTION:.0%} is {len(shared):,} entities; the strongest "
                    f"{len(rows)} are below.")
    elif shared:
        headline = (f"Compared {k} viewpoints, each flagging its top {FLAG_FRACTION:.0%}. "
                    f"{len(shared):,} entities were flagged by MORE THAN ONE viewpoint -- "
                    f"strongest agreement first, {len(rows)} shown.")
    else:
        headline = (f"Compared {k} viewpoints, each flagging its top {FLAG_FRACTION:.0%}. "
                    f"NO entity was flagged by more than one: every anomaly here belongs "
                    f"to a single perspective.")

    header = f"{'entity':>7}  " + "  ".join(f"{n:>16}" for n in names) + "   flagged_by"
    lines = [headline]

    # Before anything else, if two viewpoints are the same one twice.
    for a, b in duplicates:
        lines += ["",
                  f"WARNING: {a} and {b} are the SAME viewpoint -- identical "
                  f"columns and row filter. Their agreement is arithmetic, not "
                  f"evidence: any entity one flags, the other flags too. Report "
                  f"that they converged on one viewpoint; do not report their "
                  f"overlap as corroboration."]

    lines += ["",
              "Cells: percentile rank (higher = more anomalous), * = flagged, "
              "-- = outside that viewpoint's row filter (no verdict).",
              "", header, "-" * len(header)]

    for b in rows:
        cells = [f"{'--':>16}" if b not in p
                 else f"{p[b]:>15.1f}" + ("*" if b in f else " ")
                 for p, f in zip(percentiles, flagged)]
        by = [names[i] for i in range(k) if b in flagged[i]]
        lines.append(f"{b:>7}  " + "  ".join(cells) + "   " + (", ".join(by) or "none"))

    # -- counts over the FULL flagged sets, not the rows shown -----------------
    # With one viewpoint the headline already gave the only number there is.
    if k > 1:
        lines += ["", f"Across the full flagged sets ({len(counts):,} entities):"]
        for level in range(k, 0, -1):
            n_level = sum(1 for c in counts.values() if c == level)
            label = f"all {level}" if level == k else f"exactly {level}"
            lines.append(f"  flagged by {label} viewpoint(s): {n_level:,}")
        # Without this, real agreement is indistinguishable from the artefact of
        # flagging 10% of the data k separate times.
        n_scored = min(len(p) for p in percentiles)
        expected = k * (k - 1) / 2 * FLAG_FRACTION ** 2 * n_scored
        lines.append(f"  -> {len(shared):,} flagged by more than one, against roughly "
                     f"{expected:,.0f} expected if they agreed only by chance.")

    return "\n".join(lines)
