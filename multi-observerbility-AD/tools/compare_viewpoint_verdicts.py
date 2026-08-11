"""Tool 4: compare what several FINISHED viewpoints each say about the same blocks.

WHERE THIS SITS IN THE WORKFLOW
-------------------------------
run_lof answers "what does THIS viewpoint find?" -- one viewpoint at a time.
This tool answers the next question: "what does EACH viewpoint say about the
same block?" It executes every viewpoint, aligns their results per block, and
returns the disagreement structure -- which blocks are flagged by all viewpoints,
which by several, which by exactly one. That table is the raw material for the
final explanation ("extreme as a location, ordinary as everything else").

It belongs AFTER derivation: the viewpoints passed in are final. It is not for
choosing between candidate column sets while still exploring -- that would let
per-block results steer derivation, which is the tuning loop the project avoids.

WHY CELLS ARE PERCENTILE RANKS, NOT YES/NO FLAGS
------------------------------------------------
Raw LOF values are not comparable across viewpoints with different columns, and
binary flags cannot support the sentence the explanation step needs ("99.9th
percentile here, 37th there"). Percentile ranks are both comparable and quotable.
The scoring itself is identical to run_lof (standardise -> LOF k=20) and MUST
stay identical -- k or preprocessing drifting between the two tools would make
their outputs quietly contradict each other.

The ~8 scoring lines are duplicated from run_lof ON PURPOSE: no tool imports
another tool (tools/registry.py). If you change the scoring recipe, change it in
BOTH files.

WHAT THE DISPLAY TRUNCATES (the one place this tool editorialises)
------------------------------------------------------------------
Each viewpoint flags its top 1% -- hundreds of blocks; the union would drown a
prompt. The TABLE therefore shows only each viewpoint's top-10 blocks (union of
those), while the agreement SUMMARY counts the full top-1% sets. So the rows are
"the strongest examples", the summary is the true totals. Both say so explicitly
in the output.
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from data.california_housing import DATA

#: Fraction of each viewpoint's scored population counted as "flagged".
FLAG_FRACTION = 0.01

#: Rows shown in the table: the union of each viewpoint's strongest TOP_SHOW blocks.
TOP_SHOW = 10

#: Same neighbourhood size as run_lof -- keep the two in sync.
NEIGHBOURS_K = 20

#: Guard rails. ONE viewpoint is allowed on purpose: with a single observer point
#: there is nothing to compare, but the per-block table is still the deliverable.
#: Requiring two used to make an agent INVENT a second observer point just to
#: satisfy this tool -- tooling corrupting the analysis, which must never happen.
MAX_VIEWPOINTS = 5
MIN_ROWS = 100


def compare_viewpoint_verdicts(viewpoints: list[dict]) -> str:
    """Execute several finished viewpoints and align their verdicts per block.

    Use this AFTER you have derived all your viewpoints, to see how they agree
    and disagree about individual blocks -- which blocks every viewpoint flags,
    and which are flagged by exactly one. The output is the evidence base for
    explaining findings; it is not for choosing columns while still exploring.

    Works with a single viewpoint too: you get its flagged blocks and their
    percentiles, with no comparison section. Never add a viewpoint you did not
    mean just to have something to compare.

    Args:
        viewpoints: 1 to 5 finished viewpoints, each a dict like
            {"observer": "geographic-isolation",
             "columns": ["Latitude", "Longitude"],
             "row_filter": null}
            row_filter, when present, has the same shape run_lof uses:
            {"column": "Latitude", "min": 32.5, "max": 35.0}.

    Returns a text report: one row per notable block with each viewpoint's
    percentile rank (higher = more anomalous, * = inside that viewpoint's top
    1%, -- = outside that viewpoint's row filter, so it has no verdict), plus
    agreement totals over the full flagged sets. Bad input comes back as an
    ERROR string explaining the fix.
    """
    # -- validate, answering with readable errors ------------------------------
    if not isinstance(viewpoints, list) or not viewpoints:
        return ("ERROR: give your finished viewpoints as a list, each a dict with "
                '"observer", "columns" and optional "row_filter". One viewpoint is '
                "fine -- you will get its flagged blocks with no comparison.")
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

        # Row filter: same semantics as run_lof. The mask records WHO this
        # viewpoint scores -- blocks outside it get no verdict, shown as "--".
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

        raw_name = str(vp.get("observer") or vp.get("name") or f"V{i + 1}")
        names.append(raw_name[:16])
        columns_per_vp.append(cols)
        masks.append(mask)

    # -- score every viewpoint (identical recipe to run_lof; see docstring) ----
    percentiles, flagged, strongest = [], [], []
    for cols, mask in zip(columns_per_vp, masks):
        frame = DATA.loc[mask]
        X = frame[list(cols)].to_numpy(dtype=float)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        detector = LocalOutlierFactor(n_neighbors=NEIGHBOURS_K)
        detector.fit_predict(X)
        scores = -detector.negative_outlier_factor_          # higher = more anomalous

        n = len(frame)
        ranks = scores.argsort().argsort()                   # 0 .. n-1, ascending
        pct = {block: ranks[j] / (n - 1) * 100 for j, block in enumerate(frame.index)}

        n_flag = max(1, int(round(FLAG_FRACTION * n)))
        order = np.argsort(scores)[::-1]
        flag_set = set(frame.index[order[:n_flag]].tolist())
        top_show = frame.index[order[:TOP_SHOW]].tolist()

        percentiles.append(pct)
        flagged.append(flag_set)
        strongest.append(top_show)

    # -- align: rows = union of each viewpoint's strongest blocks --------------
    rows = sorted(set().union(*strongest),
                  key=lambda b: (-sum(b in f for f in flagged),
                                 -max(p.get(b, 0.0) for p in percentiles)))

    header = f"{'block':>7}  " + "  ".join(f"{n:>16}" for n in names) + "   flagged_by"
    if len(viewpoints) == 1:
        preamble = (f"One viewpoint, so nothing to compare. It flags its top "
                    f"{FLAG_FRACTION:.0%}; the table shows its strongest "
                    f"{TOP_SHOW} blocks ({len(rows)} rows); the total below covers "
                    "the full flagged set.")
    else:
        preamble = (f"Compared {len(viewpoints)} viewpoints. Each flags its top "
                    f"{FLAG_FRACTION:.0%}; the table shows the union of each "
                    f"viewpoint's strongest {TOP_SHOW} blocks ({len(rows)} rows); "
                    "the totals below cover the full flagged sets.")
    lines = [
        preamble,
        "",
        "Cells: percentile rank (higher = more anomalous), * = flagged, "
        "-- = outside that viewpoint's row filter (no verdict).",
        "",
        header,
        "-" * len(header),
    ]
    for block in rows:
        cells = []
        for pct, flag_set in zip(percentiles, flagged):
            if block not in pct:
                cells.append(f"{'--':>16}")
            else:
                mark = "*" if block in flag_set else " "
                cells.append(f"{pct[block]:>15.1f}{mark}")
        by = [names[i] for i in range(len(names)) if block in flagged[i]]
        lines.append(f"{block:>7}  " + "  ".join(cells) + "   " + (", ".join(by) or "none"))

    # -- agreement totals over the FULL flagged sets ---------------------------
    union_flagged = set().union(*flagged)
    if len(viewpoints) == 1:
        lines += ["",
                  f"Only one viewpoint, so there is nothing to compare: it flags "
                  f"{len(union_flagged):,} blocks in total, of which the table shows "
                  f"its strongest {min(TOP_SHOW, len(rows))}."]
    else:
        counts = {block: sum(block in f for f in flagged) for block in union_flagged}
        lines += ["", f"Agreement across the full flagged sets ({len(union_flagged):,} blocks):"]
        for level in range(len(viewpoints), 0, -1):
            n_level = sum(1 for c in counts.values() if c == level)
            label = "all " + str(level) if level == len(viewpoints) else f"exactly {level}"
            lines.append(f"  flagged by {label} viewpoint(s): {n_level:,}")

    return "\n".join(lines)
