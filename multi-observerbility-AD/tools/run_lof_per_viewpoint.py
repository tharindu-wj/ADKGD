"""Tool 3: the statistical component -- and the ONLY code path that produces a score.

The two-plane rule (PROJECT_SPEC 3.4) lives or dies here: the LLM chooses what
to look at, this file decides how unusual each row is. No other module may
compute, threshold or rank an anomaly score (INV-3).

The dataset used to live in this file. It now lives in data/california_housing.py
so that describe_column can reach it without importing a tool. Nothing about
INV-3 moved with it: scoring still happens here and nowhere else.
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from data.active import DATA


def run_lof_per_viewpoint(columns: list[str], row_filter: dict | None = None) -> str:
    """Score every row with Local Outlier Factor over the chosen viewpoint.

    This is the statistical component: it decides how unusual each row is. The
    agent decides only WHAT to look at.

    Args:
        columns: column names to observe -- the viewpoint's columns. At least 2.
        row_filter: optional, narrows WHO each row is compared against. Shape:
            {"column": "Latitude", "min": 32.5, "max": 35.0}. Omit it to compare
            every row against the whole dataset.

    Returns a text summary: rows scored, the score spread, and the 5 most
    anomalous rows. Bad input comes back as an ERROR string explaining the fix.
    """
    # -- check the inputs, answering with readable errors ----------------------
    bad = [c for c in columns if c not in DATA.columns]
    if bad:
        return f"ERROR: unknown column(s) {bad}. Call list_columns() for valid names."
    if len(columns) < 2:
        return "ERROR: give at least 2 columns -- one column has no joint structure."

    # -- apply the row filter (this narrows the comparison population) ---------
    frame = DATA
    filter_note = f"no row filter -> all {len(DATA):,} rows"
    if row_filter is not None:
        col = row_filter["column"]
        if col not in DATA.columns:
            return f"ERROR: row filter column '{col}' does not exist."
        lo = row_filter.get("min", -np.inf)
        hi = row_filter.get("max", np.inf)
        frame = DATA[(DATA[col] >= lo) & (DATA[col] <= hi)]
        filter_note = f"{col} in [{lo}, {hi}] -> kept {len(frame):,} of {len(DATA):,} rows"
        if len(frame) < 100:
            return (f"ERROR: that filter keeps only {len(frame)} rows. In a tiny group "
                    "everything looks unusual. Widen the filter.")

    # -- score with LOF --------------------------------------------------------
    X = frame[list(columns)].to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # one line, but essential: without it the
    #                                           biggest-numbered column (Population,
    #                                           up to 35k) decides every distance.
    detector = LocalOutlierFactor(n_neighbors=20)
    detector.fit_predict(X)
    scores = -detector.negative_outlier_factor_  # flip: higher = more anomalous

    # -- summarise for the agent ----------------------------------------------
    top5 = np.argsort(scores)[::-1][:5]
    table = frame.iloc[top5].round(2).copy()
    table["lof_score"] = scores[top5].round(2)

    return (f"Scored {len(frame):,} rows on {list(columns)} ({filter_note}).\n"
            f"LOF scores: median {np.median(scores):.2f}, "
            f"99% {np.percentile(scores, 99):.2f}, max {scores.max():.2f} "
            f"(1.0 = as dense as neighbours, higher = more anomalous)\n"
            f"Top 5 most anomalous rows:\n{table.to_string()}")
