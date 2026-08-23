"""Turn scorer values into a set of flags. No labels here."""
import pandas as pd


def flag_worst(values, direction, budget):
    """Boolean mask over `values`: the `budget` fraction that look worst.

    direction -1 means a LOW value is anomalous, +1 means a HIGH one is.
    Declaring it rather than remembering it is the guard against the sign bug
    that silently turns a detector into its own opposite.

    Both detectors call this, so their flag sets are produced by identical
    code and cannot drift apart.
    """
    s = pd.Series(values)
    rank = s.rank(ascending=(direction < 0))     # rank 1 = most anomalous
    n = int(budget * len(s))
    return s.index.isin(rank.nsmallest(n).index), n
