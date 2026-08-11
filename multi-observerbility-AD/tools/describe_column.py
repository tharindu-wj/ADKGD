"""Tool 2: how one column is distributed, so the agent learns the scales.

DATA comes from data/active.py (whichever dataset that currently points at),
not from another tool. Python caches modules, so every tool that imports it
reads the ONE loaded frame.
"""

from data.active import DATA


def describe_column(name: str) -> str:
    """Describe how one column is distributed, so the agent learns its scale.

    Args:
        name: the exact column name, as given by list_columns.

    Returns median, 1st and 99th percentiles, min and max. Unknown names come
    back as an ERROR string listing what to do instead.
    """
    if name not in DATA.columns:
        return (f"ERROR: '{name}' is not a column. "
                "Call list_columns() to see the valid names.")
    v = DATA[name]
    return (f"{name}: median {v.median():.2f} | 1% {v.quantile(0.01):.2f} | "
            f"99% {v.quantile(0.99):.2f} | min {v.min():.2f} | max {v.max():.2f}")
