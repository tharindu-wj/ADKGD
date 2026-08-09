"""Tool 2: how one column is distributed, so the agent learns the scales.

DATA is imported from run_lof rather than loaded again: Python caches modules,
so both tools read the one frame. (If a third consumer ever appears, that is the
moment to give the dataset its own module.)
"""

from tools.run_lof import DATA


def describe_column(name):
    """Tool 2: how one column is distributed, so the agent learns the scales."""
    if name not in DATA.columns:
        return (f"ERROR: '{name}' is not a column. "
                "Call list_columns() to see the valid names.")
    v = DATA[name]
    return (f"{name}: median {v.median():.2f} | 1% {v.quantile(0.01):.2f} | "
            f"99% {v.quantile(0.99):.2f} | min {v.min():.2f} | max {v.max():.2f}")
