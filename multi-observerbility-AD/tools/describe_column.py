"""Tool 2: how one column is distributed, so the agent learns the scales.

DATA comes from data/california_housing.py, not from another tool. Python caches
modules, so every tool that imports it reads the ONE loaded frame.
"""

from data.california_housing import DATA


def describe_column(name):
    """Tool 2: how one column is distributed, so the agent learns the scales."""
    if name not in DATA.columns:
        return (f"ERROR: '{name}' is not a column. "
                "Call list_columns() to see the valid names.")
    v = DATA[name]
    return (f"{name}: median {v.median():.2f} | 1% {v.quantile(0.01):.2f} | "
            f"99% {v.quantile(0.99):.2f} | min {v.min():.2f} | max {v.max():.2f}")
