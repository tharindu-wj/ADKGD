"""Tool 1: what data exists.

Every tool in this folder is a plain function that returns a STRING, because
that is what an LLM actually receives: text in, text out. Errors are returned
as text too, so the agent can read the error and correct itself rather than
crashing the loop.

One tool per file, registered in tools/registry.py. Both orchestrators (the
hand-written loop and the LangChain one) import the same functions from here,
so a tool is written once and behaves identically under either.

This tool is now a pure formatter: the column vocabulary it prints describes the
DATASET, so it lives in data/california_housing.py beside the frame it describes.
"""

from data.california_housing import COLUMN_MEANINGS


def list_columns():
    """Tool 1: what data exists. Usually the agent's first call."""
    lines = ["Available columns:"]
    for name, meaning in COLUMN_MEANINGS.items():
        lines.append(f"  {name}: {meaning}")
    return "\n".join(lines)
