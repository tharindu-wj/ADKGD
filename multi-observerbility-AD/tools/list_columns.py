"""Tool 1: what data exists.

Every tool in this folder is a plain function that returns a STRING, because
that is what an LLM actually receives: text in, text out. Errors are returned
as text too, so the agent can read the error and correct itself rather than
crashing the loop.

One tool per file, registered in tools/registry.py. Both orchestrators (the
hand-written loop and the LangChain one) import the same functions from here,
so a tool is written once and behaves identically under either.
"""

#: What each column means, in the agent's language. This is the ONLY place the
#: column vocabulary is written down, and it lives here because list_columns is
#: its only consumer -- if it ever gains a second one, give it its own module.
COLUMN_MEANINGS = {
    "MedInc": "median household income, in tens of thousands of dollars",
    "HouseAge": "median age of the houses, in years",
    "AveRooms": "average rooms per household (rooms / households)",
    "AveBedrms": "average bedrooms per household (bedrooms / households)",
    "Population": "people living in the block group",
    "AveOccup": "average people per household (population / households)",
    "Latitude": "degrees north; higher = further north",
    "Longitude": "degrees east; more negative = further west",
}


def list_columns():
    """Tool 1: what data exists. Usually the agent's first call."""
    lines = ["Available columns:"]
    for name, meaning in COLUMN_MEANINGS.items():
        lines.append(f"  {name}: {meaning}")
    return "\n".join(lines)
