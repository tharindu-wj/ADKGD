"""Tool: state what this audit treats as NORMAL, before it may score anything.

THE PRIVATE SEMANTICS STORE. The agent declares what normal looks like under
its goal, what a violation of that looks like, and which relations it is
talking about. run_scorer refuses to answer until it has.

The ORDERING is the whole point. A frame written after seeing scores can be
reverse-engineered from whatever happened to score well -- the same failure
that keeps run_scorer away from the root agent: choosing the question from the
answer. Declared first, a frame is a commitment. Declared last, it is a
rationalisation, and worth nothing.

Facts are never gated, only scores: the profiler stays open before a frame
exists, and a knowledge base would too, because a frame is DERIVED from those.

ONE function rather than one per agent, so both viewpoints carry an identical
tool list. Which store it writes to comes from the name of whoever called it.
"""
import json

from loaders import graph
from loaders.active import DATASET


def store_key(agent_name: str) -> str:
    """viewpoint_a -> sem_a. One store per agent, private to that agent."""
    return "sem_" + agent_name.rsplit("_", 1)[-1]


def declare_semantics(normal: str, suspicious: str, relations: list[str],
                      tool_context) -> str:
    """Declare what this audit treats as normal. Required before scoring.

    Args:
        normal: one sentence -- what NORMAL looks like under your goal.
        suspicious: one sentence -- what a violation of that looks like.
        relations: which relations your goal is about. Names must be real.
    """
    if not normal or not normal.strip():
        return "ERROR: 'normal' is empty. Say what normal looks like."
    if not suspicious or not suspicious.strip():
        return "ERROR: 'suspicious' is empty. Say what a violation looks like."

    known = sorted({r for _, r, _ in graph.load_triples(DATASET.KG)})
    if not relations:
        return (f"ERROR: name at least one relation your goal is about. "
                f"This graph has: {', '.join(known)}.")

    # Anchored to the graph, not to whatever the agent felt like typing. This
    # is also what makes semantic consistency measurable afterwards.
    unknown = [r for r in relations if r not in known]
    if unknown:
        return (f"ERROR: no relation {', '.join(repr(r) for r in unknown)} in "
                f"this graph. It has: {', '.join(known)}.")

    tool_context.state[store_key(tool_context.agent_name)] = json.dumps({
        "normal": normal.strip(),
        "suspicious": suspicious.strip(),
        "relations": list(relations),
    })

    return (f"Recorded. This audit treats as normal: {normal.strip()}\n"
            f"  suspicious: {suspicious.strip()}\n"
            f"  in scope: {', '.join(relations)}\n\n"
            f"You may now run a scorer. Nothing about this frame is checked "
            f"against an answer key -- it is your commitment, not a verdict.")
