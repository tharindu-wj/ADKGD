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


def declare_semantics(relations: list[str], entities: str, normal: str,
                      suspicious: str, impossible: str, tool_context) -> str:
    """Declare what this audit treats as normal. Required before scoring.

    Say what the things ARE, not merely that they are correct. "Normal is when
    the fact is right" restates the question -- it commits you to nothing, and
    tells whoever reads this run nothing either.

    bad   normal:     a place is contained in a valid container, in accordance
                      with the correct containment hierarchy
                      (this only says "normal is when it is correct")

    good  entities:   places of different sizes -- countries, the regions that
                      group them, and the continents that group those
          normal:     a place sits inside a larger place that really surrounds it
          impossible: containment only ever goes upward in size, so a continent
                      inside a country cannot be true whatever the names are

    Args:
        relations: which relations your goal is about. Names must be real.
        entities: what KINDS of thing sit on each side of those relations.
        normal: what a sound triple looks like under your goal.
        suspicious: what a broken one looks like.
        impossible: what could NEVER be true, whichever particular places are
            named. State the rule, not an instance of it.
    """
    for name, value in (("entities", entities), ("normal", normal),
                        ("suspicious", suspicious), ("impossible", impossible)):
        if not value or not value.strip():
            return f"ERROR: '{name}' is empty. See this tool's description."

    # The cheapest tautology guard there is. An agent that answers the last two
    # questions with the same sentence has restated rather than reasoned.
    if suspicious.strip().lower() == impossible.strip().lower():
        return ("ERROR: 'impossible' repeats 'suspicious'. 'suspicious' is what "
                "a wrong triple looks like; 'impossible' is the rule that makes "
                "it wrong -- true of every triple, whatever names appear in it.")

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
        "relations": list(relations),
        "entities": entities.strip(),
        "normal": normal.strip(),
        "suspicious": suspicious.strip(),
        "impossible": impossible.strip(),
    })

    return (f"Recorded.\n"
            f"  in scope:   {', '.join(relations)}\n"
            f"  entities:   {entities.strip()}\n"
            f"  normal:     {normal.strip()}\n"
            f"  suspicious: {suspicious.strip()}\n"
            f"  impossible: {impossible.strip()}\n\n"
            f"You may now run a scorer. Nothing about this frame is checked "
            f"against an answer key -- it is your commitment, not a verdict.")
