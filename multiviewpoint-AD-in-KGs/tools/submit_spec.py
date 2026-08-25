"""Tool: hand in the finished viewpoint. Which scorer, at what budget, and why.

WHY THIS IS A TOOL AND NOT THE AGENT'S CLOSING MESSAGE.

It used to be the closing message: the agent ended its turn with JSON and a
callback scraped the JSON back out. Measured across nine recorded runs, that is
18 agent turns:

    the frame, written by a TOOL CALL (declare_semantics)   18 / 18
    the spec,  read from the FINAL MESSAGE                  10 / 18

Eight times the agent called its tools, ran a scorer, and then simply stopped
talking. There was no closing message to read, so the decision was lost and the
run recorded nothing to evaluate. Nothing retries.

Calling a tool is something a model does in order to GET something back.
Writing a closing summary is optional politeness, and gets skipped. So anything
the run actually needs is collected the first way.

It also buys validation. A misspelled scorer or an out-of-range budget comes
straight back as an error the agent can fix while it is still running, instead
of sailing through and failing later in a script with no agent left alive to
correct it -- the same reason declare_semantics refuses a relation that is not
in the graph.
"""
import json

from tools.run_scorer import SCORERS

MAX_BUDGET = 0.5


def spec_key(agent_name: str) -> str:
    """viewpoint_a -> spec_a. Mirrors store_key in declare_semantics."""
    return "spec_" + agent_name.rsplit("_", 1)[-1]


def submit_spec(scorer: str, budget: float, why: str, summary: str,
                tool_context) -> str:
    """Hand in your finished viewpoint. Call this last; it ends your work.

    You are not finished until you have called this. A decision you only
    describe is a decision the audit never receives.

    Args:
        scorer: which scorer your viewpoint uses. One of the names run_scorer
            accepts.
        budget: the fraction of the graph to flag, above 0 and at most 0.5.
            This is a review cost -- every flagged triple is one a person would
            have to check, so a wider budget is not a free win.
        why: one or two sentences, in terms of your goal, saying why this
            scorer serves it. Not why the scorer is good in general.
        summary: what the scorer actually told you -- how many triples it
            flagged and the worst one it found.
    """
    if scorer not in SCORERS:
        return (f"ERROR: no scorer '{scorer}'. "
                f"Available: {', '.join(sorted(SCORERS))}.")
    try:
        budget = float(budget)
    except (TypeError, ValueError):
        return f"ERROR: budget must be a number, got {budget!r}."
    if not 0 < budget <= MAX_BUDGET:
        return (f"ERROR: budget must be above 0 and at most {MAX_BUDGET}, "
                f"got {budget}.")
    if not why or not why.strip():
        return "ERROR: 'why' is empty. Say why this scorer serves your goal."

    tool_context.state[spec_key(tool_context.agent_name)] = json.dumps({
        "scorer": scorer,
        "budget": budget,
        "why": why.strip(),
        "summary": (summary or "").strip(),
    })
    return (f"Submitted: {scorer} at {budget:.0%} of the graph. "
            f"Your viewpoint is recorded. You are done -- stop here.")
