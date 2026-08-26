"""Tool: the second-opinion fetch -- the other observer's flags, blind.

THE DISAGREEMENT MECHANISM. Two observers reading different shortlists rarely
judge the same fact, so same-fact disagreement -- the architecture's whole
product -- would be left to luck. This serves each observer the TRIPLES its
counterpart flagged as anomalous, so every flag ends up with two independent
verdicts.

Blind on purpose: the candidates arrive with a neutral note and NO hint of
the other observer's verdict, reasoning, or even that another observer exists.
A second opinion anchored on the first is one opinion twice -- the same
isolation rule the whole tree is built on.
"""
import json

from loaders.context import get_context
from tools._observers import is_reviewer, other_agent, principal_of, state_key

#: how many second-opinion candidates an observer can be handed
REVIEW_CAP = 15


def review_candidates(tool_context=None) -> str:
    """Fetch additional candidates for your review. Judge every one.

    A further set of candidates has been surfaced for you. They come with no
    scores and no notes -- judge each one purely against YOUR norms, exactly
    as before, through submit_verdicts.

    Args: (none)
    """
    caller = tool_context.agent_name
    if not is_reviewer(caller):
        return ("ERROR: this phase is not yours. Reviewing happens after "
                "both observers have finished.")

    principal = principal_of(caller)
    counterpart = other_agent(principal)

    their_verdicts = json.loads(
        tool_context.state.get(state_key("verdicts", counterpart)) or "{}")
    flagged = [tuple(v["triple"]) for v in their_verdicts.values()
               if v["verdict"] == "anomaly"]
    if not flagged:
        return ("Nothing awaits your review -- you are done, stop here.")

    served_key = state_key("served", principal)
    served = json.loads(tool_context.state.get(served_key) or "{}")
    already = {tuple(entry["triple"]) for entry in served.values()}

    ctx = get_context()
    lines = ["Additional candidates for your review:"]
    added = 0
    for triple in flagged:
        if triple in already:
            continue                    # this observer has already judged it
        if added >= REVIEW_CAP:
            break
        review_id = f"r{added + 1}"
        served[review_id] = {"triple": list(triple),
                             "text": ctx.triple_text(triple),
                             "note": "additional candidate for your review",
                             "scanner": "second_opinion"}
        already.add(triple)
        added += 1
        lines.append(f"  {review_id}. {ctx.triple_text(triple)}")

    tool_context.state[served_key] = json.dumps(served)

    if not added:
        return ("Every additional candidate is one you have already judged "
                "-- you are done, stop here.")
    lines.append("\nJudge each with submit_verdicts, by your norms alone.")
    return "\n".join(lines)
