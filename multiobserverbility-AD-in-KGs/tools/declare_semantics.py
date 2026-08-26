"""Tool: an observer states its norms -- BEFORE it has seen any data.

This is phase 1 of the separation the whole design turns on. Like a person:
your sense of what a normal relationship looks like comes from who you are,
formed before you meet the community you will judge. The data may later teach
you the local vocabulary and where your values apply; it must not supply the
values.

So this tool validates NOTHING against the dataset -- deliberately. Norms are
written from world knowledge and persona alone, and the phase gate keeps
every data tool locked until they exist. Their blindness is what makes them
portable: the same norms could be carried to a different graph entirely.

Once declared, norms are immutable. A commitment that can be rewritten after
looking at the data is not a commitment.
"""
import json

from tools._observers import OBSERVER_NAMES, essence, other_agent, state_key


def declare_semantics(normal: str, anomalous: str, lets_pass: str,
                      tool_context) -> str:
    """State your norms: what you consider normal, anomalous, and tolerable.

    Write these from what you know about the world and from who you are --
    your persona. You have not seen the dataset and you do not need it for
    this: a judge's values exist before the case.

    Speak in world terms, never in terms of any
    dataset's fields. Each norm must hold no matter which particular entities
    turn out to be involved.

    Args:
        normal: what a normal, unremarkable fact of this domain looks like
            to you. One or two sentences.
        anomalous: what YOU flag -- including, if your persona demands it,
            things that may be factually true. One or two sentences.
        lets_pass: what you deliberately tolerate that another judge might
            flag. This is where your viewpoint shows -- one sentence.
    """
    agent = tool_context.agent_name
    if agent not in OBSERVER_NAMES:
        return f"ERROR: only an observer declares norms; '{agent}' is not one."

    for field_name, value in (("normal", normal), ("anomalous", anomalous),
                              ("lets_pass", lets_pass)):
        if not value or not value.strip():
            return (f"ERROR: '{field_name}' is empty. See this tool's "
                    f"description.")

    own_key = state_key("norms", agent)
    if tool_context.state.get(own_key):
        return ("ERROR: your norms are already declared. Norms are a "
                "commitment made before seeing data -- they cannot be "
                "rewritten after.")

    # Two observers reciting the same norms are one observer twice.
    other_norms_raw = tool_context.state.get(
        state_key("norms", other_agent(agent)))
    if other_norms_raw:
        other_norms = json.loads(other_norms_raw)
        mine = essence(normal) + essence(anomalous) + essence(lets_pass)
        theirs = (essence(other_norms["normal"])
                  + essence(other_norms["anomalous"])
                  + essence(other_norms["lets_pass"]))
        if mine == theirs:
            return ("ERROR: these norms are identical to the other "
                    "observer's. Your persona is different -- your norms "
                    "must be too.")

    tool_context.state[own_key] = json.dumps({
        "agent": agent,
        "normal": normal.strip(),
        "anomalous": anomalous.strip(),
        "lets_pass": lets_pass.strip(),
    })

    return ("Recorded. Your norms are now fixed.\n"
            f"  normal:    {normal.strip()}\n"
            f"  anomalous: {anomalous.strip()}\n"
            f"  lets pass: {lets_pass.strip()}\n\n"
            "The dataset is now open to you. Look at what it contains, then "
            "call select_scope with the relations your norms apply to.")
