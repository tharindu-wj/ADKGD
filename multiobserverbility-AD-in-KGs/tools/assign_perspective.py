"""Tool: the root's only act -- give one observer its persona.

A persona is a BACKGROUND, not a set of norms: the kind of judge this observer
is, the stance it brings to anything it looks at. The observer will write its
own norms from this persona later, before it has seen any data -- so the
persona must not contain dataset specifics, and the root that writes it is
given no way to learn any (it has no other tools).

The two personas MUST genuinely differ. Two observers with the same background
converge on the same norms -- measured in the predecessor project, 44 out of
44 self-written frames converged -- and two identical viewpoints are one
viewpoint twice. The requirement is enforced here, in code, because a prompt
asking for it is only a suggestion.
"""
import json

from tools._observers import OBSERVER_NAMES, essence, other_agent, state_key


def assign_perspective(agent: str, persona: str, tool_context) -> str:
    """Give one observer its persona. Call once per observer.

    A persona says what KIND of judge this observer is -- the values and
    stance it brings, in two or three sentences. It must work for ANY
    dataset in this domain: describe a way of judging, never particular
    data, attributes, or statistics.

    The two observers' personas must genuinely differ -- that difference is
    the whole reason there are two of them. You are not finished until both
    observers have one.

    Args:
        agent: who this is for -- "observer_1" or "observer_2".
        persona: the stance, e.g. how a strict formalist and a descriptive
            empiricist would differ about the same facts. Two or three
            sentences, addressed to the observer as "you".
    """
    if agent not in OBSERVER_NAMES:
        return (f"ERROR: no observer named '{agent}'. "
                f"There are two: {', '.join(OBSERVER_NAMES)}.")

    if not persona or not persona.strip():
        return ("ERROR: persona is empty. Two or three sentences: what kind "
                "of judge is this observer?")

    own_key = state_key("persona", agent)
    if tool_context.state.get(own_key):
        return (f"ERROR: {agent} already has its persona. Personas are a "
                f"commitment -- they cannot be rewritten.")

    # The one rule that makes this two viewpoints instead of one twice.
    other_persona_raw = tool_context.state.get(
        state_key("persona", other_agent(agent)))
    if other_persona_raw:
        other_persona = json.loads(other_persona_raw)["persona"]
        if essence(persona) == essence(other_persona):
            return (f"ERROR: this persona is identical to "
                    f"{other_agent(agent)}'s. Give this observer a genuinely "
                    f"different way of judging.")

    tool_context.state[own_key] = json.dumps({
        "agent": agent,
        "persona": persona.strip(),
    })

    placed = [name for name in OBSERVER_NAMES
              if tool_context.state.get(state_key("persona", name))]
    if len(placed) == len(OBSERVER_NAMES):
        closing = "Both personas are placed -- you are done, stop here."
    else:
        closing = f"{other_agent(agent)} still needs its persona."

    return f"Recorded for {agent}:\n  {persona.strip()}\n\n{closing}"
