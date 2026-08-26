"""Tool: an observer maps its norms onto the dataset's actual vocabulary.

Phase 2 of the separation. The norms were written blind; now the observer has
looked at what this graph actually records, and says WHERE its norms apply --
which relations are the ones its values are about. The values did not come
from the data; the data only tells the judge where its values are relevant.
"""
import json

from loaders.context import get_context
from tools._observers import OBSERVER_NAMES, state_key


def select_scope(relations: list[str], why: str, tool_context) -> str:
    """Choose the relations your norms apply to. Call once, after looking.

    Your norms are fixed; this maps them onto this particular dataset's
    vocabulary. Pick every relation your norms have something to say about,
    and no relation they are silent on. The other observer may pick the same
    ones -- shared scope with different norms is the most valuable overlap
    there is.

    Args:
        relations: relation names, exactly as describe_dataset lists them.
        why: one or two sentences connecting your norms to this choice --
            which norm makes each of these relations yours.
    """
    agent = tool_context.agent_name
    if agent not in OBSERVER_NAMES:
        return f"ERROR: only an observer selects a scope; '{agent}' is not one."

    if not tool_context.state.get(state_key("norms", agent)):
        return ("ERROR: declare your norms first. A scope is where your "
                "norms apply -- without norms there is nothing to place.")

    if tool_context.state.get(state_key("scope", agent)):
        return ("ERROR: your scope is already selected. Like your norms, "
                "it is a commitment.")

    if not relations:
        return "ERROR: name at least one relation your norms apply to."
    if not why or not why.strip():
        return ("ERROR: 'why' is empty. Connect your norms to this choice "
                "in a sentence or two.")

    ctx = get_context()
    resolved = []
    for name in relations:
        relation_id = ctx.find_relation(name)
        if relation_id is None:
            known = ", ".join(ctx.all_relation_labels())
            return (f"ERROR: no relation '{name}'. "
                    f"Known relations: {known}.")
        entry = {"id": relation_id, "label": ctx.relation_label(relation_id)}
        if entry not in resolved:
            resolved.append(entry)

    tool_context.state[state_key("scope", agent)] = json.dumps({
        "agent": agent,
        "scope": resolved,
        "why": why.strip(),
    })

    labels = ", ".join(entry["label"] for entry in resolved)
    return (f"Recorded. Your scope: {labels}.\n"
            f"Your observability point is complete -- persona, norms, scope.\n\n"
            f"Now find and judge: call find_suspects with an assistant that "
            f"matches your norms, and give a verdict on every candidate it "
            f"serves you. You are not done until every served candidate is "
            f"judged through submit_verdicts.")
