"""The gate that keeps an observer's norms blind.

The general rule, carried from the predecessor and inverted here: a tool is
blocked while it could contaminate a commitment not yet made. Norms must come
from world knowledge and persona alone -- so until an observer has declared
them, every tool that shows it DATA is refused. Once norms exist, the data
opens and the observer maps its (now fixed) norms onto the vocabulary.

Enforced in code, not asked for in a prompt: a prompt that says "decide
before looking" is a suggestion, and the ordering is the only thing that
makes the norms a viewpoint instead of a description of the dataset.
Returning a dict makes ADK skip the tool and hand the dict back as the
response, so a blocked call reads as an ordinary tool error the agent can
learn from. Returning None lets the call through.
"""
from tools._observers import state_key

#: every tool that reveals the dataset -- locked until norms exist
DATA_TOOL_NAMES = {"describe_dataset", "describe_relation", "explain_term", "inspect_triples"}


def keep_norms_blind(tool, args, tool_context):
    """before_tool_callback on each observer."""
    if tool.name not in DATA_TOOL_NAMES:
        return None                       # declaring and selecting police themselves

    norms_declared = tool_context.state.get(
        state_key("norms", tool_context.agent_name))
    if norms_declared:
        return None                       # phase 2: the data is open

    return {"result": (
        "ERROR: you have not formed your view yet. Your norms must come from "
        "what YOU know about the world and from who you are -- not from this "
        "dataset. Call declare_semantics first; the data opens afterwards.")}
