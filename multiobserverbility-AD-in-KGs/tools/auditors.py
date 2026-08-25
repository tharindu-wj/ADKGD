"""Who the auditors are, and where each one's artifacts live in state.

Three tools share these names (assign_perspective, declare_semantics,
select_scope), so they live once, here. An auditor's artifacts are keyed by
its number: sub_agent_1 owns persona_1, norms_1, scope_1.
"""

SUB_AGENT_NAMES = ("sub_agent_1", "sub_agent_2")


def state_key(kind, agent_name):
    """persona/norms/scope + the agent's number: state_key('norms',
    'sub_agent_1') -> 'norms_1'."""
    return f"{kind}_{agent_name.rsplit('_', 1)[-1]}"


def other_agent(agent_name):
    """The twin: sub_agent_1 <-> sub_agent_2."""
    return next(n for n in SUB_AGENT_NAMES if n != agent_name)


def essence(text):
    """Text reduced to letters and digits -- punctuation cannot hide sameness.

    Used by every guard that refuses two identical stances: a stray comma or
    full stop must not smuggle the same words past the comparison.
    """
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())
