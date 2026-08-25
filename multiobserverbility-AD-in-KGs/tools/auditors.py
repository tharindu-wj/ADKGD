"""Who the auditors are, and where each one's artifacts live in state.

Three tools share these names (assign_perspective, declare_semantics,
select_scope), so they live once, here. An auditor's artifacts are keyed by
its number: sub_agent_1 owns persona_1, norms_1, scope_1.
"""

SUB_AGENT_NAMES = ("sub_agent_1", "sub_agent_2")

#: In the second-opinion phase each auditor returns under a reviewer name --
#: ADK needs unique agent names in one tree -- but it is the SAME auditor:
#: same persona, same norms, and it writes into its principal's stores.
REVIEWER_SUFFIX = "_reviewer"


def principal_of(agent_name):
    """sub_agent_1_reviewer -> sub_agent_1; anyone else is themselves."""
    if agent_name and agent_name.endswith(REVIEWER_SUFFIX):
        return agent_name[:-len(REVIEWER_SUFFIX)]
    return agent_name


def is_reviewer(agent_name):
    return bool(agent_name) and agent_name.endswith(REVIEWER_SUFFIX)


def state_key(kind, agent_name):
    """persona/norms/scope + the agent's number: state_key('norms',
    'sub_agent_1') -> 'norms_1'. Reviewers resolve to their principal."""
    return f"{kind}_{principal_of(agent_name).rsplit('_', 1)[-1]}"


def other_agent(agent_name):
    """The twin: sub_agent_1 <-> sub_agent_2."""
    return next(n for n in SUB_AGENT_NAMES if n != agent_name)


def essence(text):
    """Text reduced to letters and digits -- punctuation cannot hide sameness.

    Used by every guard that refuses two identical stances: a stray comma or
    full stop must not smuggle the same words past the comparison.
    """
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())
