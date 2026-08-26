"""NOT A TOOL -- the shared registry every tool leans on.

No agent can call this module and it performs no action, which is why its
name does not carry a verb: it answers "who are the observers, and where do
their artifacts live", nothing more. The underscore is the convention that
says so, the same way tools/scanners/ marks modules an agent cannot call
directly.

Who the observers are, and where each one's artifacts live in state.

Three tools share these names (assign_perspective, declare_semantics,
select_scope), so they live once, here. An observer's artifacts are keyed by
its number: observer_1 owns persona_1, norms_1, scope_1.
"""

OBSERVER_NAMES = ("observer_1", "observer_2")

#: In the second-opinion phase each observer returns under a reviewer name --
#: ADK needs unique agent names in one tree -- but it is the SAME observer:
#: same persona, same norms, and it writes into its principal's stores.
REVIEWER_SUFFIX = "_reviewer"


def principal_of(agent_name):
    """observer_1_reviewer -> observer_1; anyone else is themselves."""
    if agent_name and agent_name.endswith(REVIEWER_SUFFIX):
        return agent_name[:-len(REVIEWER_SUFFIX)]
    return agent_name


def is_reviewer(agent_name):
    return bool(agent_name) and agent_name.endswith(REVIEWER_SUFFIX)


def state_key(kind, agent_name):
    """persona/norms/scope + the agent's number: state_key('norms',
    'observer_1') -> 'norms_1'. Reviewers resolve to their principal."""
    return f"{kind}_{principal_of(agent_name).rsplit('_', 1)[-1]}"


def other_agent(agent_name):
    """The twin: observer_1 <-> observer_2."""
    return next(n for n in OBSERVER_NAMES if n != agent_name)


def essence(text):
    """Text reduced to letters and digits -- punctuation cannot hide sameness.

    Used by every guard that refuses two identical stances: a stray comma or
    full stop must not smuggle the same words past the comparison.
    """
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())
