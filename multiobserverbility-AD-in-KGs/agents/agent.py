"""What ADK loads: the audit tree.

    SequentialAgent "audit"            <- fixed order, never runtime-chosen
      |- Agent          "root"         <- personas from the card alone
      |- ParallelAgent  "auditors"     <- concurrent, isolated branches
           |- Agent  "sub_agent_1"     <- blind norms -> scope -> find -> judge
           |- Agent  "sub_agent_2"

ParallelAgent gives each auditor its own branch and filters sibling events,
so neither sees the other's calls. Session state is NOT branch-scoped -- the
tools key every artifact by the CALLER's name, and neither instruction names
the other's keys.
"""
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent

from agents.reviewers import reviewers
from agents.root_agent import root
from agents.sub_agents import sub_agents

auditors = ParallelAgent(
    name="auditors",
    description="Two auditors auditing the same graph from their own norms.",
    sub_agents=sub_agents,
)

#: the second-opinion phase runs only after BOTH audits are complete --
#: SequentialAgent is what guarantees that ordering
second_opinions = ParallelAgent(
    name="second_opinions",
    description="Each auditor judges the other's flags, blind.",
    sub_agents=reviewers,
)

#: the name ADK looks up in `agents.agent`
root_agent = SequentialAgent(
    name="audit",
    description=("Assign two personas from the card; each auditor declares "
                 "blind norms, maps them to a scope, judges what its chosen "
                 "assistants surface, then judges the other's flags blind."),
    sub_agents=[root, auditors, second_opinions],
)

__all__ = ["root_agent"]
