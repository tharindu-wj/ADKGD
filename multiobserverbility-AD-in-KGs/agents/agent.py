"""What ADK loads: the setup tree.

    SequentialAgent "setup"            <- fixed order, never runtime-chosen
      |- Agent          "root"         <- personas from the card alone
      |- ParallelAgent  "auditors"     <- concurrent, isolated branches
           |- Agent  "sub_agent_1"     <- blind norms, then data, then scope
           |- Agent  "sub_agent_2"

ParallelAgent gives each auditor its own branch and filters sibling events,
so neither sees the other's calls. Session state is NOT branch-scoped -- the
tools key every artifact by the CALLER's name, and neither instruction names
the other's keys.
"""
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent

from agents.root_agent import root
from agents.sub_agents import sub_agents

auditors = ParallelAgent(
    name="auditors",
    description="Two auditors forming their observability points in parallel.",
    sub_agents=sub_agents,
)

#: the name ADK looks up in `agents.agent`
root_agent = SequentialAgent(
    name="setup",
    description=("Assign two personas from the card, then let each auditor "
                 "declare blind norms and map them onto the dataset."),
    sub_agents=[root, auditors],
)

__all__ = ["root_agent"]
