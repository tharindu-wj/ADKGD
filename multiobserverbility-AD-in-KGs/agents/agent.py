"""What ADK loads: the observation tree.

    SequentialAgent "observation"            <- fixed order, never runtime-chosen
      |- Agent          "root"         <- personas from the card alone
      |- ParallelAgent  "observers"     <- concurrent, isolated branches
           |- Agent  "observer_1"     <- blind norms -> scope -> find -> judge
           |- Agent  "observer_2"

ParallelAgent gives each observer its own branch and filters sibling events,
so neither sees the other's calls. Session state is NOT branch-scoped -- the
tools key every artifact by the CALLER's name, and neither instruction names
the other's keys.
"""
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent

from agents.reviewer_agents import reviewers
from agents.root_agent import root
from agents.observer_agents import observers as observer_list

observers = ParallelAgent(
    name="observers",
    description="Two observers watching the same graph from their own norms.",
    sub_agents=observer_list,
)

#: the second-opinion phase runs only after BOTH observers have finished --
#: SequentialAgent is what guarantees that ordering
second_opinions = ParallelAgent(
    name="second_opinions",
    description="Each observer judges the other's flags, blind.",
    sub_agents=reviewers,
)

#: the name ADK looks up in `agents.agent`
root_agent = SequentialAgent(
    name="observation",
    description=("Assign two personas from the card; each observer declares "
                 "blind norms, maps them to a scope, judges what its chosen "
                 "assistants surface, then judges the other's flags blind."),
    sub_agents=[root, observers, second_opinions],
)

__all__ = ["root_agent"]
