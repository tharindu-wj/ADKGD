"""The tree ADK loads. Assembly only -- every part is defined elsewhere.

    SequentialAgent "audit"              <- fixed order, never runtime-chosen
      |- Agent          "root"           <- writes BOTH goals from the profile
      |- ParallelAgent  "viewpoints"     <- concurrent, isolated branches
           |- Agent     "viewpoint_a"    <- reads goal_a only
           |- Agent     "viewpoint_b"    <- reads goal_b only

ParallelAgent gives each sub-agent its own branch path and filters sibling
events, so neither viewpoint agent can see the other's tool calls or spec.
Session state is NOT branch-scoped, so the rest is on us: neither instruction
names the other's state key.

    adk run agents
    python scripts/3_run_agentic_detector.py

WHERE THINGS LIVE
    agents/config.py            the model, budget, state keys, tool sets
    agents/parsing.py           pulling a structured answer out of model text
    agents/root_agent.py        the goal-writing agent and its prompt
    agents/viewpoint_agents.py  the auditor factory, its prompt, and the gate

This module stays thin on purpose: ADK imports `agents.agent` and reads
`root_agent`, so this is the front door, and a front door should show the shape
of the house rather than its furniture.

ONE NAME TO BE CAREFUL WITH. `root_agent` means two different things here: the
module `agents/root_agent.py`, and the SequentialAgent this file exports under
that name. They do not collide, but only because agents/__init__.py imports no
submodules -- so when ADK checks `hasattr(agents, "root_agent")` on the package
first, it finds nothing and moves on to this module, which is what it wants.
Import a submodule from __init__.py and ADK would find the MODULE under that
attribute instead. It recovers (it type-checks and falls through), but it logs a
warning on every start, so leave __init__.py importing nothing.
"""
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent

from agents.config import (BUDGET, GATED_TOOLS, GOAL_KEYS, MODEL, MODEL_NAME,
                           PROFILER_TOOLS, SEM_KEYS, SPEC_KEYS,
                           VIEWPOINT_NAMES, VIEWPOINT_TOOLS)
from agents.root_agent import root
from agents.viewpoint_agents import make_viewpoint

viewpoints = ParallelAgent(
    name="viewpoints",
    description="Two auditors working the same graph from different goals.",
    sub_agents=[make_viewpoint(n, g, s)
                for n, g, s in zip(VIEWPOINT_NAMES, GOAL_KEYS, SPEC_KEYS)],
)

#: What `adk run agents` picks up.
root_agent = SequentialAgent(
    name="audit",
    description="Write two goals, then audit the graph from both at once.",
    sub_agents=[root, viewpoints],
)

#: Re-exported so callers import the tree and its vocabulary from one place --
#: scripts/3_run_agentic_detector.py reads state by these keys.
__all__ = ["root_agent", "viewpoints", "root",
           "GOAL_KEYS", "SPEC_KEYS", "SEM_KEYS", "VIEWPOINT_NAMES",
           "MODEL", "MODEL_NAME", "BUDGET",
           "PROFILER_TOOLS", "VIEWPOINT_TOOLS", "GATED_TOOLS"]
