"""The tool index: tool name -> function. This is what makes them "tools".

WHY THIS FILE EXISTS
--------------------
It is the seam both orchestrations bind to. The hand-written loop
(orchestrator_custom.py) and the LangChain one to come both import TOOLS from
here, so there is exactly one list of what the agent can do. Two copies would
drift silently -- a tool missing from one copy comes back as ordinary "ERROR:"
text, the agent reads it, adapts, and finishes with a run marked "completed".
The comparison between orchestrations would be quietly invalid.

ADDING A TOOL
-------------
    1. one new tools/<name>.py with one function returning a string
    2. one line here
    3. one line in the Tools: block of each agent that should see it
       (agent_custom_single/build_system_prompt.py; the ADK agents list tools
       directly in their agent.py)
Ask first whether the tool leaks per-entity information -- that decision
outlives the MVP (PROJECT_SPEC 6.5).

Imports run one way only: registry -> tool files -> data/. No tool imports
another tool, and nothing in tools/ imports an agent or an LLM backend, so
adding an agent is a pure addition.
"""

from tools.compare_viewpoints import compare_viewpoints
from tools.describe_column import describe_column
from tools.list_columns import list_columns
from tools.run_lof_per_viewpoint import run_lof_per_viewpoint

#: The dispatch table: tool name -> function. This is what makes them "tools".
TOOLS = {
    "list_columns": list_columns,
    "describe_column": describe_column,
    "run_lof_per_viewpoint": run_lof_per_viewpoint,
    "compare_viewpoints": compare_viewpoints,
}
