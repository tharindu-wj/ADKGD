"""Cell 1 -- single agent, single goal, on Google ADK.

WHAT THIS IS
------------
The same experiment as orchestrator_custom.py, driven by ADK instead of the
hand-written loop. One observer agent, the same three tools, the same dataset.
The point of having both is that the ORCHESTRATION becomes an experimental
variable: does the derived viewpoint change when only the machinery changes?

HOW IT DIFFERS FROM THE CUSTOM LOOP
-----------------------------------
    custom loop : we ask the model to reply with JSON naming a tool, and parse it
    ADK         : the model uses Gemini's NATIVE function calling; ADK builds the
                  tool schemas from our type hints and docstrings

So the tools are identical but the calling mechanism is not. That difference is
a confound to name in the write-up, not to hide.

WHERE THE GOAL COMES FROM
-------------------------
The instruction below is the observer's ROLE, fixed for every run. The GOAL --
the observer point -- is whatever you type as the first message:

    adk run agent_adk_single
    > find census block groups whose housing stock is abnormal

That split is deliberate: role in the instruction, purpose in the message, so a
single agent definition serves every observer point.

RUN IT
------
    cd multi-observerbility-AD
    adk run agent_adk_single      # terminal
    adk web                       # browser UI, shows every tool call visually
"""

import pathlib
import sys

# The shared tools/ and data/ packages live one level up, in the project root.
# ADK imports this file as part of the agent_adk_single package, so the project
# root is not guaranteed to be on sys.path -- put it there explicitly. This is
# what lets both orchestrations call the SAME tool functions rather than copies.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google.adk.agents.llm_agent import Agent  # noqa: E402

from tools.describe_column import describe_column  # noqa: E402
from tools.list_columns import list_columns  # noqa: E402
from tools.run_lof import run_lof  # noqa: E402
from utils.adk_run_saver import save_adk_run  # noqa: E402

#: Pinned deliberately. An alias like "gemini-flash-latest" can silently resolve
#: to a different model between runs, which would wreck a variance experiment.
MODEL = "gemini-3.5-flash-lite"

INSTRUCTION = """\
You are an observer agent. The user gives you a GOAL. Your job is to explore the
dataset with your tools and then define a VIEWPOINT that serves that goal:

  - columns    : which columns to observe
  - row_filter : which rows to compare against (optional -- omit for all rows)

Work in this order:
  1. list_columns to see what exists
  2. describe_column on the columns that sound relevant, to learn their scales
  3. run_lof to try a candidate viewpoint and see what it actually surfaces
  4. revise and try again if the result does not serve the goal

You have at most 10 tool calls. Evaluate at least one candidate with run_lof
before finishing.

When you are done, reply with ONLY this JSON object and no other text:

{"observer": "<short-name>",
 "goal": "<the goal you were given>",
 "columns": ["<col>", ...],
 "row_filter": null,
 "why": "<2-3 sentences: why these columns serve this goal, citing what run_lof showed>"}

Never invent an anomaly score yourself. run_lof is the only thing that scores.
"""

root_agent = Agent(
    model=MODEL,
    name="observer_single",
    description=(
        "Derives an anomaly-detection viewpoint (columns + optional row filter) "
        "from a stated goal, by exploring a dataset with tools."
    ),
    instruction=INSTRUCTION,
    tools=[list_columns, describe_column, run_lof],
    # Fires once when the agent finishes: reads back ADK's event stream, rebuilds
    # the same trace shape the custom loop produces, and writes runs/*.json.
    # Without it an ADK run leaves no artifact and cannot be compared.
    after_agent_callback=save_adk_run,
)
