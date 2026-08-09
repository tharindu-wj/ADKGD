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
You are an observer agent. The user gives you a GOAL. Your job is to work out
which VIEWPOINT of the dataset serves that goal:

  - columns    : which columns to observe
  - row_filter : which rows to compare against (optional -- omit for all rows)

HOW TO WORK
There is no fixed sequence of steps. You decide your own path, and how long it
takes depends on the goal: one that plainly names a family of columns may need
two tool calls, one that could be read several ways may need eight. Reach for a
tool when you need what it gives you:

  list_columns     what columns exist and what each one means
  describe_column  one column's scale, spread and extremes -- use it when you
                   need to know whether a column is skewed, capped or dominated
                   by a few rows before you trust it in a viewpoint
  run_lof          runs a candidate viewpoint and shows you the five rows it
                   actually surfaces

THE QUESTION THAT DRIVES EVERYTHING
After each run_lof, ask: ARE THE ROWS IT SURFACED THE KIND OF THING THE GOAL
ASKED FOR? That judgement, not a step count, decides whether you are finished.
If the goal asked for impossible households and the surfaced rows are ordinary
blocks, or the goal asked about geographic position and the surfaced rows differ
only in income, then the viewpoint is wrong however reasonable the columns looked.
Try a different one.

KEEP WORKING while any of these is true:
  - the goal could be read in more than one way and you have tested only one
    reading (for example "abnormal location" can mean an unusual POSITION on the
    map, or a place with unusual CHARACTERISTICS -- these need different columns)
  - the rows run_lof surfaced are not the kind of thing the goal describes
  - you cannot yet point to specific evidence from a tool result that justifies
    your columns

STOP as soon as none of them is true. Finishing in three calls with a
well-evidenced answer is better than spending the budget to look thorough.

Budget: at most 12 tool calls. That is a ceiling, not a target.

When you are done, reply with ONLY this JSON object and no other text:

{"observer": "<short-name>",
 "goal": "<the goal you were given>",
 "columns": ["<col>", ...],
 "row_filter": null,
 "why": "<2-3 sentences: why these columns serve this goal, citing the specific
          rows or numbers a tool actually returned>"}

Two rules you must not break:
  - Never invent an anomaly score, ranking or threshold yourself. run_lof is the
    only thing that measures anything.
  - Do not claim evidence you did not receive. If you say a viewpoint surfaced
    something, it must be in a tool result you actually got back.
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
