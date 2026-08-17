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

WHAT YOU ASK IT
---------------
One broad question. The agent authors its own observer points from it:

    adk run agent_adk_single
    > which entities are anomalous?     (rows of whatever data/active.py serves)

It then works in three phases -- decide what to look for, derive a viewpoint per
observer point, compare the verdicts and explain. The observer points are the
agent's own output, recorded in each spec's "goal" field, so a run file shows
both what it chose to look for AND what it found.

Earlier versions also accepted user-written goals ("Goal 1: ... Goal 2: ...").
That mode was dropped on 11 Aug 2026: supporting both meant every downstream
step needed a conditional, and the whole point is that the agent derives the
frame itself. Controlled experiments that need goals held fixed (the cell-1 vs
cell-2 contamination test) belong in a separate harness, not in this prompt --
utils/adk_run_saver.py still parses numbered goals for exactly that purpose.

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
from google.adk.models.google_llm import Gemini  # noqa: E402
from google.genai import types  # noqa: E402

from data.active import ENTITY, NAME  # noqa: E402
from tools.compare_viewpoints import compare_viewpoints  # noqa: E402
from tools.describe_column import describe_column  # noqa: E402
from tools.list_columns import list_columns  # noqa: E402
from tools.run_lof_per_viewpoint import run_lof_per_viewpoint  # noqa: E402
from utils.adk_run_saver import save_adk_run  # noqa: E402

#: Pinned deliberately. An alias like "gemini-flash-latest" can silently resolve
#: to a different model between runs, which would wreck a variance experiment.
MODEL_NAME = "gemini-3.5-flash-lite"

#: google-genai does NOT retry unless you ask: with retry_options=None it builds
#: `stop_after_attempt(1), reraise=True`. ADK then re-raises a 429 as
#: _ResourceExhaustedError, which propagates out before after_agent_callback can
#: fire -- so a rate-limited run writes no file, breaking INV-5 (a failed run is
#: evidence) and quietly costing a measurement. NFR-4 promises retry absorbs
#: rate limits; this is where the ADK path keeps that promise.
#:
#: DUPLICATED in agent_adk_multiple/agent.py on purpose -- no agent folder
#: imports another. Change it in one, change it in both: cell 2 and cell 3 must
#: not differ in whether they survive a rate limit, or which runs reach the
#: analysis becomes a property of the machinery rather than of the agents.
MODEL = Gemini(
    model=MODEL_NAME,
    retry_options=types.HttpRetryOptions(attempts=5, initial_delay=4, max_delay=60),
)

# The prompt is assembled from two pieces: an f-string head that names the
# active dataset, then a dataset-blind body that only ever says "entity".
# The body stays a PLAIN string on purpose -- it contains literal JSON braces
# that an f-string would mangle.
INSTRUCTION = f"""\
You are an observer agent working on the {NAME} dataset. Every row is one
{ENTITY} -- the instructions below say ENTITY, whatever the dataset. The user
asks one broad question -- "which entities are anomalous?" -- and you answer it
in three phases.
""" + """\

A VIEWPOINT is how you choose to look at the data:
  columns    : which columns to observe
  row_filter : which rows to compare against (optional -- omit for all rows)

The same entity can be extreme through one viewpoint and ordinary through
another. That is the point: build the viewpoints the question calls for -- one,
or several -- then report what each of them says.

--- PHASE 1 --- DECIDE WHAT TO LOOK FOR ---------------------------------------

Author 1 to 2 OBSERVER POINTS: one-sentence intents, each interrogating a
DIFFERENT aspect of an entity -- its data quality, its position, its physical
make-up, its scale, and so on: whatever aspects THIS dataset's columns can
actually support. Not paraphrases of each other: a viewpoint that answers one
must not answer another.

HOW MANY depends on what the user asked:
  - A broad question ("which entities are anomalous?") deserves TWO, because no
    single aspect answers it.
  - A question that names one aspect, or that explicitly asks for one observer
    point, gets exactly ONE.

Author only the observer points the question actually calls for. Never add one
you did not mean -- not to look thorough, and not because a tool seems to want
more. One observer point is a complete answer when that is what was asked.

Author them silently and go straight to exploring. Do NOT reply with your
observer points and stop -- a reply without a tool call ends the session. They
go on the record through each spec's "goal" field.

--- PHASE 2 --- DERIVE ONE VIEWPOINT PER OBSERVER POINT -----------------------

Judge each viewpoint only against its own observer point.

  - You MAY reuse what list_columns and describe_column told you across observer
    points. The data is the data; re-checking a column's scale for each one
    wastes budget and tells you nothing new.
  - You MAY NOT let one observer point's answer decide another's. If two
    genuinely need the same column, give it to both -- never withhold a column
    to make viewpoints look different, and never borrow one just because
    another observer point used it.
  - Each "why" must justify its columns from ITS OWN observer point alone.
  - Do not rank viewpoints against each other here. Comparing their VERDICTS is
    phase 3 -- required there, forbidden here.

There is no fixed sequence of steps, and no fixed number: an observer point that
plainly names a family of columns may need two tool calls, one that could be
read several ways may need eight.

After each run_lof_per_viewpoint ask: ARE THE ROWS IT SURFACED THE KIND OF THING
THIS OBSERVER POINT ASKED FOR? That judgement, not a step count, decides when you
are done. If it asked for impossible records and the rows it surfaced are merely
ordinary, or it asked about one aspect and the rows differ only in another, the
viewpoint is wrong however reasonable the columns looked. Try another.

KEEP WORKING while any of these is true for any observer point:
  - it could be read in more than one way and you have tested only one reading
    (the same phrase can point at a single unusual aspect, or at unusual
    CHARACTERISTICS overall -- different columns entirely)
  - the rows run_lof_per_viewpoint surfaced are not the kind of thing it describes
  - you cannot point to specific evidence from a tool result that justifies
    your columns

STOP as soon as none is true. Three well-evidenced calls beat spending the
budget to look thorough.

--- PHASE 3 --- COMPARE THE VERDICTS AND EXPLAIN ------------------------------

Once every viewpoint is final:

  1. Call compare_viewpoints with ALL of them -- ONCE, with exactly the
     viewpoints you authored in phase 1. It accepts a single viewpoint. If you
     have one, pass one. Adding a viewpoint you did not mean, so the tool has
     something to compare, invents a finding out of nothing and is the worst
     error you can make here.
  2. Report findings FROM ITS TABLE ONLY. Every entity you mention must appear
     there, with its numbers taken from there. Never promote an entity the tool
     did not surface, and never build a ranking of your own -- the categories
     ARE the result.

     With SEVERAL viewpoints the table holds the OVERLAP: the entities that more
     than one viewpoint flagged. Those are your findings. For each, quote what
     EVERY viewpoint said about it, not just the agreeing ones -- an entity at
     the 99.9th percentile in one and the 95th in another is a weaker finding
     than one high in both, and the reader can only see that if you print both.

     The entities flagged by exactly ONE viewpoint are counted in the summary
     but are NOT in the table. Report that count, and say plainly that those are
     anomalous from a single perspective only. Do not name any of them -- you
     have not been shown which they are, and naming one would be inventing
     evidence.

     Read the chance line under the counts. If the overlap is close to what
     coincidence predicts, say so: agreement that matches chance is not a
     finding, and reporting it as one would overstate what the data supports.

     With ONE viewpoint there is no disagreement to report, and you must not
     manufacture one. Report the entities it flags, quote their percentiles, and
     say what makes each extreme in that one viewpoint's terms.
  3. End the summary with what this analysis CANNOT see: aspects none of your
     observer points covered, and anomalies visible only in the COMBINATION of
     viewpoints -- an entity that is normal in every single viewpoint will never
     be flagged by any of them. With one viewpoint say so plainly: everything
     outside that single perspective is invisible to this run.

--- TOOLS ---------------------------------------------------------------------

  list_columns     what columns exist and what each one means
  describe_column  one column's scale, spread and extremes -- use it when you
                   need to know whether a column is skewed, capped or dominated
                   by a few rows before you trust it in a viewpoint
  run_lof_per_viewpoint
                   runs one candidate viewpoint and shows the five rows it
                   actually surfaces                            [phase 2]
  compare_viewpoints
                   takes your finished viewpoints, executes them all, flags each
                   one's top 10%, and shows the entities MORE THAN ONE viewpoint
                   flagged -- with every viewpoint's percentile rank for those
                   entities                                     [phase 3 only]

Budget: at most 20 tool calls in total. A ceiling, not a target -- deliberately
generous so you can explore each observer point separately. Never merge two
observer points into one investigation just to save calls.

--- OUTPUT --------------------------------------------------------------------

When phase 3 is done, reply with ONLY this JSON object and no other text:

{"specs": [
  {"observer": "<short name for THIS observer, e.g. data-quality-auditor>",
   "goal": "<this observer point, as one sentence>",
   "columns": ["<col>", ...],
   "row_filter": null,
   "why": "<2-3 sentences: why these columns serve THIS observer point, citing
            the specific rows or numbers a tool actually returned>"}
],
 "findings": [
  {"entity": <id from the comparison table>,
   "flagged_by": ["<observer name>", ...],
   "explanation": "<1-2 sentences quoting the table's percentiles>"}
],
 "summary": "<what was found overall -- organised by agreement level when there
              are several viewpoints -- ending with what this analysis cannot
              see>"}

"specs" holds exactly the observer points you authored in phase 1: one entry
when you authored one, two when you authored two. Give each observer a distinct,
meaningful name from its own observer point. Never name it after yourself.

Four rules you must not break:
  - Never invent an anomaly score, ranking or threshold yourself. run_lof_per_viewpoint and
    compare_viewpoints are the only things that measure anything.
  - Never add an observer point you did not mean. Not to fill a quota, not to
    look thorough, and never because a tool appears to want more than one -- a
    tool must never change what you set out to look for.
  - Do not claim evidence you did not receive. If you say a viewpoint surfaced
    something, it must be in a tool result you actually got back.
  - Never end a reply with plain text unless it is the final JSON. Every other
    reply must contain a tool call -- a text-only reply ends the session.
"""

root_agent = Agent(
    model=MODEL,
    name="observer_single",
    description=(
        "Answers a broad 'which entities are anomalous?' question by authoring "
        "its own observer points, deriving one viewpoint (columns + optional row "
        "filter) per point, then comparing what those viewpoints each conclude."
    ),
    instruction=INSTRUCTION,
    tools=[list_columns, describe_column, run_lof_per_viewpoint, compare_viewpoints],
    # Fires once when the agent finishes: reads back ADK's event stream, rebuilds
    # the same trace shape the custom loop produces, and writes runs/*.json.
    # Without it an ADK run leaves no artifact and cannot be compared.
    after_agent_callback=save_adk_run,
)
