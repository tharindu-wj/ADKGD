"""The root agent: reads the graph's shape, writes two goals, stops.

It never scores anything and never sees a score. That is the point of it --
a goal written by something that can watch what scores well is a question
chosen from its answer.

Its whole output is two sentences. `split_goals` puts one in each viewpoint's
state key, which is the only channel between the root and the auditors.
"""
import json

from google.adk.agents.llm_agent import Agent

from agents.config import BUDGET, GOAL_KEYS, MODEL, PROFILER_TOOLS
from agents import telemetry
from agents.parsing import first_json_object, last_text
from loaders.active import DATASET

ROOT_INSTRUCTION = f"""\
You are preparing an audit of a knowledge graph called {DATASET.NAME}, looking
for facts that are wrong.

Your job is NOT to find them. Your job is to write TWO GOALS, which two other
auditors will each work on independently.

First look at the graph with your tools. Then write the goals.

A goal is ONE SENTENCE stating a PURPOSE someone could hold. It must not name a
method, a threshold, or a number -- those are for the auditors to decide.

  good: "Audit this graph for places recorded as bordering one another that
         are nowhere near each other."
  bad:  "Find facts whose neighbourhood support is below 0.1."   (a method)
  bad:  "Flag the least plausible 10 percent."                   (a threshold)
  bad:  "Audit only the African triples."                        (a slice)

Both goals must apply to the WHOLE graph. Two auditors given separate halves
cannot disagree, and disagreement is the point.

Use at most {BUDGET - 2} tool calls, then answer.

Answer with JSON only:
  {{"goals": ["<first goal>", "<second goal>"], "why": "<one sentence>"}}
"""


def split_goals(callback_context):
    """after_agent_callback on the root: put one goal in each viewpoint's key.

    Written here rather than relying on output_key alone, which only fires when
    an event is marked final -- the same signal that has been observed to drop
    a result silently in the sibling project. Parsing the event stream and
    writing state back keeps state and the run file true to each other.

    Sets both keys to "" when nothing parses, so a viewpoint agent sees an
    unambiguous absence rather than half a sentence.
    """
    raw = str(callback_context.state.get("goals_raw") or "") or last_text(callback_context)
    parsed = first_json_object(raw) or {}
    goals = parsed.get("goals") or []

    for key, goal in zip(GOAL_KEYS, list(goals) + ["", ""]):
        callback_context.state[key] = goal if isinstance(goal, str) else ""
    callback_context.state["goals_json"] = json.dumps(parsed) if parsed else ""
    return None


root = Agent(
    model=MODEL,
    name="root",
    description="Writes two audit goals for this graph, and nothing else.",
    instruction=ROOT_INSTRUCTION,
    tools=PROFILER_TOOLS,
    include_contents="none",
    output_key="goals_raw",
    after_agent_callback=split_goals,
    # Observation only -- both return None, so nothing about the run changes.
    after_model_callback=telemetry.record_response,
    on_model_error_callback=telemetry.record_error,
)
