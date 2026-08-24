"""Two viewpoint agents on Google ADK, with a root that writes their goals.

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
    python scripts/5_run_agents.py

WHAT EACH AGENT CAN REACH
    root         list_relations, describe_relation, sample
    viewpoint    the same three, plus run_scorer

The root has no scorer on purpose. If it could score, it could write goals by
looking at what happens to score well -- choosing the question from the answer.
It writes goals from structure alone.
"""
import json
import pathlib
import re
import sys

# This file is imported by `adk run agents` with agents/ as the working root,
# so the repo root has to go on the path before the project imports below.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google.adk.agents.llm_agent import Agent  # noqa: E402
from google.adk.agents.parallel_agent import ParallelAgent  # noqa: E402
from google.adk.agents.sequential_agent import SequentialAgent  # noqa: E402
from google.adk.models.google_llm import Gemini  # noqa: E402

from loaders.active import DATASET  # noqa: E402
from tools.declare_semantics import declare_semantics, store_key  # noqa: E402
from tools.describe_relation import describe_relation  # noqa: E402
from tools.list_relations import list_relations  # noqa: E402
from tools.run_scorer import run_scorer  # noqa: E402
from tools.sample import sample  # noqa: E402

MODEL_NAME = "gemini-3.5-flash-lite"
MODEL = Gemini(model=MODEL_NAME)

#: tool budget per agent. The final reply consumes a step and an error retry
#: costs another, so the wall sits above the number the prompt asks for.
BUDGET = 8

GOAL_KEYS = ("goal_a", "goal_b")
SPEC_KEYS = ("spec_a", "spec_b")
SEM_KEYS = ("sem_a", "sem_b")
VIEWPOINT_NAMES = ("viewpoint_a", "viewpoint_b")

PROFILER_TOOLS = [list_relations, describe_relation, sample]

#: Tools that produce a SCORE. These wait behind declare_semantics; everything
#: else -- the profiler now, a knowledge base later -- stays open, because a
#: frame is derived from facts and cannot be derived from its own answer.
GATED_TOOLS = {"run_scorer"}


# --------------------------------------------------------------------------- #
# Parsing helpers. ADK hands back text; these pull the JSON out of it.        #
# --------------------------------------------------------------------------- #

def _first_json_object(text: str):
    """First balanced {...} in the text, parsed. None if there isn't one.

    Models fence their JSON, prefix it with prose, or both. Scanning for a
    balanced object is more forgiving than a regex and cheaper than a retry.
    """
    if not text:
        return None
    for start in (m.start() for m in re.finditer(r"\{", text)):
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _last_text(callback_context) -> str:
    """The final text this agent produced in this invocation."""
    out = []
    for event in callback_context.session.events:
        if event.invocation_id != callback_context.invocation_id:
            continue
        if event.author != callback_context.agent_name:
            continue
        content = getattr(event, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                out.append(part.text)
    return out[-1] if out else ""


# --------------------------------------------------------------------------- #
# The root -- writes both goals                                               #
# --------------------------------------------------------------------------- #

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
    raw = str(callback_context.state.get("goals_raw") or "") or _last_text(callback_context)
    parsed = _first_json_object(raw) or {}
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
)


# --------------------------------------------------------------------------- #
# The viewpoint agents -- one goal each, no channel between them              #
# --------------------------------------------------------------------------- #

VIEWPOINT_INSTRUCTION = f"""\
You are auditing a knowledge graph called {DATASET.NAME} for facts that are
wrong.

YOUR GOAL:
GOAL_SLOT

FIRST, before you may score anything, call declare_semantics to say what your
goal treats as NORMAL, what a violation of that looks like, and which relations
you are talking about. Look at the graph with the profiler first if that helps
you decide -- but decide before you score. A scorer will refuse you until you
have, and that is deliberate: a frame chosen after seeing scores is only a
description of the scores.

THEN work out how to look for it. You may run a scorer, read the triples it
flagged, and run a different one if you are not convinced. Judge those triples
against the frame you declared -- nothing will tell you whether a flag was
right, and no answer key exists for you to consult.

Scorers available to run_scorer:
  plausibility     what a trained embedding model makes of the triple
  neighbourhood    whether the two ends of the triple share any connections

You may keep one, keep both, or discard one. Say which, and why, in terms of
your goal.

Use at most {BUDGET - 2} tool calls, then answer.

Answer with JSON only:
  {{"scorer": "<name>", "budget": <fraction between 0 and 0.5>,
    "why": "<one or two sentences>",
    "summary": "<what run_scorer told you: how many flagged, the worst triple>"}}
"""


def require_semantics(tool, args, tool_context):
    """before_tool_callback: no scoring until this agent has declared a frame.

    Returning a dict makes ADK skip the tool and hand the dict back as its
    response, so the agent reads this as an ordinary tool error and can fix
    itself. Returning None lets the call through.

    Enforced here rather than asked for in the prompt: a prompt that says
    "declare first" is a suggestion, and the ordering is the only thing making
    the frame a commitment instead of a rationalisation.
    """
    if tool.name not in GATED_TOOLS:
        return None                       # facts are never gated, only scores
    if tool_context.state.get(store_key(tool_context.agent_name)):
        return None
    return {"result": (
        "ERROR: declare_semantics first. State what NORMAL means under your "
        "goal, what a violation looks like, and which relations you are "
        "talking about. Only then will a scorer answer you.")}


def make_capture_spec(spec_key: str):
    """after_agent_callback: write this agent's spec to state ourselves.

    Same reason as split_goals -- output_key alone has been seen to miss.
    """
    def capture_spec(callback_context):
        raw = str(callback_context.state.get(spec_key + "_raw") or "") \
            or _last_text(callback_context)
        parsed = _first_json_object(raw)
        callback_context.state[spec_key] = json.dumps(parsed) if parsed else ""
        return None

    return capture_spec


def make_viewpoint(name: str, goal_key: str, spec_key: str) -> Agent:
    """One viewpoint agent, bound to one goal key and one spec key.

    A factory rather than two hand-written agents so the two cannot drift
    apart. Any asymmetry between them would confound the experiment: the only
    difference here is WHICH KEY each reads and writes.
    """
    return Agent(
        model=MODEL,
        name=name,
        description=("Derives one viewpoint -- which scorer, at what budget -- "
                     "from a single goal, and reports nothing else."),
        instruction=VIEWPOINT_INSTRUCTION.replace("GOAL_SLOT", "{" + goal_key + "}"),
        tools=PROFILER_TOOLS + [declare_semantics, run_scorer],
        # The gate reads the CALLER's name, so one callback serves both twins
        # and neither can satisfy the other's precondition.
        before_tool_callback=require_semantics,
        # Strips PREVIOUS turns, so a second question in one `adk run` session
        # cannot leak an earlier run's specs into this one.
        include_contents="none",
        output_key=spec_key + "_raw",
        after_agent_callback=make_capture_spec(spec_key),
    )


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
