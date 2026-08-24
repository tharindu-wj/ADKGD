"""The viewpoint agents: one goal each, and no channel between them.

`make_viewpoint` is a factory rather than two hand-written agents so the two
cannot drift apart. Any asymmetry between them would confound the experiment:
the only difference is WHICH KEY each reads and writes.

THE GATE lives here too. `require_semantics` refuses a scorer to any agent that
has not yet declared what it is looking for, so a frame is always a commitment
made before the evidence rather than a description of it. It is enforced in
code because a prompt saying "declare first" is only a suggestion.
"""
import json

from google.adk.agents.llm_agent import Agent

from agents.config import BUDGET, GATED_TOOLS, MODEL, VIEWPOINT_TOOLS
from agents.parsing import first_json_object, last_text
from loaders.active import DATASET
from tools.declare_semantics import store_key

VIEWPOINT_INSTRUCTION = f"""\
You are auditing a knowledge graph called {DATASET.NAME} for facts that are
wrong.

YOUR GOAL:
GOAL_SLOT

FIRST, before you may score anything, call declare_semantics. Look at the graph
with the profiler first if that helps you decide -- but decide before you score.
A scorer will refuse you until you have, and that is deliberate: a frame chosen
after seeing scores is only a description of the scores.

Write the frame from what you KNOW ABOUT THE WORLD, not from the graph. You
know what a country is, what a continent is, what it means for one place to
contain another. Put that knowledge in, and say what follows from it. A frame
that only says "normal is when the fact is correct" restates the question and
is worth nothing -- name the kinds of thing involved, and give the rule that
holds no matter which places are named.

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
            or last_text(callback_context)
        parsed = first_json_object(raw)
        callback_context.state[spec_key] = json.dumps(parsed) if parsed else ""
        return None

    return capture_spec


def make_viewpoint(name: str, goal_key: str, spec_key: str) -> Agent:
    """One viewpoint agent, bound to one goal key and one spec key."""
    return Agent(
        model=MODEL,
        name=name,
        description=("Derives one viewpoint -- which scorer, at what budget -- "
                     "from a single goal, and reports nothing else."),
        instruction=VIEWPOINT_INSTRUCTION.replace("GOAL_SLOT", "{" + goal_key + "}"),
        tools=VIEWPOINT_TOOLS,
        # The gate reads the CALLER's name, so one callback serves both twins
        # and neither can satisfy the other's precondition.
        before_tool_callback=require_semantics,
        # Strips PREVIOUS turns, so a second question in one `adk run` session
        # cannot leak an earlier run's specs into this one.
        include_contents="none",
        output_key=spec_key + "_raw",
        after_agent_callback=make_capture_spec(spec_key),
    )
