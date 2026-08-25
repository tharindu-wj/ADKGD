"""What the tree is made of. No behaviour, no prompts -- just the parts list.

Kept apart from the agents themselves so that changing a model, widening a
budget or handing out one more tool does not mean reading a page of prompt to
find where the change goes.

WHAT EACH AGENT CAN REACH
    root         PROFILER_TOOLS                 -- facts about the graph
    viewpoint    VIEWPOINT_TOOLS                -- the same, plus a frame and a scorer

The root has no scorer on purpose. If it could score, it could write goals by
watching what happens to score well -- choosing the question from the answer.
It writes goals from structure alone.
"""
from google.adk.models.google_llm import Gemini
from google.genai import types

from tools.declare_semantics import declare_semantics
from tools.describe_relation import describe_relation
from tools.list_relations import list_relations
from tools.run_scorer import run_scorer
from tools.sample import sample
from tools.submit_spec import submit_spec

MODEL_NAME = "gemini-3.5-flash-lite"

#: WAIT OUT A RATE LIMIT RATHER THAN LOSING THE RUN.
#:
#: The free tier allows 15 requests per minute per model. This tree needs 18-19
#: (root 4, each viewpoint 7-8), so the last requests were being refused with a
#: 429 -- and a refused request looks exactly like an agent that chose to stop.
#: That cost two prompt rewrites chasing a behaviour that was never happening.
#:
#: Trimming calls cannot fix it: two viewpoints at 6-7 each is 12-14 before the
#: root does anything. So the tree waits instead.
#:
#: 429 is already in google-genai's retriable status codes, but retry is OFF
#: unless HttpRetryOptions is passed -- `retry_args(None)` is a never-retry
#: strategy. The defaults are also too shallow here: 1s, 2s, 4s, 8s tops out
#: around 15s against a window the API itself says needs ~53s. These waits
#: (10, 20, 40, 70) cover it.
#:
#: This costs nothing on a run that stays under the limit. Only the refused
#: call waits; the rest proceed at full speed.
RETRY = types.HttpRetryOptions(
    attempts=5, initial_delay=10, max_delay=70, exp_base=2, jitter=1,
)

MODEL = Gemini(model=MODEL_NAME, retry_options=RETRY)

#: Tool budget per agent, as ASKED FOR in the prompt. Nothing enforces it --
#: ADK's own ceiling is max_llm_calls=500 -- but the agents broadly respect it.
#: Raised from 8 when submit_spec was added: a viewpoint now needs to profile,
#: declare a frame, score, and hand in, and at 6 an agent could spend its whole
#: allowance before it had anything to submit.
BUDGET = 10

#: One key per agent, per artifact. The suffix is what `make_viewpoint` binds
#: each twin to, and what `store_key` in declare_semantics derives from the
#: caller's name -- keep the _a / _b endings in step across all three.
GOAL_KEYS = ("goal_a", "goal_b")
SPEC_KEYS = ("spec_a", "spec_b")
SEM_KEYS = ("sem_a", "sem_b")
VIEWPOINT_NAMES = ("viewpoint_a", "viewpoint_b")

PROFILER_TOOLS = [list_relations, describe_relation, sample]

#: Both twins get an IDENTICAL list. Any asymmetry in what they can reach would
#: confound the experiment -- the only difference between them is their goal.
#: Note the shape: declare a frame, score against it, hand in. Every artifact
#: the run needs is written by a tool call, never scraped from a closing message.
VIEWPOINT_TOOLS = PROFILER_TOOLS + [declare_semantics, run_scorer, submit_spec]

#: Tools that produce a SCORE. These wait behind declare_semantics; everything
#: else -- the profiler now, a knowledge base later -- stays open, because a
#: frame is derived from facts and cannot be derived from its own answer.
GATED_TOOLS = {"run_scorer"}
