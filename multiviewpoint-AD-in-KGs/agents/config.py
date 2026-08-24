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

from tools.declare_semantics import declare_semantics
from tools.describe_relation import describe_relation
from tools.list_relations import list_relations
from tools.run_scorer import run_scorer
from tools.sample import sample

MODEL_NAME = "gemini-3.5-flash-lite"
MODEL = Gemini(model=MODEL_NAME)

#: tool budget per agent. The final reply consumes a step and an error retry
#: costs another, so the wall sits above the number the prompt asks for.
BUDGET = 8

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
VIEWPOINT_TOOLS = PROFILER_TOOLS + [declare_semantics, run_scorer]

#: Tools that produce a SCORE. These wait behind declare_semantics; everything
#: else -- the profiler now, a knowledge base later -- stays open, because a
#: frame is derived from facts and cannot be derived from its own answer.
GATED_TOOLS = {"run_scorer"}
