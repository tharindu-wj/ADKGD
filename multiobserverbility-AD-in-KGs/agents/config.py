"""What the agents are made of. No behaviour, no prompts -- the parts list."""
from google.adk.models.google_llm import Gemini
from google.genai import types

from tools.assign_perspective import assign_perspective
from tools._observers import OBSERVER_NAMES, state_key
from tools.declare_semantics import declare_semantics
from tools.describe_dataset import describe_dataset
from tools.describe_relation import describe_relation
from tools.find_suspects import find_suspects
from tools.explain_term import explain_term
from tools.inspect_triples import inspect_triples
from tools.select_scope import select_scope
from tools.submit_verdicts import submit_verdicts

MODEL_NAME = "gemini-3.5-flash-lite"

#: Wait out a rate limit rather than losing the run. The free tier allows 15
#: requests per minute; retry is OFF in google-genai unless options are passed,
#: and its default backoff (1,2,4,8s) is far shorter than the ~53s window the
#: API asks for. A run that stays under the limit pays nothing.
RETRY = types.HttpRetryOptions(
    attempts=5, initial_delay=10, max_delay=70, exp_base=2, jitter=1,
)

MODEL = Gemini(model=MODEL_NAME, retry_options=RETRY)

#: tool-call budgets the prompts ask for. Nothing enforces them; guidance.
ROOT_TOOL_BUDGET = 4         # two assign_perspective calls + retry room
OBSERVER_TOOL_BUDGET = 16   # declare, look, select, then find and judge

#: the dataset tools -- open in phase 2 only, the phase gate holds the door
DATA_TOOLS = [describe_dataset, describe_relation, explain_term, inspect_triples]

#: the root sees the dataset CARD in its instruction and nothing else --
#: no data tools AT ALL, so its personas cannot be schema-shaped
ROOT_TOOLS = [assign_perspective]

OBSERVER_TOOLS = ([declare_semantics, select_scope] + DATA_TOOLS
                   + [find_suspects, submit_verdicts])

#: session-state keys the run script reads after the tree finishes
PERSONA_KEYS = tuple(state_key("persona", n) for n in OBSERVER_NAMES)
NORMS_KEYS = tuple(state_key("norms", n) for n in OBSERVER_NAMES)
SCOPE_KEYS = tuple(state_key("scope", n) for n in OBSERVER_NAMES)
SCANNER_KEYS = tuple(state_key("scanners", n) for n in OBSERVER_NAMES)
SERVED_KEYS = tuple(state_key("served", n) for n in OBSERVER_NAMES)
VERDICT_KEYS = tuple(state_key("verdicts", n) for n in OBSERVER_NAMES)
