"""The root: reads the dataset CARD, assigns two differing personas, stops.

It has NO data tools -- not as a policy it is asked to follow, but as a tool
list it cannot exceed. Everything it knows about the dataset is the card in
its instruction, so the personas it writes cannot be shaped by the schema.
The root is the circumstance that makes the two observers different people;
it is not the author of their views -- each observer writes its own norms
from its persona, later, still blind.
"""
from google.adk.agents.llm_agent import Agent

from agents import telemetry
from agents.config import MODEL, ROOT_TOOLS, ROOT_TOOL_BUDGET
from loaders.active import DATASET

ROOT_INSTRUCTION = f"""\
Two observers are about to judge a dataset for anomalies. You decide what KIND
of judge each one is.

All anyone knows about the dataset, including you, is this:
  "{DATASET.CARD}"

Give each observer a PERSONA: the values and stance it brings to anything it
looks at, in two or three sentences addressed to it as "you". A persona
describes a way of judging -- it must never mention particular data,
attributes, or statistics, and it must work for any dataset fitting the
description above.

The two personas must be GENUINELY DIFFERENT ways of judging the same things
-- different enough that the two observers could disagree about the very same
fact. Perspectives that differ only in which topics they mention are the
same judge twice.

Call assign_perspective once for observer_1 and once for observer_2.
Nothing is recorded until those calls succeed, and you are not finished until
both have. Use at most {ROOT_TOOL_BUDGET} tool calls in all.
"""

root = Agent(
    model=MODEL,
    name="root",
    description=("Assigns both observers their personas from the dataset "
                 "card alone. Does nothing else."),
    instruction=ROOT_INSTRUCTION,
    tools=ROOT_TOOLS,
    # A fresh context every turn: a second question in one `adk web` session
    # cannot leak an earlier run's personas into this one.
    include_contents="none",
    # Observation only -- both return None, so nothing about the run changes.
    after_model_callback=telemetry.record_response,
    on_model_error_callback=telemetry.record_error,
)
