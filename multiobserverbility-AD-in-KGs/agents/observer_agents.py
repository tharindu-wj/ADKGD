"""The two observers: blind norms first, then the data, then a scope.

A factory rather than two hand-written agents so the twins cannot drift
apart -- the ONLY difference between them is which persona they carry and
which state keys they write. Any other asymmetry would confound the
experiment.

The two-phase separation is enforced twice over: the phase gate refuses every
data tool until the observer's norms exist, and the run script later verifies
from the trace that declare_semantics really did precede the first look.
"""
from google.adk.agents.llm_agent import Agent

from agents import telemetry
from agents.config import MODEL, OBSERVER_TOOLS, OBSERVER_TOOL_BUDGET
from agents.phase_gate import keep_norms_blind
from loaders.active import DATASET
from tools._observers import OBSERVER_NAMES, state_key

OBSERVER_INSTRUCTION = f"""\
You are an observer. You will eventually judge facts for anomalies; today you
establish your observability point -- the position you will judge from.

All you know about the dataset is this:
  "{DATASET.CARD}"

WHO YOU ARE:
PERSONA_SLOT

Work in two strict phases.

PHASE 1 -- DECLARE YOUR NORMS, before looking at anything.
From your persona and what you know about the world, call declare_semantics:
what a normal fact of this domain looks like to you, what YOU flag as
anomalous (which, depending on who you are, may include things that are
factually true), and what you deliberately let pass that another judge might
flag. Speak in world terms -- you have not seen the dataset, and every data
tool will refuse you until your norms are recorded.

PHASE 2 -- MAP YOUR NORMS ONTO THE DATASET.
Once your norms are recorded the data opens. Look at what this graph actually
records -- describe_dataset first, then whatever you need -- and call
select_scope with the relations your norms apply to, saying which norm makes
each of them yours. Your norms are fixed; only the mapping is yours to
choose now.

PHASE 3 -- FIND AND JUDGE.
You cannot read the whole graph, so assistants sweep it for you: call
find_suspects with a scanner whose kind of suspicious MATCHES YOUR NORMS
(its description lists the menu), and say which norm it serves. Then judge --
every candidate you are served gets a verdict through submit_verdicts, by
YOUR norms alone: anomaly, ok, out_of_scope, or unsure, each with one
sentence of why. A candidate an assistant found suspicious can still be ok
by your norms, and a fact that is literally true can still be an anomaly by
them -- the assistants find, but only you judge.

You are done when every served candidate is judged. Use at most
{OBSERVER_TOOL_BUDGET} tool calls in all.
"""


def make_observer(name: str) -> Agent:
    """One observer, bound to its own persona, norms and scope keys."""
    persona_key = state_key("persona", name)
    return Agent(
        model=MODEL,
        name=name,
        description=("Declares its own norms blind, then maps them onto the "
                     "dataset's vocabulary as a scope."),
        # {persona_N} is filled from session state -- the root wrote it there.
        instruction=OBSERVER_INSTRUCTION.replace(
            "PERSONA_SLOT", "{" + persona_key + "}"),
        tools=OBSERVER_TOOLS,
        # The gate reads the CALLER's name, so one callback serves both twins
        # and neither can satisfy the other's precondition.
        before_tool_callback=keep_norms_blind,
        include_contents="none",
        after_model_callback=telemetry.record_response,
        on_model_error_callback=telemetry.record_error,
    )


observers = [make_observer(name) for name in OBSERVER_NAMES]
