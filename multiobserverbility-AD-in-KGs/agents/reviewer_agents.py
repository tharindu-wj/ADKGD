"""The second-opinion phase: each observer returns to judge a further batch.

Same observer, same persona, same recorded norms -- a different ADK agent
name only because one tree cannot hold two agents called observer_1. The
instruction does not say where the batch came from, and review_candidates
serves it with a neutral note: an opinion anchored on another opinion is one
opinion twice.

This phase exists because two observers reading different shortlists rarely
meet on the same fact -- and a fact judged by BOTH, differently, is the
architecture's product.
"""
from google.adk.agents.llm_agent import Agent

from agents import telemetry
from agents.config import MODEL
from loaders.active import DATASET
from tools._observers import REVIEWER_SUFFIX, OBSERVER_NAMES, state_key
from tools.review_candidates import review_candidates
from tools.submit_verdicts import submit_verdicts

REVIEWER_INSTRUCTION = f"""\
You are an observer of a knowledge graph called {DATASET.NAME}, mid-observation.
Your norms are declared and fixed; you have already judged one set of
candidates.

WHO YOU ARE:
PERSONA_SLOT

A further set of candidates has been surfaced for your review. Call
review_candidates to receive them, then give EVERY one a verdict through
submit_verdicts -- by your norms alone, exactly as before: anomaly, ok,
out_of_scope, or unsure, each with one sentence of why.

You are done when every candidate from review_candidates is judged, or when
it tells you nothing awaits you. Use at most 6 tool calls.
"""


def make_reviewer(principal: str) -> Agent:
    """The returning observer for one principal."""
    persona_key = state_key("persona", principal)
    return Agent(
        model=MODEL,
        name=principal + REVIEWER_SUFFIX,
        description="The same observer, judging a further batch by its norms.",
        instruction=REVIEWER_INSTRUCTION.replace(
            "PERSONA_SLOT", "{" + persona_key + "}"),
        tools=[review_candidates, submit_verdicts],
        include_contents="none",
        after_model_callback=telemetry.record_response,
        on_model_error_callback=telemetry.record_error,
    )


reviewers = [make_reviewer(name) for name in OBSERVER_NAMES]
