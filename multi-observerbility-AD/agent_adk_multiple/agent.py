"""Cells 2.5 and 3 -- TWO observer agents, on Google ADK.

WHAT THIS IS
------------
agent_adk_single puts ONE mind on one or two observer points. Its phase-2 rule
says "you MAY NOT let one observer point's answer decide another's" -- a request
addressed to a single context that can see both goals and both sets of results.
Nothing enforces it and nothing can check it.

This file makes the same prohibition STRUCTURAL. Two observer agents, one goal
each, no channel between them; a third agent compares what they concluded.
That buys exactly one thing -- a verifiable independence claim -- and no new
capability. Same four tools, same LOF, same k=20.

WHICH CELL YOU GET DEPENDS ON WHAT YOU TYPE
-------------------------------------------
    Goal 1: <intent>                     -> cell 2.5, BOTH observers get it
    Goal 1: <intent>  Goal 2: <intent>   -> cell 3,   one each

Cell 2.5 is the redundancy control and it is not a throwaway. Two minds asked
the SAME question show the most agreement you should ever expect, which is the
ceiling that cell 3's overlap has to be read against. compare_viewpoints
already prints the chance floor; 2.5 supplies the other bound.

    adk run agent_adk_multiple
    > Goal 1: find records that cannot describe a real place
    > Goal 2: find neighbourhoods that do not fit their region

THE PATTERN (Google Cloud: Multi-Agent Parallel inside Multi-Agent Sequential)
-----------------------------------------------------------------------------
    SequentialAgent "observers_two"     <- fixed order, never runtime-chosen
      |- ParallelAgent "observers"      <- concurrent, isolated branches
      |    |- observer_a                <- sees goal_a only
      |    +- observer_b                <- sees goal_b only
      +- comparer                       <- reads both specs, cannot revise them

Deliberately deterministic. Coordinator, Swarm and Loop are all rejected, and
not on cost: if an LLM chooses the routing then the wiring varies between runs,
and "same goals, one mind vs two minds" stops being answerable. See
AGENTIC_DESIGN.md for the full pattern mapping.

WHAT ADK ISOLATES -- VERIFIED, NOT ASSUMED (ADK 2.6.3)
------------------------------------------------------
ParallelAgent gives each sub-agent its own branch path, and events are filtered
by branch when a request is built (flows/llm_flows/contents.py). Measured
against the branch strings this tree actually produces:

    observer_a  <-  observer_b's events        BLOCKED by the framework
    observer_b  <-  observer_a's events        BLOCKED by the framework
    comparer    <-  either observer's events   BLOCKED (specs, never traces)
    observer_a  <-  the user's own turn        VISIBLE (see LIMIT below)

Two consequences, one good and one to be honest about:

  GOOD  the comparer cannot read the observers' reasoning, only the specs they
        published. It judges finished viewpoints, not the arguments behind them.

  LIMIT session state is NOT branch-scoped, and the user's turn IS visible to
        both observers. So when you type two goals, both observers can see both
        goals in that turn -- what they cannot see is each other's COLUMNS or
        RESULTS. Result-level isolation is enforced by the framework;
        goal-level isolation is enforced by construction, because each
        observer's instruction names only its own goal and neither instruction
        references the other's state key.

        Write it that way in the thesis. Do not claim more.

include_contents='none' on the observers is still worth setting: it strips
PREVIOUS turns, so a second question in the same `adk run` session cannot leak
the first run's specs into a fresh observer. It does not hide the current turn.
"""

import json
import pathlib
import re
import sys

# Same reason as agent_adk_single: ADK imports this as part of a package, so the
# project root is not guaranteed to be on sys.path. Both orchestrations must
# reach the SAME tools/ and data/ modules, not copies.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google.adk.agents.llm_agent import Agent  # noqa: E402
from google.adk.agents.parallel_agent import ParallelAgent  # noqa: E402
from google.adk.agents.sequential_agent import SequentialAgent  # noqa: E402
from google.adk.models.google_llm import Gemini  # noqa: E402
from google.genai import types  # noqa: E402

from data.active import ENTITY, NAME  # noqa: E402
from tools.compare_viewpoints import compare_viewpoints  # noqa: E402
from tools.describe_column import describe_column  # noqa: E402
from tools.list_columns import list_columns  # noqa: E402
from tools.run_lof_per_viewpoint import run_lof_per_viewpoint  # noqa: E402
from utils.adk_run_saver import (  # noqa: E402
    build_trace, events_of, parse_specs, save_adk_multi_run, text_of)

#: Pinned, exactly as in agent_adk_single. An alias can silently resolve to a
#: different model between runs, which would wreck a variance experiment -- and
#: here it would also confound the one-mind/two-mind comparison.
MODEL_NAME = "gemini-3.5-flash-lite"

#: RETRY IS NOT OPTIONAL IN A PARALLEL CELL, and ADK gives you none by default.
#:
#: google-genai builds `stop_after_attempt(1), reraise=True` when retry_options
#: is None -- literally no retry. ADK then re-raises a 429 as
#: _ResourceExhaustedError (models/google_llm.py). ParallelAgent merges its
#: sub-agents with asyncio.TaskGroup, which CANCELS the siblings and propagates,
#: so one 429 in either observer would:
#:      kill the other observer, mid-derivation
#:      skip the comparer entirely
#:      skip after_agent_callback -- so NO RUN FILE IS WRITTEN AT ALL
#: That last one breaks INV-5: a failed run is supposed to be evidence, and here
#: it would leave nothing behind. NFR-4 already promises rate limits are absorbed
#: by retry; this is where the ADK path keeps that promise.
#:
#: The exposure is worse here than in the single-agent cell for a structural
#: reason: two observers fire CONCURRENTLY, doubling the instantaneous request
#: rate against a free-tier per-minute ceiling.
#:
#: 429 is already in google-genai's retriable set (408, 429, 500, 502, 503, 504).
#: initial_delay is 4s rather than the library's 1s because a free-tier
#: per-minute window needs waiting out, not nibbling at; with exponential jitter
#: and 5 attempts that spans about a minute.
#:
#: DUPLICATED in agent_adk_single/agent.py on purpose -- no agent folder imports
#: another, the same rule that duplicates the LOF recipe across the two tools.
#: Change it in one, change it in both, or the two cells stop being comparable:
#: a cell that retries survives where a cell that does not dies, and which runs
#: reach the analysis would then depend on the machinery.
MODEL = Gemini(
    model=MODEL_NAME,
    retry_options=types.HttpRetryOptions(attempts=5, initial_delay=4, max_delay=60),
)

#: PER-AGENT, not shared. One shared pool would let observer_a spend the budget
#: and starve observer_b, and a starved observer is not an independent one.
#: Lower than the single agent's 20 because each observer handles ONE observer
#: point instead of two.
OBSERVER_BUDGET = 8

#: State keys. The observers write these; the comparer reads them. They are the
#: only channel in the system, and it runs one way.
GOAL_KEYS = ("goal_a", "goal_b")
SPEC_KEYS = ("spec_a", "spec_b")
OBSERVER_NAMES = ("observer_a", "observer_b")
COMPARER_NAME = "comparer"


# --------------------------------------------------------------------------- #
# Phase 1 leaves the prompt: the goals are given, not authored                #
# --------------------------------------------------------------------------- #

#: "Goal 1: ...", "Goal 2. ...", "goal 1 - ..." -- each goal runs until the next
#: "Goal <n>" marker or the end of the message. DOTALL so a goal may wrap lines.
_GOAL_RE = re.compile(
    r"goal\s*\d+\s*[:.\-]\s*(.+?)(?=\s*goal\s*\d+\s*[:.\-]|$)",
    re.IGNORECASE | re.DOTALL,
)

#: Below this, an "observer point" is too thin to derive anything from and the
#: run would waste quota. Guessed generously -- it only catches typos like
#: "Goal 1: x", not terse-but-real intents.
MIN_GOAL_CHARS = 15

_HOW_TO_ASK = (
    "Give one or two observer points, each a one-sentence INTENT with no column "
    "names and no thresholds:\n\n"
    "    Goal 1: find records that cannot describe a real place\n"
    "    Goal 2: find neighbourhoods that do not fit their region\n\n"
    "One goal runs cell 2.5 -- both observers get that same goal, which measures "
    "how much agreement to expect at all. Two goals runs cell 3."
)


def _split_goals(text: str) -> list[str]:
    """Pull the numbered observer points out of the user's message.

    Returns [] when the message carries no "Goal <n>:" marker at all. That is
    deliberately not treated as "one goal is the whole message": a broad
    question like "which entities are anomalous?" names no aspect, so an observer
    given it would have to invent one -- which is phase 1 creeping back in
    through the side door, in two contexts at once.
    """
    return [g.strip() for g in _GOAL_RE.findall(text) if g.strip()]


def seed_goals(callback_context):
    """before_agent_callback on the root: put each goal in its own state key.

    This is the dispatcher, and it is deliberately CODE rather than an LLM. A
    planner agent that read the message and decided who gets what would be
    Google's Coordinator pattern -- adaptive routing, chosen per run, which
    makes two runs non-comparable. This always does the same thing.

    Returning types.Content SKIPS the whole pipeline, which is how a bad
    message costs nothing instead of burning two observers' quota on nonsense.
    Returning None lets the run proceed.
    """
    goals = _split_goals(text_of(callback_context.user_content))

    if not goals:
        return types.Content(role="model", parts=[types.Part(
            text="No observer point found in that message.\n\n" + _HOW_TO_ASK)])
    short = [g for g in goals if len(g) < MIN_GOAL_CHARS]
    if short:
        return types.Content(role="model", parts=[types.Part(
            text=f"This observer point is too short to derive a viewpoint from: "
                 f"{short[0]!r}.\n\n" + _HOW_TO_ASK)])
    if len(goals) > 2:
        return types.Content(role="model", parts=[types.Part(
            text=f"{len(goals)} goals given; this cell runs exactly two "
                 f"observers.\n\n" + _HOW_TO_ASK)])

    # One goal -> BOTH observers get it. That is cell 2.5, and the duplication is
    # the point: identical intent, independent minds.
    callback_context.state[GOAL_KEYS[0]] = goals[0]
    callback_context.state[GOAL_KEYS[1]] = goals[1] if len(goals) > 1 else goals[0]
    callback_context.state["cell"] = "3" if len(goals) > 1 else "2.5"
    return None


# --------------------------------------------------------------------------- #
# The observer -- agent_adk_single's phase 2, for ONE observer point          #
# --------------------------------------------------------------------------- #

# GOAL_SLOT is replaced with a literal "{goal_a}" / "{goal_b}", which ADK then
# substitutes from session state. Everything after the head stays a PLAIN
# string: it contains literal JSON braces, and an f-string would mangle them.
# ADK leaves those alone -- its templater only substitutes braces whose contents
# are a valid Python identifier (utils/instructions_utils.py), and '"observer":
# "x"' is not one.
_OBSERVER_HEAD = f"""\
You are an observer agent working on the {NAME} dataset. Every row is one
{ENTITY} -- the instructions below say ENTITY, whatever the dataset.

YOUR OBSERVER POINT, and the only thing you answer to:

    GOAL_SLOT

""" + """\
Nothing else is your concern. You do not know what else is being looked at, and
you must not speculate: judge this observer point on its own terms, and derive
the viewpoint IT calls for.

A VIEWPOINT is how you choose to look at the data:
  columns    : which columns to observe (at least 2)
  row_filter : which rows to compare against (optional -- omit for all rows)

Columns decide WHAT is measured. The row filter decides WHO an entity is
compared against, and the same entity can be ordinary against one population
and extreme against another.

--- DERIVE THE VIEWPOINT ------------------------------------------------------

Explore with the tools, then commit. There is no fixed sequence and no fixed
number of steps: an observer point that plainly names a family of columns may
need two tool calls, one that could be read several ways may need six.

After each run_lof_per_viewpoint ask: ARE THE ROWS IT SURFACED THE KIND OF THING
MY OBSERVER POINT ASKED FOR? That judgement, not a step count, decides when you
are done. If it asked for impossible records and the rows it surfaced are merely
ordinary, or it asked about one aspect and the rows differ only in another, the
viewpoint is wrong however reasonable the columns looked. Try another.

KEEP WORKING while any of these is true:
  - your observer point could be read in more than one way and you have tested
    only one reading (the same phrase can point at a single unusual aspect, or
    at unusual CHARACTERISTICS overall -- different columns entirely)
  - the rows run_lof_per_viewpoint surfaced are not the kind of thing it describes
  - you cannot point to specific evidence from a tool result that justifies
    your columns

STOP as soon as none is true. Three well-evidenced calls beat spending the
budget to look thorough.

--- TOOLS ---------------------------------------------------------------------

  list_columns     what columns exist and what each one means
  describe_column  one column's scale, spread and extremes -- use it when you
                   need to know whether a column is skewed, capped or dominated
                   by a few rows before you trust it in a viewpoint
  run_lof_per_viewpoint
                   runs one candidate viewpoint and shows the five rows it
                   actually surfaces

Budget: at most 8 tool calls. A ceiling, not a target.

--- OUTPUT --------------------------------------------------------------------

When your viewpoint is final, reply with ONLY this JSON object and no other text:

{"observer": "<short name for yourself, from your observer point, e.g.
               data-quality-auditor>",
 "goal": "<your observer point, copied as one sentence>",
 "columns": ["<col>", "<col>"],
 "row_filter": null,
 "why": "<2-3 sentences: why these columns serve THIS observer point, citing
          the specific rows or numbers a tool actually returned>"}

Rules you must not break:
  - Never invent an anomaly score, ranking or threshold yourself.
    run_lof_per_viewpoint is the only thing here that measures anything.
  - Do not claim evidence you did not receive. If you say your viewpoint
    surfaced something, it must be in a tool result you actually got back.
  - Do NOT compare, rank or report entities. You derive ONE viewpoint and stop.
    Something else does the comparing, and it needs your spec, not your verdict.
  - Never end a reply with plain text unless it is the final JSON. Every other
    reply must contain a tool call -- a text-only reply ends your turn.
"""


def make_capture_spec(agent_name: str, spec_key: str):
    """after_agent_callback: write this observer's spec to state ourselves.

    WHY THIS EXISTS -- ADK's output_key is not reliable enough to build on.
    output_key only fires when `event.is_final_response()` is true
    (llm_agent.py). That is the SAME signal that silently dropped every spec
    from run 20260812_075010, which is why build_trace already carries a
    last_text fallback. Leaving the comparer's only input to output_key means a
    spec can exist in the event stream, be recorded in the run file by that
    fallback, and STILL be invisible to the comparer -- the run file would then
    show two viewpoints that were never actually compared.

    So parse it here, from the same events the run file is built from, and write
    it back canonically. Belt and braces over output_key, and it makes state and
    the run file agree by construction.

    Sets the key to "" when no spec could be parsed, so {spec_a?} renders empty
    and the comparer sees an unambiguous absence rather than half-formed text.
    """
    def capture_spec(callback_context):
        own = events_of(
            [e for e in callback_context.session.events
             if e.invocation_id == callback_context.invocation_id],
            agent_name,
        )
        _, final_text = build_trace(own)
        found = parse_specs(str(callback_context.state.get(spec_key) or "") or final_text)
        callback_context.state[spec_key] = json.dumps(found[0]) if found else ""
        return None

    return capture_spec


def make_observer(name: str, goal_key: str, spec_key: str) -> Agent:
    """One observer agent, bound to one goal state key and one spec state key.

    A factory rather than two hand-written agents so the two observers cannot
    drift apart: any asymmetry between them would confound the experiment, and
    the only difference here is WHICH KEY each reads and writes.
    """
    return Agent(
        model=MODEL,
        name=name,
        description=(
            "Derives one viewpoint -- columns plus an optional row filter -- "
            "from a single observer point, using tools, and reports nothing else."
        ),
        instruction=_OBSERVER_HEAD.replace("GOAL_SLOT", "{" + goal_key + "}"),
        tools=[list_columns, describe_column, run_lof_per_viewpoint],
        # Strips PREVIOUS turns, so a second question in one `adk run` session
        # cannot leak an earlier run's specs in here. It does not hide the
        # current turn -- see the module docstring.
        include_contents="none",
        # The final JSON lands in state under this key. It is the only thing
        # this agent contributes to the run, and the comparer's only input.
        output_key=spec_key,
        # ...and this re-derives the same key from the event stream, because
        # output_key alone has already been observed to miss. See the factory.
        after_agent_callback=make_capture_spec(name, spec_key),
    )


# --------------------------------------------------------------------------- #
# The comparer -- agent_adk_single's phase 3, reading two finished specs      #
# --------------------------------------------------------------------------- #

def pin_viewpoints(tool, args, tool_context):
    """before_tool_callback: the comparer compares the specs that EXIST. Full stop.

    WHY THIS EXISTS -- a prompt rule was not enough, and the failure was silent.

    Run 20260818_083909: observer_a made five tool calls, scored
    [AveRooms, AveBedrms, AveOccup], and then emitted nothing -- no spec. Its
    state key was empty, so {spec_a?} rendered blank. The comparer's instruction
    says "Never invent the missing viewpoint". It invented one anyway: it passed
    a viewpoint labelled "observer_a" (the label from its own instruction
    heading, since there was no self-chosen name to copy) with exactly the three
    columns observer_a had been scoring, then reported 456 shared entities and
    ten findings as an agreement between two observers. One of those observers
    had never committed a viewpoint.

    Whether the model reconstructed those columns from its own priors -- this
    dataset is in every LLM's training corpus, PROJECT_SPEC section 10 -- or
    guessed the canonical trio for that goal does not matter. The finding was
    fabricated either way, and nothing in the run file said so.

    So the argument is no longer the model's to choose. Whatever viewpoints it
    proposes are DISCARDED and replaced with the specs actually in state.
    Mutating args in place and returning None makes ADK call the tool with the
    replacement (flows/llm_flows/functions.py); returning a dict skips the tool
    and hands that dict back as the result.

    This is the same principle as INV-3 one level up: the LLM does not get to
    decide what gets measured, only what the measurement means.
    """
    if tool.name != "compare_viewpoints":
        return None

    pinned, labels = [], []
    for letter, key in zip("AB", SPEC_KEYS):
        specs = parse_specs(str(tool_context.state.get(key) or ""))
        if not specs:
            continue
        spec = specs[0]
        pinned.append({"observer": str(spec.get("observer") or f"observer_{letter.lower()}"),
                       "columns": spec.get("columns"),
                       "row_filter": spec.get("row_filter")})
        labels.append(letter)

    if not pinned:
        return {"result": "ERROR: neither observer published a viewpoint, so there "
                          "is nothing to compare. Report this run as having produced "
                          "no viewpoints. Do not describe any entity."}

    # Two observers on ONE goal (cell 2.5) routinely pick the same name for
    # themselves -- run 20260818_081042 had both call themselves
    # "data-quality-auditor", which would head two table columns identically and
    # leave the reader unable to tell which observer said what. Slot-prefix only
    # when they actually collide, since the self-chosen name is informative.
    if len({p["observer"][:16] for p in pinned}) < len(pinned):
        for letter, p in zip(labels, pinned):
            p["observer"] = f"{letter}:{p['observer']}"

    args["viewpoints"] = pinned
    return None


# The '?' in {spec_a?} is ADK's OPTIONAL form. Without it, an observer that failed
# leaves the key unset and templating raises KeyError: Context variable not
# found -- so one dead observer would take the whole run down instead of
# degrading to a one-viewpoint report.
_COMPARER_INSTRUCTION = f"""\
You are the comparer. Two observer agents each derived ONE viewpoint on the
{NAME} dataset, independently -- neither could see the other's work, and neither
can see yours. Every row is one {ENTITY}.

""" + """\
Observer A's viewpoint:
{spec_a?}

Observer B's viewpoint:
{spec_b?}

Each is a JSON object with "observer", "goal", "columns", "row_filter" and "why".

--- WHAT YOU DO ---------------------------------------------------------------

  1. Call compare_viewpoints ONCE. Pass the viewpoints above as you see them.

     The arguments you send are PINNED BY CODE before the tool runs: whatever
     you pass is replaced with the specs the observers actually published. You
     therefore cannot add a viewpoint, drop one, or edit one, and there is no
     point trying. Read the tool's reply to learn what was really compared --
     the observer names in the table are authoritative, not the ones you sent.

     If a viewpoint above is BLANK, that observer published nothing. Expect the
     tool to report ONE viewpoint, and say plainly in your summary that this is
     a one-viewpoint run because the other observer produced no spec. Never
     describe it as agreement between two observers.

  2. Report findings FROM ITS TABLE ONLY. Every entity you mention must appear
     there, with its numbers taken from there. Never promote an entity the tool
     did not surface, and never build a ranking of your own -- the table IS the
     result.

     The table holds the OVERLAP: the entities BOTH viewpoints flagged. For
     each, quote what EACH observer said about it, not just the stronger one --
     an entity at the 99.9th percentile in one and the 95th in the other is a
     weaker finding than one high in both, and the reader can only see that if
     you print both.

     Entities flagged by exactly ONE viewpoint are counted in the summary but
     are NOT in the table. Report that count and say plainly that those are
     anomalous from a single perspective only. Do not name any of them -- you
     have not been shown which they are, and naming one would be inventing
     evidence.

  3. Read the chance line under the counts, and say what the overlap means
     against it. Agreement close to what coincidence predicts is NOT a finding,
     and reporting it as one would overstate the data.

     If the tool warns that the two viewpoints are IDENTICAL, say so first and
     plainly: their agreement is arithmetic, not evidence, and no complementary
     finding can be drawn from it. That outcome is itself worth reporting -- two
     independent observers landed on the same viewpoint.

  4. Say whether the two observer points actually looked at different things.
     Compare their "goal" and "columns" fields. Shared columns are not a fault
     -- two intents may honestly need the same measurement -- but the reader
     needs to know whether this was one perspective twice or two perspectives.

  5. End with what this analysis CANNOT see: aspects neither observer point
     covered, and anomalies visible only in the COMBINATION of viewpoints -- an
     entity that is normal in BOTH viewpoints will never be flagged by either.

--- TOOL ----------------------------------------------------------------------

  compare_viewpoints   takes the finished viewpoints, executes them, flags each
                       one's top 10%, and shows the entities MORE THAN ONE
                       viewpoint flagged, with every viewpoint's percentile for
                       those entities, plus what chance alone would produce.

That is your only tool, on purpose. You do not get run_lof_per_viewpoint: an
agent that can re-score data can go looking for columns that make a nicer
story, and the viewpoints you were given are not open to revision.

--- OUTPUT --------------------------------------------------------------------

When you have the table, reply with ONLY this JSON object and no other text:

{"findings": [
  {"entity": <id from the comparison table>,
   "flagged_by": ["<observer name>", ...],
   "explanation": "<1-2 sentences quoting the table's percentiles>"}
],
 "summary": "<what was found overall: the overlap and what it means against the
              chance line, the single-perspective count, whether the two
              observer points really differed, and what this analysis cannot
              see>"}

Rules you must not break:
  - Never invent a score, ranking or threshold. compare_viewpoints is the only
    thing here that measures anything.
  - Never revise, re-derive or second-guess a viewpoint. You compare what you
    were given; you do not send anything back.
  - Do not claim evidence you did not receive.
  - Never end a reply with plain text unless it is the final JSON.
"""


# --------------------------------------------------------------------------- #
# The wiring                                                                  #
# --------------------------------------------------------------------------- #

observer_a = make_observer(OBSERVER_NAMES[0], GOAL_KEYS[0], SPEC_KEYS[0])
observer_b = make_observer(OBSERVER_NAMES[1], GOAL_KEYS[1], SPEC_KEYS[1])

# ParallelAgent carries a deprecation notice in ADK 2.6.3 pointing at the new
# graph-based Workflow. It still works, Workflow uses the SAME _BranchPath
# isolation, and Workflow is a different class hierarchy (BaseNode, not
# BaseAgent) with a heavier API. So: pin the ADK version (requirements.txt) and
# migrate AFTER the cells are measured, not during -- the same reasoning that
# pinned the model.
observers = ParallelAgent(
    name="observers",
    description=("Two observers derive one viewpoint each, concurrently "
                 "and in isolation from one another."),
    sub_agents=[observer_a, observer_b],
)

comparer = Agent(
    model=MODEL,
    name=COMPARER_NAME,
    description=(
        "Compares two finished viewpoints' verdicts and reports the overlap. "
        "Cannot revise either viewpoint and cannot score anything itself."
    ),
    instruction=_COMPARER_INSTRUCTION,
    tools=[compare_viewpoints],
    # Replaces whatever viewpoints the model proposes with the specs actually in
    # state. Without it the comparer can fabricate a missing observer's
    # viewpoint, which it did on 18 Aug 2026 -- see pin_viewpoints.
    before_tool_callback=pin_viewpoints,
)

root_agent = SequentialAgent(
    name="observers_two",
    description=(
        "Two observer agents derive one viewpoint each from their own observer "
        "point, in isolation; a third compares what those viewpoints conclude."
    ),
    sub_agents=[observers, comparer],
    # Parses the numbered goals into state before anything runs, and rejects a
    # malformed message without spending a single LLM call.
    before_agent_callback=seed_goals,
    # Fires once at the end: rebuilds a per-observer trace from ADK's event
    # stream and writes runs/*.json. Without it a run leaves no artifact.
    after_agent_callback=save_adk_multi_run,
)
