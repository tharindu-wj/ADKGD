"""Turn an ADK run into the same run file the custom orchestrator writes.

WHY THIS EXISTS
---------------
`adk run` and `adk web` do not know about utils/save_run.py, so an ADK run would
leave no artifact behind -- no spec, no trace, nothing to compare. This module
bridges that gap with ONE ADK hook:

    root_agent = Agent(..., after_agent_callback=save_adk_run)

after_agent_callback fires once, when the agent finishes. At that point ADK has
recorded everything as Events, and the callback can read them back.

HOW THE TRANSLATION WORKS
-------------------------
ADK and the custom loop record the same story in different shapes:

    custom loop        one trace entry per step, built as the loop runs
    ADK                a flat stream of Events, built by the framework

An ADK tool call spans TWO events: the model emits an event containing a
function_call, then the tool's answer arrives in a later event as a
function_response. We walk the stream, pair them up by call id, and emit the
same {"step", "thinking", "tool", "args", "result"} entries the custom
orchestrator produces -- so both orchestrations write ONE schema and stay
comparable.

Returning None from the callback leaves the agent's own output untouched: this
hook observes, it never changes what the agent said.
"""

import json
import re

from utils.save_run import save_run

#: Which model backend these agents talk to. ADK is configured for Gemini in
#: agent.py; recorded so run files stay comparable with the custom loop's
#: "gemini" runs.
BACKEND_NAME = "gemini"


def _text_of(event_or_content) -> str:
    """Return the plain text of an Event or a bare Content. '' when there is none.

    Both shapes turn up: session events wrap their text in `.content`, while
    callback_context.user_content IS a Content already. Handling both here keeps
    the two cases from drifting apart -- an earlier version only understood
    Events, which silently lost the goal.
    """
    content = getattr(event_or_content, "content", event_or_content)
    if not content or not getattr(content, "parts", None):
        return ""
    text = "".join(part.text for part in content.parts if getattr(part, "text", None))
    # Strip any byte-order mark: piping a goal in from PowerShell prefixes one,
    # and an invisible ﻿ would make two otherwise-identical goals compare
    # as different when analysing runs/.
    return text.replace("﻿", "").strip()


def _parse_specs(text: str) -> list:
    """Pull the ViewSpecs out of the agent's closing message. Always a list.

    The instruction asks for {"specs": [...]} -- one entry per goal -- but models
    sometimes wrap it in prose or a ```json fence, so take everything from the
    first '{' to the last '}'. A bare single spec (no "specs" wrapper) is also
    accepted and returned as a one-element list, because that is what the agent
    naturally produces for a single goal and there is no reason to reject it.

    Returns [] when there is no valid JSON, which save_run records as
    status="exhausted": a run that produced no spec is still evidence.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    if isinstance(payload, dict) and isinstance(payload.get("specs"), list):
        return [s for s in payload["specs"] if isinstance(s, dict)]
    if isinstance(payload, dict) and "columns" in payload:
        return [payload]          # a bare single spec
    return []


#: A numbered goal marker: "Goal 1:", "Goal 2.", "goal 3)". The DIGIT and the
#: delimiter are both required, which is what stops an ordinary mention like
#: "compare with goal 2 style analysis" from being read as a new goal.
#: Not anchored to line starts, deliberately -- see _parse_goals.
_GOAL_MARKER = re.compile(r"\bgoal\s*\d+\s*[:.)]\s*", re.IGNORECASE)

#: A bare "Goal:" prefix on a single unnumbered goal, stripped for tidiness.
_BARE_PREFIX = re.compile(r"^\s*goal\s*[:.)]\s*", re.IGNORECASE)


def _parse_goals(message: str) -> list:
    """Split the user's message into individual goals.

    Both layouts work, because the two ways of running an agent differ:

        Goal 1: find X  Goal 2: find Y        <- one line  (`adk run`)
        Goal 1: find X
        Goal 2: find Y                        <- several lines (`adk web`)

    `adk run` is a line-based REPL: it reads ONE line per turn, so a multi-line
    message piped into it silently loses everything after the first newline.
    That is why the marker is not anchored to line starts -- on the terminal the
    goals have to share a line.

    A message with no numbered markers is one unnumbered goal and comes back as
    a single item, so single-goal runs -- the cell-1 condition -- keep working
    with nothing to remember.

    The experimental condition is read off len(goals): 1 = cell 1 (one agent,
    one goal), 2+ = cell 2 (one agent, several goals held at once).
    """
    message = (message or "").strip()
    if not message:
        return []

    # Everything before the first marker is preamble, so drop parts[0].
    parts = _GOAL_MARKER.split(message)
    goals = [p.strip() for p in parts[1:] if p.strip()]
    if goals:
        return goals

    return [_BARE_PREFIX.sub("", message).strip()]


def build_trace(events):
    """Convert an ADK event stream into the custom loop's trace format.

    Args:
        events: the Events for ONE invocation, in order.

    Returns:
        (trace, final_text) -- the step list, and the agent's closing message.
    """
    trace = []
    pending = {}        # function_call id -> the step entry awaiting its result
    final_text = ""

    for event in events:
        # 1. The model asked for tools. One event can carry several calls.
        calls = event.get_function_calls() or []
        for call in calls:
            entry = {
                "step": len(trace) + 1,
                "thinking": _text_of(event),   # any prose the model emitted alongside
                "tool": call.name,
                "args": dict(call.args or {}),
                "result": None,                # filled in when the response arrives
            }
            trace.append(entry)
            pending[call.id] = entry

        # 2. Tool results come back in a later event; pair them by call id.
        for response in event.get_function_responses() or []:
            entry = pending.pop(response.id, None)
            if entry is not None:
                answer = response.response
                # ADK wraps non-dict tool returns as {"result": ...}; our tools
                # return strings, so unwrap to keep the trace human-readable.
                if isinstance(answer, dict) and set(answer) == {"result"}:
                    answer = answer["result"]
                entry["result"] = answer if isinstance(answer, str) else json.dumps(answer)

        # 3. The closing message -- the agent's final answer, holding the spec.
        if event.is_final_response() and not calls:
            text = _text_of(event)
            if text:
                final_text = text

    return trace, final_text


def save_adk_run(callback_context):
    """ADK after_agent_callback: write this run to runs/ then get out of the way.

    Wire it up with:
        Agent(..., after_agent_callback=save_adk_run)

    Returns None always, so the agent's own response is left untouched.
    """
    # Only this invocation's events. A session accumulates turns -- without the
    # filter, a second question in the same `adk run` session would re-save the
    # first one's steps as well.
    events = [
        e for e in callback_context.session.events
        if e.invocation_id == callback_context.invocation_id
    ]

    trace, final_text = build_trace(events)
    specs = _parse_specs(final_text)

    if specs:
        trace.append({"step": len(trace) + 1, "thinking": "", "final_specs": specs})

    message = _text_of(callback_context.user_content) or "(no goal recorded)"
    goals = _parse_goals(message)

    path = save_run(
        goal=message,          # the raw message, verbatim
        backend_name=BACKEND_NAME,
        spec=specs,            # save_run unpacks: 1 spec -> final_spec, N -> final_specs
        trace=trace,
        orchestrator="adk",
        goals=goals,
    )
    print(f"\n[run saved] {path}  ({len(trace)} steps, {len(goals)} goal(s), "
          f"{len(specs)} spec(s){'' if specs else ' -- none parsed'})")
    return None
