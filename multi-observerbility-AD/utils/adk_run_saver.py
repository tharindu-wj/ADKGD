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


def _parse_spec(text: str):
    """Pull the final ViewSpec out of the agent's closing message.

    The instruction asks for bare JSON, but models sometimes wrap it in prose or
    a ```json fence -- so take everything from the first '{' to the last '}'.
    Returns None if there is no valid JSON, which save_run records as
    status="exhausted": a run that produced no spec is still evidence.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


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
    spec = _parse_spec(final_text)

    if spec is not None:
        trace.append({"step": len(trace) + 1, "thinking": "", "final_spec": spec})

    goal = _text_of(callback_context.user_content) or "(no goal recorded)"

    path = save_run(
        goal=goal,
        backend_name=BACKEND_NAME,
        spec=spec,
        trace=trace,
        orchestrator="adk",
    )
    print(f"\n[run saved] {path}  ({len(trace)} steps, "
          f"{'completed' if spec else 'no spec parsed'})")
    return None
