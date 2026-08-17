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

from data.active import NAME as DATASET_NAME
from utils.save_run import save_run

#: Which model backend these agents talk to. ADK is configured for Gemini in
#: agent.py; recorded so run files stay comparable with the custom loop's
#: "gemini" runs.
BACKEND_NAME = "gemini"


def text_of(event_or_content) -> str:
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


def _parse_payload(text: str) -> dict:
    """The agent's whole closing JSON object, or {} when there is none.

    parse_specs pulls the viewpoint specs out of this; the findings-phase
    fields ("findings", "summary") ride along here so the run file can keep
    them too. Tolerates prose and ```json fences the same way parse_specs does.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_specs(text: str) -> list:
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
    last_text = ""      # fallback -- see the note below step 3

    for event in events:
        # 1. The model asked for tools. One event can carry several calls.
        calls = event.get_function_calls() or []
        for call in calls:
            entry = {
                "step": len(trace) + 1,
                "thinking": text_of(event),   # any prose the model emitted alongside
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
        # is_final_response() is the intended signal, but a live run on 12 Aug
        # 2026 produced a closing text event it did NOT mark final, and the specs
        # silently vanished from the run file. So ALSO remember the last
        # text-bearing, call-free event as a fallback: if no event is marked
        # final, the closing message is still whatever text came last.
        if not calls:
            text = text_of(event)
            if text:
                last_text = text
                if event.is_final_response():
                    final_text = text

    return trace, final_text or last_text


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
    payload = _parse_payload(final_text)
    specs = parse_specs(final_text)

    if specs:
        trace.append({"step": len(trace) + 1, "thinking": "", "final_specs": specs})

    message = text_of(callback_context.user_content) or "(nothing recorded)"

    # Findings-phase output: the deliverable of phase 3, kept in the run file.
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else None
    summary = payload.get("summary") if isinstance(payload.get("summary"), str) else None

    path = save_run(
        user_prompt=message,   # the question the user asked, verbatim
        backend_name=BACKEND_NAME,
        specs=specs,           # always a list, one entry per observer point
        trace=trace,
        orchestrator="adk",
        findings=findings,
        summary=summary,
        dataset=DATASET_NAME,  # which dataset data/active.py served this run
    )
    print(f"\n[run saved] {path}  ({len(trace)} steps, {len(specs)} observer "
          f"point(s), {len(findings) if findings else 0} finding(s))")
    return None


# --------------------------------------------------------------------------- #
# Two-observer runs (agent_adk_multiple, cells 2.5 and 3)                     #
# --------------------------------------------------------------------------- #

#: Which agent produced which state key, in report order. Kept here rather than
#: imported from agent_adk_multiple: utils/ must not import an agent folder, or
#: the other orchestration could not use this writer.
_OBSERVER_SLOTS = (("observer_a", "goal_a", "spec_a"),
                   ("observer_b", "goal_b", "spec_b"))
_COMPARER_NAME = "comparer"


def stop_reason(events):
    """Why an agent stopped, when it stopped without answering. '' if it looks fine.

    Added 18 Aug 2026 after run 20260818_083909, where observer_a made five tool
    calls, received a good result, and then emitted NOTHING -- no spec, no text.
    The run file recorded "exhausted" and gave no way to tell a quota failure
    from a model that simply gave up, so the run could not be diagnosed at all.

    Event inherits error_code, error_message and finish_reason from LlmResponse,
    so the answer was in the stream the whole time and we were discarding it.
    """
    for event in reversed(events):
        code = getattr(event, "error_code", None)
        message = getattr(event, "error_message", None)
        if code or message:
            return f"{code or 'error'}: {message or ''}".strip()
    for event in reversed(events):
        reason = getattr(event, "finish_reason", None)
        # STOP is the normal ending; anything else is why the model quit early
        # (MAX_TOKENS, SAFETY, RECITATION, ...).
        if reason is not None and str(reason).upper().rsplit(".", 1)[-1] not in ("STOP", "NONE"):
            return f"finish_reason: {reason}"
    return ""


def events_of(events, agent_name):
    """The events one sub-agent produced.

    Two tests, ORed, because the two event kinds are attributed differently: a
    ParallelAgent sub-agent's events carry its branch path (".../observer_a"),
    while `author` is the reliable marker for the agent's own messages. Using
    both means a tool response cannot fall through the gap and vanish from that
    observer's trace.
    """
    return [e for e in events
            if (getattr(e, "branch", None) or "").endswith(agent_name)
            or e.author == agent_name]


def save_adk_multi_run(callback_context):
    """ADK after_agent_callback for the TWO-observer cells. One run, one file.

    Wire it up on the root SequentialAgent:
        SequentialAgent(..., after_agent_callback=save_adk_multi_run)

    Why a second function rather than a flag on save_adk_run: a two-observer run
    has N traces, not one, and each observer's goal and spec must stay attached
    to the agent that produced them. Folding that into the single-agent writer
    would put a branch in every line of it.

    Specs are read from STATE first, because state is what the comparer actually
    saw, then from the observer's own closing text as a fallback -- ADK's
    output_key only fires on is_final_response(), the same signal that silently
    lost every spec on 12 Aug 2026.

    Returns None always, so the agent's own response is left untouched.
    """
    events = [
        e for e in callback_context.session.events
        if e.invocation_id == callback_context.invocation_id
    ]
    state = callback_context.state

    # -- per-observer traces and specs -----------------------------------------
    observers, specs, claimed = [], [], set()
    for agent_name, goal_key, spec_key in _OBSERVER_SLOTS:
        own = events_of(events, agent_name)
        claimed.update(id(e) for e in own)
        trace, final_text = build_trace(own)

        # state holds the raw closing text; fall back to the trace's own final
        # text if output_key never fired (an observer that died mid-run).
        spec_list = parse_specs(str(state.get(spec_key) or "") or final_text)
        spec = spec_list[0] if spec_list else None
        if spec:
            specs.append(spec)

        record = {
            "observer": agent_name,
            "goal": state.get(goal_key),
            # status is per observer on purpose: one dead observer must be
            # visible as such, not silently reduce the run to one viewpoint.
            "status": "completed" if spec else "exhausted",
            "spec": spec,
            "steps_taken": len(trace),
            "trace": trace,
        }
        # Only when it failed. Without these two a dead observer is
        # indistinguishable from a lazy one, which is exactly why run
        # 20260818_083909 could not be diagnosed.
        if not spec:
            why = stop_reason(own)
            if why:
                record["stop_reason"] = why
            # ALWAYS written when there is no spec, even as "" -- the empty
            # string is itself the finding. Three cases, three appearances:
            #   ""              the observer emitted nothing at all
            #   reasoning prose the model thought and never answered (ADK drops
            #                   thought-only responses on the floor:
            #                   llm_agent.py checks `not part.thought`)
            #   broken JSON     it tried to answer and the parse failed
            # Untruncated, like the trace (INV-5): a truncated diagnostic is the
            # one that cuts off just before the interesting part.
            record["raw_final_text"] = final_text
        observers.append(record)

    # -- the comparer: everything not attributable to an observer ----------------
    # Defined by subtraction so no event can be lost. The comparer's branch is
    # the root's, which no endswith test would match.
    comparer_events = [e for e in events if id(e) not in claimed]
    comparer_trace, comparer_text = build_trace(comparer_events)
    payload = _parse_payload(comparer_text)

    comparer_record = {
        "observer": _COMPARER_NAME,
        "goal": None,                       # the comparer has no observer point
        "status": "completed" if payload else "exhausted",
        "spec": None,                       # it derives nothing
        "steps_taken": len(comparer_trace),
        "trace": comparer_trace,
    }
    if not payload:
        why = stop_reason(comparer_events)
        if why:
            comparer_record["stop_reason"] = why
        # Same reasoning as the observers above. A comparer that ran the tool and
        # then failed to report is the worst case to debug blind: the numbers
        # exist in its trace but nothing says why they never became findings.
        comparer_record["raw_final_text"] = comparer_text
    observers.append(comparer_record)

    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else None
    summary = payload.get("summary") if isinstance(payload.get("summary"), str) else None

    # `trace` at the top level is the whole invocation in order, so existing
    # readers (and INV-8) still get the field they expect; `observers` is where
    # the per-agent split lives.
    whole_trace, _ = build_trace(events)
    if specs:
        whole_trace.append({"step": len(whole_trace) + 1, "thinking": "",
                            "final_specs": specs})

    path = save_run(
        user_prompt=text_of(callback_context.user_content) or "(nothing recorded)",
        backend_name=BACKEND_NAME,
        specs=specs,
        trace=whole_trace,
        orchestrator="adk_two",     # NOT "adk": one-mind and two-mind runs must
        findings=findings,          # be distinguishable, that IS the experiment
        summary=summary,
        dataset=DATASET_NAME,
        cell=state.get("cell"),     # "2.5" (same goal) or "3" (two goals)
        observers=observers,
        # save_run would call this "completed" on the strength of ONE spec.
        # In a two-observer cell that is a lie: run 20260818_083909 said
        # completed with observer_a dead, and the comparer went on to report a
        # two-observer agreement that never happened.
        status_override=("completed" if all(
            o["status"] == "completed" for o in observers)
            else "partial" if specs else "exhausted"),
    )
    done = sum(1 for o in observers if o["status"] == "completed")
    print(f"\n[run saved] {path}  (cell {state.get('cell')}, {done}/{len(observers)} "
          f"agents completed, {len(specs)} spec(s), "
          f"{len(findings) if findings else 0} finding(s))")
    return None
