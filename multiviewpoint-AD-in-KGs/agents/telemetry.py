"""Did every model call actually happen? Records what the trace cannot show.

WHY THIS EXISTS.

A run file's trace records tool CALLS. It cannot record a call that was never
made -- and those look identical from the outside:

    an agent that ran a scorer and then chose to stop
    an agent whose next request the API refused

On the free Gemini tier the quota is 15 requests per minute per model. This
tree needs 16 (root 4, each viewpoint 6), so the last request is routinely
refused with a 429 and the agent simply vanishes mid-turn. That was read as
"the agents often fail to answer" for several rounds of debugging -- two prompt
rewrites were spent on a behaviour that was never happening -- before anyone
instrumented it and found 31 model responses of which ZERO returned nothing.

So: every run now records how many model calls each agent actually completed
and every error that stopped one. A truncated run says so, loudly, instead of
being quietly reported as an agent that declined to answer.

Module-level state is deliberate and safe here: one run per process, and
`reset()` is called before each. `record_error` always returns None, which is
what makes this observation rather than handling -- the error propagates
exactly as it would without us.
"""
import collections

CALLS = collections.Counter()
ERRORS = []

#: Substrings that mean "the API refused", not "the agent decided".
_QUOTA = ("RESOURCE_EXHAUSTED", "429", "quota", "rate limit")


def reset():
    """Call before a run. Otherwise counts accumulate across runs."""
    CALLS.clear()
    ERRORS.clear()


def record_response(callback_context, llm_response):
    """after_model_callback: count a model call that came back."""
    CALLS[callback_context.agent_name] += 1
    return None                       # never alter the response


def record_error(callback_context, llm_request, error):
    """on_model_error_callback: record a model call that did not come back."""
    text = str(error)
    ERRORS.append({
        "agent": callback_context.agent_name,
        "type": type(error).__name__,
        "quota": any(q.lower() in text.lower() for q in _QUOTA),
        "message": " ".join(text.split())[:300],
    })
    return None                       # returning None re-raises, unchanged


def health():
    """What happened to this run's model calls, for the run file."""
    return {
        "model_calls": dict(CALLS),
        "total_model_calls": sum(CALLS.values()),
        "errors": list(ERRORS),
        "truncated": bool(ERRORS),
        "quota_exhausted": any(e["quota"] for e in ERRORS),
    }


def render(h):
    """One human-readable verdict on whether the run is worth believing."""
    calls = ", ".join(f"{a} {n}" for a, n in sorted(h["model_calls"].items()))
    lines = [f"  model calls: {h['total_model_calls']} ({calls})"]
    if not h["errors"]:
        lines.append("  every model call completed -- this run is complete.")
        return "\n".join(lines)

    if h["quota_exhausted"]:
        lines.append("  *** TRUNCATED BY THE API QUOTA, NOT BY THE AGENTS ***")
        lines.append("  An agent stopped because its request was refused. Any")
        lines.append("  missing goal, frame or spec below is a rate limit, and")
        lines.append("  says NOTHING about how the agents behave. Do not read")
        lines.append("  this run as evidence. Wait a minute and run it again.")
    else:
        lines.append("  *** TRUNCATED BY A MODEL ERROR ***")
    for e in h["errors"]:
        lines.append(f"    {e['agent']}: {e['type']} -- {e['message'][:120]}")
    return "\n".join(lines)
