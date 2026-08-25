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
import logging
import time

CALLS = collections.Counter()
ERRORS = []
RETRIES = []
_started = [None]

#: Substrings that mean "the API refused", not "the agent decided".
_QUOTA = ("RESOURCE_EXHAUSTED", "429", "quota", "rate limit")

#: The client logs one INFO line per backoff before it sleeps. A retry that
#: SUCCEEDS never reaches on_model_error_callback -- ADK sits above the retry --
#: so without this a run would just quietly get slower and the fact that we are
#: sitting on the rate-limit ceiling would be invisible again.
_GENAI_LOGGER = "google_genai._api_client"


class _RetryWatcher(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "Retrying" in msg:
            RETRIES.append(" ".join(msg.split())[:200])


_watcher = _RetryWatcher(level=logging.INFO)


def reset():
    """Call before a run. Otherwise counts accumulate across runs."""
    CALLS.clear()
    ERRORS.clear()
    RETRIES.clear()
    _started[0] = time.monotonic()
    lg = logging.getLogger(_GENAI_LOGGER)
    if _watcher not in lg.handlers:
        lg.addHandler(_watcher)
    lg.setLevel(min(lg.level or logging.INFO, logging.INFO))


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
    started = _started[0]
    return {
        "model_calls": dict(CALLS),
        "total_model_calls": sum(CALLS.values()),
        "errors": list(ERRORS),
        "truncated": bool(ERRORS),
        "quota_exhausted": any(e["quota"] for e in ERRORS),
        # A run that had to wait is a run at the ceiling, even if it completed.
        "retries": len(RETRIES),
        "seconds": round(time.monotonic() - started, 1) if started else None,
    }


def render(h):
    """One human-readable verdict on whether the run is worth believing."""
    calls = ", ".join(f"{a} {n}" for a, n in sorted(h["model_calls"].items()))
    took = f" in {h['seconds']}s" if h.get("seconds") else ""
    lines = [f"  model calls: {h['total_model_calls']} ({calls}){took}"]

    if h.get("retries"):
        lines.append(f"  {h['retries']} request(s) were rate limited and RETRIED. "
                     f"The run is")
        lines.append("  intact -- the waiting is why it took this long -- but the")
        lines.append("  tree is sitting on the quota ceiling, so a run that is not")
        lines.append("  retried is a run that got lucky on timing.")

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
