"""Did every model call actually happen? Records what a trace cannot show.

A run's trace records tool CALLS. It cannot record a call that was never
made -- and from the outside these look identical:

    an agent that chose to stop
    an agent whose next request the API refused

The free Gemini tier allows 15 requests per minute per model, and reading the
second case as the first cost the predecessor project two prompt rewrites
chasing a behaviour that was never happening. So every run records how many
model calls each agent completed and every error that stopped one, and a
truncated run says so instead of posing as agent behaviour.

Module-level state is deliberate: one run per process, `reset()` first.
Both callbacks return None, which makes this observation, not handling.
"""
import collections
import logging
import time

CALLS = collections.Counter()
ERRORS = []
RETRIES = []
_started = [None]

#: substrings that mean "the API refused", not "the agent decided"
_QUOTA_MARKERS = ("RESOURCE_EXHAUSTED", "429", "quota", "rate limit")

#: the genai client logs one INFO line per backoff before it sleeps. A retry
#: that SUCCEEDS never reaches the error callback, so without watching this
#: log a run at the rate-limit ceiling would just look mysteriously slow.
_GENAI_LOGGER = "google_genai._api_client"


class _RetryWatcher(logging.Handler):
    def emit(self, record):
        message = record.getMessage()
        if "Retrying" in message:
            RETRIES.append(" ".join(message.split())[:200])


_watcher = _RetryWatcher(level=logging.INFO)


def reset():
    """Call before a run. Otherwise counts accumulate across runs."""
    CALLS.clear()
    ERRORS.clear()
    RETRIES.clear()
    _started[0] = time.monotonic()
    genai_logger = logging.getLogger(_GENAI_LOGGER)
    if _watcher not in genai_logger.handlers:
        genai_logger.addHandler(_watcher)
    genai_logger.setLevel(min(genai_logger.level or logging.INFO, logging.INFO))


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
        "quota": any(m.lower() in text.lower() for m in _QUOTA_MARKERS),
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
        "retries": len(RETRIES),
        "seconds": round(time.monotonic() - started, 1) if started else None,
    }


def render(h):
    """One human-readable verdict on whether the run is worth believing."""
    calls = ", ".join(f"{agent} {n}" for agent, n in sorted(h["model_calls"].items()))
    took = f" in {h['seconds']}s" if h.get("seconds") else ""
    lines = [f"  model calls: {h['total_model_calls']} ({calls}){took}"]

    if h.get("retries"):
        lines.append(f"  {h['retries']} request(s) were rate limited and "
                     f"RETRIED -- the run is intact, but it sat on the quota "
                     f"ceiling; the waiting is why it took this long.")

    if not h["errors"]:
        lines.append("  every model call completed -- this run is complete.")
        return "\n".join(lines)

    if h["quota_exhausted"]:
        lines.append("  *** TRUNCATED BY THE API QUOTA, NOT BY THE AGENT ***")
        lines.append("  Whatever is missing below is a rate limit, and says")
        lines.append("  NOTHING about agent behaviour. Wait a minute, rerun.")
    else:
        lines.append("  *** TRUNCATED BY A MODEL ERROR ***")
    for e in h["errors"]:
        lines.append(f"    {e['agent']}: {e['type']} -- {e['message'][:120]}")
    return "\n".join(lines)
