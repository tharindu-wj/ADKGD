"""Why does an agent's turn end? Records every LLM response, not just the tools.

    python scripts/check_agent_turns.py
    python scripts/check_agent_turns.py --runs 3

The trace in a run file shows tool CALLS. It cannot show a turn that ended
without one, which is exactly the failure worth understanding: agents were
observed running a scorer and then stopping -- no closing message, no
submit_spec, nothing recorded.

This attaches an after_model_callback to each LLM agent and prints, for every
single model response: what came back (tool calls / text / neither), the
finish_reason the API gave, any error, and the token counts. The production
agents are left alone -- the instrumentation is bolted on here at runtime, so
nothing about the tree changes when this script is not running.

Reading it: FinishReason.STOP with no parts means the model chose to end its
turn with nothing to say, which is a prompting problem. MAX_TOKENS means the
response was cut off, which is not. An error_code means the call never
succeeded and the "agent stopped" reading was wrong all along.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import collections
import os

if sys.platform == "win32":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=1, help="how many times to run the tree")
ap.add_argument("--pause", type=int, default=40,
                help="seconds between runs; back-to-back runs get rate limited")
args = ap.parse_args()


def find_key():
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    for path in (ROOT / ".env", ROOT.parent / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_KEY"):
                return v.strip().strip('"').strip("'")
    return None


key = find_key()
if not key:
    raise SystemExit("No Gemini key. See the Setup section of README.md.")
os.environ["GOOGLE_API_KEY"] = key

import asyncio  # noqa: E402
import time  # noqa: E402

from google.adk.agents.llm_agent import Agent  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from agents.agent import root_agent  # noqa: E402
from loaders.active import DATASET  # noqa: E402

RECORD = []


def make_probe(name):
    """after_model_callback: record what came back, then get out of the way."""
    def probe(callback_context, llm_response):
        parts = getattr(getattr(llm_response, "content", None), "parts", None) or []
        calls = [p.function_call.name for p in parts
                 if getattr(p, "function_call", None)]
        text = "".join(p.text for p in parts if getattr(p, "text", None))
        u = getattr(llm_response, "usage_metadata", None)
        RECORD.append({
            "agent": name,
            "calls": calls,
            "text_len": len(text.strip()),
            "finish": str(getattr(llm_response, "finish_reason", None) or "-"),
            "error": llm_response.error_code,
            "message": llm_response.error_message,
            "partial": bool(getattr(llm_response, "partial", False)),
            "in_tok": getattr(u, "prompt_token_count", None) if u else None,
            "out_tok": getattr(u, "candidates_token_count", None) if u else None,
        })
        return None                      # never alter the response
    return probe


def instrument(agent):
    """Bolt a probe onto every LLM agent in the tree, in place."""
    n = 0
    if isinstance(agent, Agent):
        agent.after_model_callback = make_probe(agent.name)
        n += 1
    for sub in getattr(agent, "sub_agents", []) or []:
        n += instrument(sub)
    return n


print(f"instrumented {instrument(root_agent)} LLM agents\n")

for attempt in range(1, args.runs + 1):
    RECORD.clear()
    app, user, session = "kg_audit", "local", f"probe{attempt}"
    svc = InMemorySessionService()
    asyncio.run(svc.create_session(app_name=app, user_id=user, session_id=session))
    runner = Runner(app_name=app, agent=root_agent, session_service=svc)

    for _ in runner.run(user_id=user, session_id=session,
                        new_message=types.Content(role="user", parts=[types.Part(
                            text=f"Audit the {DATASET.NAME} graph.")])):
        pass

    print(f"=== run {attempt}: {len(RECORD)} model responses " + "=" * 34)
    print(f"  {'#':>2} {'agent':<13}{'returned':<34}{'finish':<26}{'in':>7}{'out':>6}")
    per = collections.Counter()
    for i, r in enumerate(RECORD, 1):
        per[r["agent"]] += 1
        if r["calls"]:
            what = "call " + ",".join(r["calls"])
        elif r["text_len"]:
            what = f"text ({r['text_len']} chars)"
        else:
            what = "*** NOTHING ***"
        finish = r["finish"].replace("FinishReason.", "")
        if r["error"]:
            finish = f"{r['error']}: {(r['message'] or '')[:30]}"
        print(f"  {i:>2} {r['agent']:<13}{what[:33]:<34}{finish[:25]:<26}"
              f"{r['in_tok'] or 0:>7}{r['out_tok'] or 0:>6}")

    empty = [r for r in RECORD if not r["calls"] and not r["text_len"]]
    print(f"\n  responses per agent: {dict(per)}")
    print(f"  responses that returned NOTHING: {len(empty)}")
    for r in empty:
        print(f"    {r['agent']}: finish={r['finish']} error={r['error']} "
              f"in={r['in_tok']} out={r['out_tok']}")

    if attempt < args.runs:
        print(f"\n  pausing {args.pause}s\n")
        time.sleep(args.pause)
