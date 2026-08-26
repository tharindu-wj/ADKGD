"""Run one observation: personas -> blind norms -> scopes -> verdicts.

    python scripts/3_run_observers.py
    python scripts/3_run_observers.py --quiet

The trigger message is pinned to "Begin the observation." on purpose: the dataset
CARD in the instructions must be the only channel through which any agent
learns the domain, and a free-form message here could leak schema into
phase 1.

Besides recording the run, this script PROVES the separation for it: from the
trace, for each observer, declare_semantics must come before that observer's
first data-tool call. A run that violates the ordering is stamped invalid.
"""
import json
import os
import sys
from pathlib import Path

# Scripts live in scripts/, so Python puts THAT on sys.path, not the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import datetime  # noqa: E402

from loaders.active import DATASET  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--quiet", action="store_true",
                    help="skip the live trace, print only the result")
args = parser.parse_args()

if not DATASET.KG.exists():
    raise SystemExit(f"missing {DATASET.KG}. Run scripts/1_prepare_graph.py first.")


def find_key():
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    for env_file in (ROOT / ".env", ROOT.parent / ".env",
                     ROOT.parent / "multiviewpoint-AD-in-KGs" / ".env"):
        if not env_file.exists():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_KEY"):
                return value.strip().strip('"').strip("'")
    return None


key = find_key()
if not key:
    raise SystemExit(f"No Gemini key. Put GOOGLE_API_KEY=... in a .env beside {ROOT}.")
os.environ["GOOGLE_API_KEY"] = key

import asyncio  # noqa: E402

from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from agents import telemetry  # noqa: E402
from agents.agent import root_agent  # noqa: E402
from agents.config import (SCANNER_KEYS, NORMS_KEYS, PERSONA_KEYS,  # noqa: E402
                           SCOPE_KEYS, SERVED_KEYS, VERDICT_KEYS)
from agents.phase_gate import DATA_TOOL_NAMES  # noqa: E402
from tools._observers import OBSERVER_NAMES  # noqa: E402

APP, USER, SESSION = "multi_observer", "local", "run"

session_service = InMemorySessionService()
asyncio.run(session_service.create_session(app_name=APP, user_id=USER,
                                           session_id=SESSION))
runner = Runner(app_name=APP, agent=root_agent, session_service=session_service)

print(f"dataset: {DATASET.NAME}   graph: {DATASET.KG.name}")
print(f"card   : {DATASET.CARD}")
print("running the setup tree\n")

telemetry.reset()      # counts are module-level; start this run from zero

trace = []
tool_calls = []        # (agent, tool_name) in event order, for the proof
for event in runner.run(
        user_id=USER, session_id=SESSION,
        new_message=types.Content(role="user", parts=[types.Part(
            text="Begin the observation.")])):
    author = getattr(event, "author", "?")
    content = getattr(event, "content", None)
    for part in getattr(content, "parts", None) or []:
        if getattr(part, "function_call", None):
            call = part.function_call
            line = f"[{author}] -> {call.name}({dict(call.args or {})})"
            tool_calls.append((author, call.name))
        elif getattr(part, "function_response", None):
            head = str(part.function_response.response)[:90].replace("\n", " ")
            line = f"[{author}] <- {head}"
        elif getattr(part, "text", None):
            line = f"[{author}] {part.text.strip()[:200]}"
        else:
            continue
        trace.append(line)
        if not args.quiet:
            print(line)

# ---- what landed in state -------------------------------------------------
session = asyncio.run(session_service.get_session(app_name=APP, user_id=USER,
                                                  session_id=SESSION))
state = dict(session.state)


def parsed(state_key):
    raw = state.get(state_key) or ""
    return json.loads(raw) if raw else None


personas = [parsed(k) for k in PERSONA_KEYS]
norms = [parsed(k) for k in NORMS_KEYS]
scopes = [parsed(k) for k in SCOPE_KEYS]
scanners_used = [parsed(k) or {} for k in SCANNER_KEYS]
served = [parsed(k) or {} for k in SERVED_KEYS]
verdicts = [parsed(k) or {} for k in VERDICT_KEYS]

# ---- the blindness proof --------------------------------------------------
# Per observer: position of its norms declaration vs its first data-tool call.
# Event order within one agent is preserved, so index comparison is the proof.
blindness = []
for name in OBSERVER_NAMES:
    declared_at = first_data_at = None
    for position, (agent, tool_name) in enumerate(tool_calls):
        if agent != name:
            continue
        if tool_name == "declare_semantics" and declared_at is None:
            declared_at = position
        if tool_name in DATA_TOOL_NAMES and first_data_at is None:
            first_data_at = position
    verified = (declared_at is not None
                and (first_data_at is None or declared_at < first_data_at))
    blindness.append({"agent": name, "declared_at": declared_at,
                      "first_data_call_at": first_data_at,
                      "verified": verified})

blindness_holds = all(entry["verified"] for entry in blindness)
health = telemetry.health()

print("\n" + "=" * 68)
print(telemetry.render(health))
print()
for entry in blindness:
    if entry["verified"]:
        looked = ("never looked at the data" if entry["first_data_call_at"] is None
                  else f"first data call at #{entry['first_data_call_at']}")
        print(f"  BLINDNESS VERIFIED  {entry['agent']}: norms at "
              f"#{entry['declared_at']}, {looked}")
    else:
        print(f"  *** BLINDNESS VIOLATED  {entry['agent']}: "
              f"norms at {entry['declared_at']}, first data call at "
              f"{entry['first_data_call_at']} -- THIS RUN IS INVALID ***")
print()

import collections  # noqa: E402

from loaders.context import get_context  # noqa: E402

context = get_context()

for name, persona, norm, scope, gens, mine, judged in zip(
        OBSERVER_NAMES, personas, norms, scopes, scanners_used, served,
        verdicts):
    print(f"  {name}")
    print(f"    persona:   {(persona or {}).get('persona', 'MISSING')}")
    if norm:
        print(f"    normal:    {norm['normal']}")
        print(f"    anomalous: {norm['anomalous']}")
        print(f"    lets pass: {norm['lets_pass']}")
    else:
        print("    norms:     MISSING")
    if scope:
        print(f"    scope:     {', '.join(e['label'] for e in scope['scope'])}")
    else:
        print("    scope:     MISSING")
    for scanner_name, why in gens.items():
        print(f"    assistant: {scanner_name} -- {why}")
    counts = collections.Counter(v["verdict"] for v in judged.values())
    print(f"    judged {len(judged)}/{len(mine)} served: "
          f"{dict(counts) if counts else 'none'}")
    for cid, v in sorted(judged.items())[:4]:
        print(f"      {cid} [{v['verdict']:>12}] "
              f"{context.triple_text(tuple(v['triple']))}")
        print(f"          {v['why']}")
    print()

everything_placed = (all(personas) and all(norms) and all(scopes)
                     and all(served) and all(verdicts)
                     and all(len(j) == len(s) for j, s in zip(verdicts, served)))
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out = DATASET.RUNS / f"run_{stamp}_observers.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "dataset": DATASET.NAME,
    #: the exact card this run's phase 1 could see -- provenance
    "card": DATASET.CARD,
    "status": ("invalid" if not blindness_holds
               else "truncated" if health["truncated"]
               else "completed" if everything_placed
               else "incomplete"),
    "health": health,
    "blindness": blindness,
    "personas": personas,
    "norms": norms,
    "scopes": scopes,
    "scanners": scanners_used,
    "served": served,
    "verdicts": verdicts,
    "trace": trace,
}, indent=2), encoding="utf-8")
print(f"  saved to {out.relative_to(ROOT)}")
