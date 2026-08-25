"""Run the ADK agent tree once and save the run.

    python scripts/3_run_agentic_detector.py
    adk run agents                 the same tree, interactively

Needs a Gemini key: GEMINI_KEY / GOOGLE_API_KEY in the environment, or a .env
beside this repo.

No labels are read anywhere in here. Scoring what came out is a separate step.
"""
import json
import os
import sys
from pathlib import Path

# Scripts live in scripts/, so Python puts THAT on sys.path, not the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import datetime

from loaders.active import DATASET

ap = argparse.ArgumentParser()
ap.add_argument("--quiet", action="store_true")
args = ap.parse_args()

if not DATASET.KG.exists():
    raise SystemExit(f"missing {DATASET.KG}. Run scripts/1_inject_anomalies.py first.")


def find_key():
    """ADK reads GOOGLE_API_KEY. Accept the other spellings and a .env too."""
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    for path in (ROOT / ".env", ROOT.parent / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_KEY"):
                return value.strip().strip('"').strip("'")
    return None


key = find_key()
if not key:
    raise SystemExit(
        "No Gemini key. Set GEMINI_KEY in the environment, or put a line\n"
        f"    GEMINI_KEY=...\nin {ROOT / '.env'} or {ROOT.parent / '.env'}.")
os.environ["GOOGLE_API_KEY"] = key

from google.adk.runners import Runner          # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types                 # noqa: E402

from agents import telemetry  # noqa: E402
from agents.agent import GOAL_KEYS, SEM_KEYS, SPEC_KEYS, root_agent  # noqa: E402

APP, USER, SESSION = "kg_audit", "local", "run"

session_service = InMemorySessionService()
import asyncio  # noqa: E402
asyncio.run(session_service.create_session(app_name=APP, user_id=USER,
                                           session_id=SESSION))

runner = Runner(app_name=APP, agent=root_agent, session_service=session_service)

print(f"dataset: {DATASET.NAME}   file: {DATASET.KG.name}")
print(f"running the ADK tree: {root_agent.name}\n")

telemetry.reset()      # counts are module-level; start this run from zero

trace = []
for event in runner.run(
        user_id=USER, session_id=SESSION,
        new_message=types.Content(role="user", parts=[types.Part(
            text=f"Audit the {DATASET.NAME} graph.")])):
    author = getattr(event, "author", "?")
    content = getattr(event, "content", None)
    for part in getattr(content, "parts", None) or []:
        if getattr(part, "function_call", None):
            call = part.function_call
            line = f"[{author}] -> {call.name}({dict(call.args or {})})"
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


def parsed(key):
    raw = state.get(key) or ""
    try:
        return json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return None


goals = [state.get(k) or "" for k in GOAL_KEYS]
semantics = [parsed(k) for k in SEM_KEYS]
specs = [parsed(k) for k in SPEC_KEYS]

health = telemetry.health()

print("\n" + "=" * 68)
print(telemetry.render(health))
print()
for i, goal in enumerate(goals, 1):
    print(f"  goal {i}: {goal or 'MISSING'}")
print()
for i, sem in enumerate(semantics, 1):
    if not sem:
        print(f"  frame {i}: MISSING")
        continue
    print(f"  frame {i}: in scope -- {', '.join(sem['relations'])}")
    for field in ("entities", "normal", "suspicious", "impossible"):
        if sem.get(field):
            print(f"           {field + ':':<12}{sem[field]}")
print()
for i, spec in enumerate(specs, 1):
    print(f"  spec {i}: {spec or 'MISSING'}")

# ---- carry out each spec, and record what it found ------------------------
# The agents decided; this executes their decision. Deterministic, no LLM, and
# no labels -- the run file must say what was FOUND, not whether it was right.
# Saving the full ranking rather than just the flagged slice is what lets the
# evaluator report at any cut-off later without re-running anything.
import numpy as np  # noqa: E402

from loaders import graph  # noqa: E402
from tools.run_scorer import SCORERS  # noqa: E402

triples = graph.load_triples(DATASET.KG)
findings = []

for i, spec in enumerate(specs, 1):
    if not spec or spec.get("scorer") not in SCORERS:
        print(f"\n  agent {i}: no usable spec, nothing to carry out")
        continue

    mod = SCORERS[spec["scorer"]]
    if mod.NEEDS_MODEL:
        model_dir = DATASET.MODELS / "distmult"
        if not (model_dir / "trained_model.pkl").exists():
            raise SystemExit(f"missing {model_dir}. Run scripts/2_train_plausibility_scorer.py first.")
        values = mod.score(triples, model_dir=model_dir, kg_path=DATASET.KG)
    else:
        values = mod.score(triples)

    v = np.asarray(values, dtype=float)
    anomaly = -v if mod.DIRECTION < 0 else v      # HIGH = anomalous, always
    order = np.argsort(-anomaly, kind="stable")
    n_flag = max(1, int(round(float(spec["budget"]) * len(triples))))

    findings.append({
        "agent": i,
        "scorer": spec["scorer"],
        "budget": float(spec["budget"]),
        "flagged": n_flag,
        # The frame this agent committed to BEFORE it was allowed to score.
        # Kept beside the ranking so the evaluator can ask whether the flags
        # match what the agent said it was looking for -- no labels needed.
        "semantics": semantics[i - 1],
        "ranked": [[*triples[j], round(float(v[j]), 6)] for j in order],
    })
    worst = triples[order[0]]
    print(f"\n  agent {i}: flagged {n_flag} of {len(triples)}, "
          f"worst is {worst[0]} {worst[1]} {worst[2]}")

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out = ROOT / "runs" / f"run_{stamp}_adk.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "dataset": DATASET.NAME,
    "orchestration": "adk",
    # "truncated" is not the same failure as "incomplete", and conflating them
    # is what let a rate limit masquerade as agent behaviour for days. A
    # truncated run says nothing about how the agents behave.
    "status": ("truncated" if health["truncated"]
               else "completed" if all(goals) and all(specs) and all(semantics)
               else "incomplete"),
    "health": health,
    "goals": goals,
    "semantics": semantics,
    "specs": specs,
    "findings": findings,
    "trace": trace,
}, indent=2), encoding="utf-8")
print(f"\n  saved to {out.relative_to(ROOT)}")
