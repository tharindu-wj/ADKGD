"""Writing one run -- the derived spec plus the full agent trace -- to its own file.

WHY THIS IS NOT INSIDE AN ORCHESTRATOR
--------------------------------------
Both orchestrations (the hand-written loop and the LangChain one to come) must
write the SAME run-file schema, or the comparison between them is not a
comparison at all -- it is two systems logging different things. Keeping the
writer here, outside both, is what makes that guarantee structural rather than
a matter of remembering.

Run files are experimental artifacts, not logs: they are the evidence base for
the derivation-variance analysis, which is why runs/ is committed to git.

SCHEMA RULE (PROJECT_SPEC INV-8)
--------------------------------
Append-only. New fields may be added; existing fields are never renamed or
repurposed, because saved runs must stay readable by later analysis code.
"""

import json
import os
import pathlib
from datetime import datetime

#: runs/ is anchored to the project root (the parent of utils/), not to the
#: current directory. Without this, `adk web` -- which may be launched from
#: anywhere -- would scatter run files into whichever folder you happened to be in.
RUNS_DIR = pathlib.Path(__file__).resolve().parents[1] / "runs"


def save_run(user_prompt, backend_name, specs, trace, orchestrator="custom",
             findings=None, summary=None):
    """Write one run -- spec plus the full agent trace -- to its own file.

    Files land in runs/, named by timestamp, orchestrator and backend, e.g.
        runs/run_20260809_182848_custom_gemini.json
        runs/run_20260809_201500_adk_gemini.json

    One file per run (never overwritten) is what makes the variance experiment
    possible: run the same goal five times, then compare the five files to see
    how differently the agent explored and what it settled on.

    Parameters
    ----------
    user_prompt:
        Exactly what the user typed, verbatim. For the custom loop that is a
        goal; for the ADK agent it is a broad question it then decomposes.
        Deliberately NOT called "goal": every observer's own goal lives in its
        spec, and having both meanings share one word made run files hard to
        read.
    backend_name:
        "dummy", "gemini", ... Used in the filename and recorded in the file.
    specs:
        The derived viewpoints, one per observer point, as a list. An empty list
        means the agent never finalised. A single dict is also accepted and
        wrapped, so callers with one viewpoint need not build a list themselves.
    trace:
        One entry per step: thinking, tool, args, and the FULL tool result.
        The console truncates long results for readability; the trace never does.
    orchestrator:
        Which machinery drove the loop -- "custom" (the hand-written loop) or
        "adk". Recorded because BOTH write here: without it a run file cannot say
        which orchestration produced it, and the comparison between them is
        unmeasurable. `backend` alone does not distinguish them -- both say
        "gemini".
    findings, summary:
        The findings-phase output, present only when the agent answered a broad
        question (self-authored observer points -> viewpoints -> verdict
        comparison -> explained findings). None otherwise; written only when set.

    Returns
    -------
    The path written, so the caller can tell the user where the run landed.

    SCHEMA NOTE
    -----------
    The schema was simplified once, on 11 Aug 2026, breaking the append-only
    rule (PROJECT_SPEC INV-8) deliberately and while no analysis code existed to
    break. Two changes:

      `goal` / `goals`  ->  `user_prompt`
          `goal` at the top level collided with each spec's OWN `goal` -- the
          observer point -- in the same file, and `goals` had decayed into a
          one-item copy of it once the agent began authoring its own observer
          points. Now `goal` appears in exactly one place and means one thing.

      `final_spec` + `final_specs`  ->  `final_specs` only
          Carrying a singular field beside a plural one meant every reader
          needed a branch, for no gain: a run with one viewpoint is simply a
          list of length one.

    Runs written before that date keep the old fields -- they are evidence and
    are left untouched. Analysis code that must read both eras:

        prompt = d.get("user_prompt") or d.get("goal")
        specs  = d.get("final_specs") or ([d["final_spec"]] if d.get("final_spec") else [])
    """
    os.makedirs(RUNS_DIR, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RUNS_DIR / f"run_{run_id}_{orchestrator}_{backend_name}.json"

    # Timestamps are second-resolution, so two runs finishing in the same second
    # would land on the same filename and the second would silently destroy the
    # first. Never overwrite a run: add a suffix instead. Real agent runs take
    # seconds, but scripted batches (the variance experiment runs one goal N
    # times) can easily collide, and a lost run is a lost measurement.
    suffix = 2
    while path.exists():
        run_id = f"{datetime.now():%Y%m%d_%H%M%S}-{suffix}"
        path = RUNS_DIR / f"run_{run_id}_{orchestrator}_{backend_name}.json"
        suffix += 1

    # Always a list, even for one viewpoint -- readers never need a branch.
    # A caller passing a single dict gets it wrapped.
    spec_list = [s for s in (specs if isinstance(specs, list) else [specs]) if s]

    record = {
        "run_id": run_id,
        "orchestrator": orchestrator,   # "custom" | "adk"
        "backend": backend_name,        # "dummy" | "gemini"
        "user_prompt": user_prompt,     # what the user typed, verbatim
        "status": "completed" if spec_list else "exhausted",
        "steps_taken": len(trace),
        "final_specs": spec_list,       # one entry per observer point; each
        "trace": trace,                 # entry carries its OWN goal
    }
    if findings is not None:
        record["findings"] = findings   # findings-phase only (broad questions)
    if summary is not None:
        record["summary"] = summary

    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return str(path)
