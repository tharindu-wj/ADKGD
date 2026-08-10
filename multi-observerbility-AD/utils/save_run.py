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


def save_run(goal, backend_name, spec, trace, orchestrator="custom", goals=None):
    """Write one run -- spec plus the full agent trace -- to its own file.

    Files land in runs/, named by timestamp, orchestrator and backend, e.g.
        runs/run_20260809_182848_custom_gemini.json
        runs/run_20260809_201500_adk_gemini.json

    One file per run (never overwritten) is what makes the variance experiment
    possible: run the same goal five times, then compare the five files to see
    how differently the agent explored and what it settled on.

    Parameters
    ----------
    goal:
        The observer point, verbatim -- the sentence the user supplied.
    backend_name:
        "dummy", "gemini", ... Used in the filename and recorded in the file.
    spec:
        The derived viewpoint dict, or None if the agent never finalised. May
        also be a LIST of specs -- one per goal -- when an agent was given
        several goals at once; see the schema note below.
    trace:
        One entry per step: thinking, tool, args, and the FULL tool result.
        The console truncates long results for readability; the trace never does.
    orchestrator:
        Which machinery drove the loop -- "custom" (the hand-written loop) or
        "adk". Recorded because BOTH write here: without it a run file cannot say
        which orchestration produced it, and the comparison between them is
        unmeasurable. `backend` alone does not distinguish them -- both say
        "gemini".
    goals:
        The individual goals the agent was given, split out of `goal`. One item
        means the cell-1 condition (one agent, one goal); two or more means
        cell 2 (one agent, several goals at once). The condition is read off
        len(goals) -- there is no separate label to keep in sync.

    Returns
    -------
    The path written, so the caller can tell the user where the run landed.

    SCHEMA NOTE (INV-8: append-only)
    --------------------------------
    Older runs carry `final_spec` (a single dict). Multi-goal runs carry
    `final_specs` (a list, one per goal) and leave `final_spec` null. Nothing was
    renamed, so every run file ever written stays readable. Analysis code reads
    both shapes in one line:

        specs = d.get("final_specs") or ([d["final_spec"]] if d.get("final_spec") else [])
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

    # Accept either one spec or a list of them, and normalise into both fields:
    # single-goal runs keep the historical `final_spec`, multi-goal runs use
    # `final_specs`. Callers never have to care which they are producing.
    spec_list = [s for s in (spec if isinstance(spec, list) else [spec]) if s]
    single_spec = spec_list[0] if len(spec_list) == 1 else None
    multi_specs = spec_list if len(spec_list) > 1 else None

    with open(path, "w") as f:
        json.dump({
            "run_id": run_id,
            "orchestrator": orchestrator,   # "custom" | "adk"
            "backend": backend_name,        # "dummy" | "gemini"
            "goal": goal,                   # the raw message, verbatim
            "goals": goals if goals is not None else [goal],  # 1 = cell 1, 2+ = cell 2
            "status": "completed" if spec_list else "exhausted",
            "steps_taken": len(trace),
            "final_spec": single_spec,      # set when there is exactly one spec
            "final_specs": multi_specs,     # set when there are several
            "trace": trace,
        }, f, indent=2)
    return str(path)
