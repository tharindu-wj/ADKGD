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
from datetime import datetime


def save_run(goal, backend_name, spec, trace):
    """Write one run -- spec plus the full agent trace -- to its own file.

    Files land in runs/, named by timestamp and backend, e.g.
        runs/run_20260809_182848_gemini.json

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
        The derived viewpoint dict, or None if the agent never finalised.
    trace:
        One entry per step: thinking, tool, args, and the FULL tool result.
        The console truncates long results for readability; the trace never does.

    Returns
    -------
    The path written, so the caller can tell the user where the run landed.
    """
    os.makedirs("runs", exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("runs", f"run_{run_id}_{backend_name}.json")

    with open(path, "w") as f:
        json.dump({
            "run_id": run_id,
            "backend": backend_name,
            "goal": goal,
            "status": "completed" if spec else "exhausted",   # did the agent finalise,
            "steps_taken": len(trace),                        # or run out of steps?
            "final_spec": spec,          # None if exhausted; else duplicated from the
            "trace": trace,              # last trace entry so it is easy to grab
        }, f, indent=2)
    return path
