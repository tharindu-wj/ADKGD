"""The observer agents.

    adk web .                        serves this package
    adk run agents                   runs it in the terminal
    python scripts/3_run_observers.py    runs one observation and records it

ADK imports `agents.agent` and reads `root_agent` out of it, so that module
stays the front door. The rest is split by job:

    config.py           the model, retry policy, budgets, tool lists
    telemetry.py        did every model call actually happen?
    phase_gate.py       the data stays locked until an observer's norms exist
    root_agent.py       assigns each observer its perspective, from the card
    observer_agents.py  the two observers: blind norms -> scope -> find -> judge
    reviewer_agents.py  the same observers returning for blind second opinions
    agent.py            assembles the tree ADK loads

THIS FILE EXISTS FOR ONE REASON. The submodules import `loaders` and `tools`,
which live at the repo root rather than in this package. Python runs a
package's __init__ before any module inside it, so the path fix belongs here
and nowhere else. Import nothing here -- a submodule imported from __init__
becomes a package attribute, and an attribute named `root_agent` would shadow
what ADK looks up.
"""
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
