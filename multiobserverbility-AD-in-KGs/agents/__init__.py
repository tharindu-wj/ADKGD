"""The audit agents. Milestone 1: the root alone.

    adk web .                     serves this package
    adk run agents                runs it in the terminal
    python scripts/2_run_root.py  runs it and records what it produced

ADK imports `agents.agent` and reads `root_agent` out of it, so that module
stays the front door. The rest is split by job:

    config.py      the model, retry policy, budgets, tool lists
    telemetry.py   did every model call actually happen?
    root_agent.py  the root and its instruction
    agent.py       what ADK loads

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
