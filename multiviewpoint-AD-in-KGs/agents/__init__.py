"""The audit agent tree.

    adk web .                        serves this package
    adk run agents                   runs it in the terminal
    python scripts/3_run_agentic_detector.py    runs it and records what it found

ADK imports `agents.agent` and reads `root_agent` out of it, so that module
stays the front door. Everything else here is split out of it by job:

    config.py      the model, the budget, the state keys, the tool sets
    parsing.py     getting a structured answer back out of the model's text
    root.py        the agent that writes the two goals
    viewpoint_agents.py  the factory that builds the two auditors
    agent.py       assembles those into the tree ADK loads

THIS FILE EXISTS FOR ONE REASON. The submodules import `loaders` and `tools`,
which live at the repo root rather than in this package. Python runs a
package's __init__ before any module inside it, and that is the only ordering
guarantee available once the tree is spread across several files -- so the
path fix belongs here and nowhere else.
"""
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
