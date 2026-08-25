"""Prove the two-phase gate and every tool guard, offline. No agent, no API.

    python scripts/check_gate.py

Every line must end PASS. This is the blindness machinery -- if any of it is
loose, the norms stop being blind and the whole separation is decoration.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.phase_gate import keep_norms_blind  # noqa: E402
from tools.assign_perspective import assign_perspective  # noqa: E402
from tools.declare_semantics import declare_semantics  # noqa: E402
from tools.select_scope import select_scope  # noqa: E402

failures = []


def check(label, actual):
    verdict = "PASS" if actual else "FAIL"
    if not actual:
        failures.append(label)
    print(f"  {verdict}  {label}")


class FakeTool:
    def __init__(self, name):
        self.name = name


class FakeToolContext:
    """One shared state dict, one caller name -- what ADK really provides."""
    def __init__(self, agent_name, state):
        self.agent_name = agent_name
        self.state = state


state = {}
agent_1 = FakeToolContext("sub_agent_1", state)
agent_2 = FakeToolContext("sub_agent_2", state)
root = FakeToolContext("root", state)

print("\nassign_perspective (the root's guard rails)")
check("rejects an unknown auditor",
      assign_perspective("agent_x", "p", root).startswith("ERROR"))
check("rejects an empty persona",
      assign_perspective("sub_agent_1", "  ", root).startswith("ERROR"))
first = assign_perspective("sub_agent_1", "You judge by the rules.", root)
check("accepts the first persona", first.startswith("Recorded"))
check("refuses rewriting a placed persona",
      assign_perspective("sub_agent_1", "Changed my mind.", root).startswith("ERROR"))
check("refuses an identical persona for the twin (punctuation-proof)",
      assign_perspective("sub_agent_2", "You judge, by the rules!", root).startswith("ERROR"))
second = assign_perspective("sub_agent_2", "You judge only facts.", root)
check("accepts a genuinely different persona", second.startswith("Recorded"))
check("announces completion", "done" in second)

print("\nphase 1: the data is locked until norms exist")
for tool_name in ("describe_dataset", "describe_relation", "lookup", "sample"):
    blocked = keep_norms_blind(FakeTool(tool_name), {}, agent_1)
    check(f"{tool_name} blocked before norms", blocked is not None)
check("declare_semantics itself is never blocked",
      keep_norms_blind(FakeTool("declare_semantics"), {}, agent_1) is None)
check("select_scope refuses before norms",
      select_scope(["spouse"], "w", agent_1).startswith("ERROR"))

print("\ndeclaring norms")
check("rejects an empty field",
      declare_semantics("n", " ", "l", agent_1).startswith("ERROR"))
check("root cannot declare norms",
      declare_semantics("n", "a", "l", root).startswith("ERROR"))
ok = declare_semantics(
    "committed relationships are mutual and exclusive",
    "a one-sided record of an inherently mutual bond, even if the fact is real",
    "unusual arrangements that are honestly recorded", agent_1)
check("accepts complete blind norms", ok.startswith("Recorded"))
check("refuses re-declaration (norms are immutable)",
      declare_semantics("x", "y", "z", agent_1).startswith("ERROR"))
check("refuses identical norms for the twin",
      declare_semantics(
          "Committed relationships are mutual, and exclusive.",
          "a one-sided record of an inherently mutual bond -- even if the fact is real",
          "unusual arrangements that are honestly recorded!", agent_2).startswith("ERROR"))

print("\nphase 2: the data opens for the declared auditor only")
check("data open for sub_agent_1 after its norms",
      keep_norms_blind(FakeTool("describe_dataset"), {}, agent_1) is None)
check("data still locked for sub_agent_2 (no norms yet)",
      keep_norms_blind(FakeTool("describe_dataset"), {}, agent_2) is not None)

print("\nselecting scope")
check("rejects an unknown relation",
      select_scope(["marriage"], "w", agent_1).startswith("ERROR"))
check("rejects an empty why",
      select_scope(["spouse"], "  ", agent_1).startswith("ERROR"))
ok = select_scope(["spouse", "unmarried partner", "sibling"],
                  "my mutuality norm concerns inherently two-way bonds", agent_1)
check("accepts a valid scope", ok.startswith("Recorded"))
check("refuses re-selection (scope is a commitment)",
      select_scope(["child"], "w", agent_1).startswith("ERROR"))
check("stores resolved ids", '"P26"' in state["scope_1"])

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all checks pass -- the separation is enforced, not suggested.")
