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


# Relation probes come from the loaded dataset, so the rig ports with it.
# REL_A must be one the one_way_links scanner can fire on (mostly two-way,
# with at least one one-way edge), because the phase-3 checks below serve a
# candidate from it. Derived by the scanner's own criterion, not by name.
import collections  # noqa: E402

from loaders.context import get_context  # noqa: E402

_ctx = get_context()
_by_relation = collections.defaultdict(list)
for _t in _ctx.triples:
    _by_relation[_t[1]].append(_t)
_gappy = []
for _rid, _triples in _by_relation.items():
    _present = set(_triples)
    _one_way = sum(1 for t in _triples if (t[2], t[1], t[0]) not in _present)
    _symmetry = 1 - _one_way / len(_triples)
    if _symmetry >= 0.5 and _one_way:
        _gappy.append((_symmetry, _rid))
if not _gappy:
    raise SystemExit("this rig needs a mostly-two-way relation with at least "
                     "one one-way edge; the loaded dataset has none.")
REL_A_ID = max(_gappy)[1]
REL_A = _ctx.relation_label(REL_A_ID)
_others = [l for l in _ctx.all_relation_labels() if l != REL_A]
REL_B, REL_C = _others[0], _others[1 % len(_others)]

state = {}
agent_1 = FakeToolContext("observer_1", state)
agent_2 = FakeToolContext("observer_2", state)
root = FakeToolContext("root", state)

print("\nassign_perspective (the root's guard rails)")
check("rejects an unknown observer",
      assign_perspective("agent_x", "p", root).startswith("ERROR"))
check("rejects an empty persona",
      assign_perspective("observer_1", "  ", root).startswith("ERROR"))
first = assign_perspective("observer_1", "You judge by the rules.", root)
check("accepts the first persona", first.startswith("Recorded"))
check("refuses rewriting a placed persona",
      assign_perspective("observer_1", "Changed my mind.", root).startswith("ERROR"))
check("refuses an identical persona for the twin (punctuation-proof)",
      assign_perspective("observer_2", "You judge, by the rules!", root).startswith("ERROR"))
second = assign_perspective("observer_2", "You judge only facts.", root)
check("accepts a genuinely different persona", second.startswith("Recorded"))
check("announces completion", "done" in second)

print("\nphase 1: the data is locked until norms exist")
for tool_name in ("describe_dataset", "describe_relation", "explain_term", "inspect_triples"):
    blocked = keep_norms_blind(FakeTool(tool_name), {}, agent_1)
    check(f"{tool_name} blocked before norms", blocked is not None)
check("declare_semantics itself is never blocked",
      keep_norms_blind(FakeTool("declare_semantics"), {}, agent_1) is None)
check("select_scope refuses before norms",
      select_scope([REL_A], "w", agent_1).startswith("ERROR"))

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

print("\nphase 2: the data opens for the declared observer only")
check("data open for observer_1 after its norms",
      keep_norms_blind(FakeTool("describe_dataset"), {}, agent_1) is None)
check("data still locked for observer_2 (no norms yet)",
      keep_norms_blind(FakeTool("describe_dataset"), {}, agent_2) is not None)

print("\nselecting scope")
check("rejects an unknown relation",
      select_scope(["zz_no_such_relation"], "w", agent_1).startswith("ERROR"))
check("rejects an empty why",
      select_scope([REL_A], "  ", agent_1).startswith("ERROR"))
ok = select_scope([REL_A, REL_B, REL_C],
                  "my mutuality norm concerns inherently two-way bonds", agent_1)
check("accepts a valid scope", ok.startswith("Recorded"))
check("refuses re-selection (scope is a commitment)",
      select_scope([REL_B], "w", agent_1).startswith("ERROR"))
check("stores resolved ids", REL_A_ID in state["scope_1"])

print("\nphase 3: finding and judging")
from tools.find_suspects import find_suspects  # noqa: E402
from tools.submit_verdicts import submit_verdicts  # noqa: E402

fresh = FakeToolContext("observer_2", {})
check("find_suspects refuses without a scope",
      find_suspects("one_way_links", "w", 1, fresh).startswith("ERROR"))
check("submit_verdicts refuses before anything is served",
      submit_verdicts([{"id": "c1", "verdict": "ok", "why": "w"}],
                      fresh).startswith("ERROR"))

check("unknown assistant is a readable error",
      find_suspects("psychic", "w", 1, agent_1).startswith("ERROR"))
check("first call to an assistant requires a why",
      find_suspects("one_way_links", "", 1, agent_1).startswith("ERROR"))
page = find_suspects("one_way_links",
                       "my mutuality norm concerns two-way bonds", 1, agent_1)
check("candidates served with stable ids", "c1." in page)
check("serving is recorded", "served_1" in state and "c1" in state["served_1"])

check("verdict on an id never served is refused, batch not recorded",
      submit_verdicts([{"id": "c999", "verdict": "ok", "why": "w"}],
                      agent_1).startswith("ERROR"))
check("a made-up verdict word is refused",
      submit_verdicts([{"id": "c1", "verdict": "guilty", "why": "w"}],
                      agent_1).startswith("ERROR"))
check("an empty why is refused",
      submit_verdicts([{"id": "c1", "verdict": "ok", "why": " "}],
                      agent_1).startswith("ERROR"))
ok = submit_verdicts([{"id": "c1", "verdict": "anomaly",
                       "why": "one-sided record of a mutual bond"}], agent_1)
check("a valid verdict is recorded", ok.startswith("Recorded"))
check("progress says what remains", "unjudged" in ok or "done" in ok)

# Second-opinion guards -- reuses the state above (agent_1 judged c1 'anomaly').
print("\nsecond opinions")
from tools.review_candidates import review_candidates  # noqa: E402
from tools._observers import principal_of  # noqa: E402

check("principal resolution", principal_of("observer_1_reviewer") == "observer_1")
check("a principal cannot fetch reviews",
      review_candidates(agent_1).startswith("ERROR"))
reviewer_2 = FakeToolContext("observer_2_reviewer", state)
page = review_candidates(reviewer_2)
check("reviewer 2 receives agent 1's flag, blind",
      "r1." in page and "anomaly" not in page and "observer" not in page.split("review")[0])
check("review serving lands in the PRINCIPAL's store", "r1" in state["served_2"])
ok = submit_verdicts([{"id": "r1", "verdict": "ok",
                       "why": "really married; a one-sided record is still a real fact"}],
                     reviewer_2)
check("reviewer's verdict recorded for the principal",
      ok.startswith("Recorded") and "r1" in state["verdicts_2"])
again = review_candidates(reviewer_2)
check("re-fetch does not duplicate", "already judged" in again or "r2" not in again)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all checks pass -- the separation is enforced, not suggested.")
