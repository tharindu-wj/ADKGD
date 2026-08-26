"""Print every context tool's output, for human eyes. No agent, no API.

    python scripts/check_context.py

Run this before wiring any agent. Whatever these print is exactly what the
agents will read -- if something here is wrong or unreadable, every run
downstream inherits it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.describe_dataset import describe_dataset  # noqa: E402
from tools.describe_relation import describe_relation  # noqa: E402
from tools.explain_term import explain_term  # noqa: E402
from tools.inspect_triples import inspect_triples  # noqa: E402


def banner(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


banner("describe_dataset()")
print(describe_dataset())

# Probes are drawn from whatever dataset is loaded -- this rig must work
# unchanged when loaders/active.py points somewhere else.
from loaders.context import get_context  # noqa: E402
ctx = get_context()
probe_relation = ctx.all_relation_labels()[0]
probe_entity = sorted(info["label"] for info in ctx.entities.values()
                      if len(info["label"]) >= 6)[0]
near_miss = probe_entity[:4]

banner(f"describe_relation('{probe_relation}')")
print(describe_relation(probe_relation))

banner("describe_relation('zz_no_such_relation')  -- must be a readable error")
print(describe_relation("zz_no_such_relation"))

banner(f"explain_term('{probe_entity}')")
print(explain_term(probe_entity))

banner(f"explain_term('{probe_relation}')")
print(explain_term(probe_relation))

banner(f"explain_term('{near_miss}')  -- a near-miss, must suggest close names")
print(explain_term(near_miss))

banner(f"inspect_triples('{probe_relation}', 5)")
print(inspect_triples(probe_relation, 5))

banner("inspect_triples(n=5)  -- whole graph")
print(inspect_triples(n=5))

banner("determinism -- same seed twice, then a different seed")
a = inspect_triples(probe_relation, 3, seed=7)
b = inspect_triples(probe_relation, 3, seed=7)
c = inspect_triples(probe_relation, 3, seed=8)
print(f"  same seed identical: {a == b}")
print(f"  different seed differs: {a != c}")
