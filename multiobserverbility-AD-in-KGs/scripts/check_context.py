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
from tools.lookup import lookup  # noqa: E402
from tools.sample import sample  # noqa: E402


def banner(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


banner("describe_dataset()")
print(describe_dataset())

banner("describe_relation('spouse')")
print(describe_relation("spouse"))

banner("describe_relation('diplomatic relation')")
print(describe_relation("diplomatic relation"))

banner("describe_relation('borders')  -- must be a readable error")
print(describe_relation("borders"))

banner("lookup('Leonhard Euler')")
print(lookup("Leonhard Euler"))

banner("lookup('sibling')")
print(lookup("sibling"))

banner("lookup('Marie')  -- a near-miss, must suggest close names")
print(lookup("Marie"))

banner("sample('spouse', 5)")
print(sample("spouse", 5))

banner("sample(n=5)  -- whole graph")
print(sample(n=5))

banner("determinism -- same seed twice, then a different seed")
a = sample("spouse", 3, seed=7)
b = sample("spouse", 3, seed=7)
c = sample("spouse", 3, seed=8)
print(f"  same seed identical: {a == b}")
print(f"  different seed differs: {a != c}")
