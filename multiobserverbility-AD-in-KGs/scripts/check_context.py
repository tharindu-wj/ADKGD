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
from tools.show_examples import show_examples  # noqa: E402


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

banner("explain_term('Leonhard Euler')")
print(explain_term("Leonhard Euler"))

banner("explain_term('sibling')")
print(explain_term("sibling"))

banner("explain_term('Marie')  -- a near-miss, must suggest close names")
print(explain_term("Marie"))

banner("show_examples('spouse', 5)")
print(show_examples("spouse", 5))

banner("show_examples(n=5)  -- whole graph")
print(show_examples(n=5))

banner("determinism -- same seed twice, then a different seed")
a = show_examples("spouse", 3, seed=7)
b = show_examples("spouse", 3, seed=7)
c = show_examples("spouse", 3, seed=8)
print(f"  same seed identical: {a == b}")
print(f"  different seed differs: {a != c}")
