"""Print what every profiler tool returns, so the output can be read by eye.

These strings are what an agent sees and reasons from. If they are confusing
to you they will be confusing to it, and no prompt will fix that.

    python scripts/check_profiler.py
"""
import sys
from pathlib import Path

# Scripts live in scripts/, so Python puts THAT on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loaders import graph
from loaders.active import DATASET
from tools.registry import TOOLS
from utils import profile


def banner(text):
    print("\n" + "=" * 68)
    print("  " + text)
    print("=" * 68)


if not DATASET.KG.exists():
    raise SystemExit(f"missing {DATASET.KG}. Run scripts/1_inject_anomalies.py first.")

print(f"dataset: {DATASET.NAME}   file: {DATASET.KG.name}")
print(f"registered tools: {', '.join(TOOLS)}")

banner("list_relations()")
print(TOOLS["list_relations"]())

triples = graph.load_triples(DATASET.KG)
relations = [r["relation"] for r in profile.relation_summary(triples)]

for rel in relations:
    banner(f"describe_relation({rel!r})")
    print(TOOLS["describe_relation"](rel))

banner("sample('locatedin', 5)")
print(TOOLS["sample"]("locatedin", 5))

banner("sample('neighbor', 5)")
print(TOOLS["sample"]("neighbor", 5))

# Errors come back as readable text, not exceptions, so an agent can correct
# itself mid-run instead of the run dying.
banner("error paths -- these must READ as advice, not as a stack trace")
print(TOOLS["describe_relation"]("bordering"))
print()
print(TOOLS["sample"]("bordering"))

banner("the cap holds")
print(TOOLS["sample"]("neighbor", 500).splitlines()[0])

banner("determinism")
a = TOOLS["sample"]("neighbor", 5)
b = TOOLS["sample"]("neighbor", 5)
print(f"same seed, two calls, identical: {a == b}")
c = TOOLS["sample"]("neighbor", 5, seed=1)
print(f"different seed gives different rows: {a != c}")
