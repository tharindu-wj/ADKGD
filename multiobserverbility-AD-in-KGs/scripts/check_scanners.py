"""Print what every scanner finds, for human eyes. No agent, no API.

    python scripts/check_scanners.py

Run before any agent touches them. Whatever a scanner surfaces is exactly
what a judge will be handed -- if the candidates here are junk, every verdict
downstream is junk with a rationale.

unlikely_facts is skipped politely until 2_train_scorer.py has run.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loaders.context import get_context  # noqa: E402
from tools.scanners import (unlikely_facts, too_many_values,  # noqa: E402
                              one_way_links, odd_types)

ctx = get_context()
ALL_RELATIONS = set(ctx.relations)

for scanner in (one_way_links, too_many_values, odd_types,
                  unlikely_facts):
    print("\n" + "=" * 72)
    print(f"  {scanner.NAME}  (scope = every relation)")
    print("=" * 72)
    try:
        found = scanner.find(ALL_RELATIONS, ctx)
    except RuntimeError as refusal:
        print(f"  refused: {refusal}")
        continue

    again = scanner.find(ALL_RELATIONS, ctx)
    print(f"  {len(found)} candidates   deterministic: {found == again}")
    for triple, note in found[:6]:
        print(f"    {ctx.triple_text(triple)}")
        print(f"        {note}")
    if len(found) > 6:
        print(f"    ... and {len(found) - 6} more")

# The scoped probe uses the loaded dataset's rarest relation, so this rig
# works unchanged when loaders/active.py points somewhere else.
import collections  # noqa: E402

counts = collections.Counter(r for _, r, _ in ctx.triples)
probe_id, _ = counts.most_common()[-1]
print(f"\nscoped run -- one_way_links on '{ctx.relation_label(probe_id)}' only:")
for triple, note in one_way_links.find({probe_id}, ctx)[:8]:
    print(f"    {ctx.triple_text(triple)}   [{note}]")
