"""Print what every generator finds, for human eyes. No agent, no API.

    python scripts/check_generators.py

Run before any agent touches them. Whatever a generator surfaces is exactly
what a judge will be handed -- if the candidates here are junk, every verdict
downstream is junk with a rationale.

implausible_links is skipped politely until 2_train_scorer.py has run.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loaders.context import get_context  # noqa: E402
from tools.generators import (implausible_links, multiplicity_outliers,  # noqa: E402
                              reciprocity_gaps, type_clashes)

ctx = get_context()
ALL_RELATIONS = set(ctx.relations)

for generator in (reciprocity_gaps, multiplicity_outliers, type_clashes,
                  implausible_links):
    print("\n" + "=" * 72)
    print(f"  {generator.NAME}  (scope = every relation)")
    print("=" * 72)
    try:
        found = generator.find(ALL_RELATIONS, ctx)
    except RuntimeError as refusal:
        print(f"  refused: {refusal}")
        continue

    again = generator.find(ALL_RELATIONS, ctx)
    print(f"  {len(found)} candidates   deterministic: {found == again}")
    for triple, note in found[:6]:
        print(f"    {ctx.triple_text(triple)}")
        print(f"        {note}")
    if len(found) > 6:
        print(f"    ... and {len(found) - 6} more")

print("\nscoped run -- reciprocity_gaps on spouse only:")
spouse_only = {ctx.find_relation("spouse")}
for triple, note in reciprocity_gaps.find(spouse_only, ctx):
    print(f"    {ctx.triple_text(triple)}   [{note}]")
