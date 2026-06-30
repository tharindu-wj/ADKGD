"""RUNG 2 - Find anti-symmetry by hand.   (local, no deps)

Question: what makes a role-swap a contradiction?  This is the core idea the
whole thesis rests on, and you can derive it by counting.
"""
from collections import defaultdict
from _common import banner, hint, load_dataset


def main():
    banner("RUNG 2 - Find anti-symmetry by hand")
    kg, data_dir, enc, gan = load_dataset()
    id2rel = kg["id2rel"]

    # (h, t) -> set of relations going h -> t
    edges = defaultdict(set)
    for h, r, t in kg["triples_train"]:
        edges[(h, t)].add(r)

    print("\nFor a sample of facts: does the REVERSE (t, r, h) also exist?\n")
    sym = anti = shown = 0
    for h, r, t in kg["triples_train"][:300]:
        has_reverse = r in edges.get((t, h), ())
        sym += has_reverse
        anti += not has_reverse
        if shown < 14:
            kind = "SYMMETRIC  (reverse present)" if has_reverse else "anti-symmetric (no reverse)"
            print(f"  {id2rel[r][:46]:46s} {kind}")
            shown += 1

    print(f"\nin the 300-fact sample: {sym} symmetric, {anti} anti-symmetric")
    hint("anti-symmetric relations are where contradictions come from: if (h,r,t) "
         "holds and (t,r,h) NEVER does, then ASSERTING (t,r,h) is a contradiction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
