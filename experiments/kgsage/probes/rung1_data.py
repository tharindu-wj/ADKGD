"""RUNG 1 - Read the raw material.   (local, no deps)

Question: what is the model even looking at?
"""
from _common import banner, hint, load_dataset


def main():
    banner("RUNG 1 - Read the raw material")
    kg, data_dir, enc, gan = load_dataset()
    id2ent, id2rel = kg["id2ent"], kg["id2rel"]

    print(f"\n{kg['n_ent']:,} entities | {kg['n_rel']} relations | "
          f"{len(kg['triples_train']):,} train triples")

    print("\nFive facts  (head --[relation]--> tail):")
    for (h, r, t) in kg["triples_train"][:5]:
        print(f"  {id2ent[h]}  --[{id2rel[r]}]-->  {id2ent[t]}")

    # one entity's neighborhood = the set of triples touching it (RGCN's raw input)
    h0 = kg["triples_train"][0][0]
    nbrs = [(h, r, t) for (h, r, t) in kg["triples_train"] if h == h0 or t == h0][:8]
    print(f"\nNeighborhood of '{id2ent[h0]}' ({len(nbrs)} of its triples shown):")
    for (h, r, t) in nbrs:
        print(f"  {id2ent[h]} --[{id2rel[r]}]--> {id2ent[t]}")

    hint("every fact is a (head, relation, tail) of integer IDs. An entity's "
         "neighborhood (all triples touching it) is exactly what the RGCN aggregates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
