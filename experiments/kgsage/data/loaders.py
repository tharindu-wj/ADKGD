"""Generic KG loader for KGSAGE.

Loads any TSV-format KG that has train.txt / valid.txt / test.txt in a single
directory. The loader is format-agnostic — works for FB15K-237, WN18RR,
NELL-995, and any custom dataset in the same format.

It returns integer (h, r, t) triples plus the string<->int vocabulary maps,
which is everything the trainer and corruption generation need. It also
returns a directed edge list (edge_index / edge_type) over the TRAIN graph,
which the NeighbourhoodContextEncoder
(kgsage.gan.neighbourhood_context_encoder) consumes for message passing in
Phase 1 to build the context table E'.

Vocab strategy: first-seen ordering. Train.txt is loaded first, so its entities
and relations get the lowest IDs. This matches the standard KGE convention and
means valid/test only-entities (if any) get higher IDs.

Dependency note: this module is deliberately standard-library only (no torch,
no numpy) so that `import kgsage` succeeds on machines without torch. The edge
list is therefore returned as plain Python lists — the encoder tensorises it.
"""
import os


def load_kg(data_dir):
    """Load a KG from `data_dir/{train,valid,test}.txt`.

    Each line in those files is three tab-separated strings:
        head_string<TAB>relation_string<TAB>tail_string

    Returns a dict:
      ent2id, rel2id     : string -> int (vocab; train-first ordering)
      id2ent, id2rel     : int -> string (inverse maps)
      triples_train      : list of (h, r, t) integer tuples
      triples_valid      : list of (h, r, t) integer tuples
      triples_test       : list of (h, r, t) integer tuples
      triple_set_train   : set of train triples (collision check)
      triple_set_all     : set of train+valid+test triples (collision filter)
      edge_index         : [[src...], [dst...]] over train (shape [2, E]), lists
      edge_type          : [rel...] over train (shape [E]), list
      n_ent, n_rel       : vocabulary sizes
    """
    # ─── Step 1: parse train.txt first to lock in the vocab order ──────
    ent2id = {}
    rel2id = {}
    triples_train = _parse_split(
        os.path.join(data_dir, "train.txt"),
        ent2id,
        rel2id,
        add_to_vocab=True,
    )

    # ─── Step 2: parse valid.txt and test.txt; allow vocab extension ───
    # On most KGs the train vocab covers everything, but some have entities
    # or relations that only appear in valid/test — we tolerate this by
    # extending the vocab as we go.
    triples_valid = _parse_split(
        os.path.join(data_dir, "valid.txt"),
        ent2id,
        rel2id,
        add_to_vocab=True,
    )
    triples_test = _parse_split(
        os.path.join(data_dir, "test.txt"),
        ent2id,
        rel2id,
        add_to_vocab=True,
    )

    # ─── Step 3: build inverse maps and triple sets ────────────────────
    id2ent = {i: s for s, i in ent2id.items()}
    id2rel = {i: s for s, i in rel2id.items()}

    triple_set_train = set(triples_train)
    triple_set_all = set(triples_train) | set(triples_valid) | set(triples_test)

    # ─── Step 4: build the message-passing edge list over the train graph ──
    edge_index, edge_type = build_edge_index(triples_train)

    return {
        "ent2id": ent2id,
        "rel2id": rel2id,
        "id2ent": id2ent,
        "id2rel": id2rel,
        "triples_train": triples_train,
        "triples_valid": triples_valid,
        "triples_test": triples_test,
        "triple_set_train": triple_set_train,
        "triple_set_all": triple_set_all,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "n_ent": len(ent2id),
        "n_rel": len(rel2id),
    }


def build_edge_index(triples):
    """Build a directed edge list for RGCN message passing from int triples.

    Returns (edge_index, edge_type) as PLAIN PYTHON LISTS so this module stays
    torch/numpy-free (see the module docstring). The encoder tensorises them:

      edge_index : [[src0, src1, ...], [dst0, dst1, ...]]   logical shape [2, E]
      edge_type  : [rel0, rel1, ...]                        logical shape [E]

    One directed edge  h -> t  per triple, labelled with relation r (edge_type
    in [0, n_rel)). INVERSE edges (t -> h) are intentionally NOT added here —
    the NeighbourhoodContextEncoder appends them itself, so it owns the
    num_relations = 2*n_rel modelling decision and loaders stays purely
    structural.
    """
    src = [h for (h, r, t) in triples]
    dst = [t for (h, r, t) in triples]
    rel = [r for (h, r, t) in triples]
    return [src, dst], rel


def _parse_split(file_path, ent2id, rel2id, add_to_vocab):
    """Read one TSV file. Update ent2id/rel2id in place. Return integer triples."""
    triples = []

    if not os.path.exists(file_path):
        return triples  # valid.txt and test.txt are optional in some KGs

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue  # skip blank or malformed lines
            h_str, r_str, t_str = parts

            # First-seen ordering: each new string gets the next free ID.
            if add_to_vocab:
                if h_str not in ent2id:
                    ent2id[h_str] = len(ent2id)
                if t_str not in ent2id:
                    ent2id[t_str] = len(ent2id)
                if r_str not in rel2id:
                    rel2id[r_str] = len(rel2id)

            triples.append((ent2id[h_str], rel2id[r_str], ent2id[t_str]))

    return triples
