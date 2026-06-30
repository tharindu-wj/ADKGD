"""Generic KG loader for KGSAGE.

Loads any TSV-format KG that has train.txt / valid.txt / test.txt in a single
directory. The loader is format-agnostic — works for FB15K-237, WN18RR,
NELL-995, and any custom dataset in the same format.

It returns integer (h, r, t) triples plus the string<->int vocabulary maps,
which is everything the GAN trainer and inference need.

Vocab strategy: first-seen ordering. Train.txt is loaded first, so its entities
and relations get the lowest IDs. This matches the standard KGE convention and
means valid/test only-entities (if any) get higher IDs.
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
        "n_ent": len(ent2id),
        "n_rel": len(rel2id),
    }


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
