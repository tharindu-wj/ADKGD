"""Knowledge graph loader.

Reads a folder of TSV files (train.txt, valid.txt, test.txt) and turns it
into integer-indexed triples plus string->int vocab maps.

A knowledge graph is just a list of (head, relation, tail) facts like:
    Alice  born_in  Australia
    Bob    born_in  UK
    ...

We keep strings on disk (human-readable) but use integer IDs in the GAN
(fast tensor operations). The vocab maps let us go back and forth.
"""
import os


def load_kg(data_dir):
    """Load a KG from `data_dir/{train,valid,test}.txt`.

    Each line in those files is three tab-separated strings:
        head_string<TAB>relation_string<TAB>tail_string

    We merge all three splits into one big triple list because anomaly
    detection treats the whole graph as one corpus.

    Returns a dict with:
      ent2id, rel2id     : string -> int  (vocab; first-seen ordering)
      id2ent, id2rel     : int -> string  (inverse maps)
      triples            : list of (h, r, t) integer tuples
      triple_set         : same as `triples` but a Python set for O(1)
                           "is this real?" checks
      n_ent, n_rel       : vocabulary sizes
    """
    ent2id = {}
    rel2id = {}
    triples = []

    for split_filename in ("train.txt", "valid.txt", "test.txt"):
        file_path = os.path.join(data_dir, split_filename)
        if not os.path.exists(file_path):
            continue  # valid.txt / test.txt are optional

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue  # skip blank or malformed lines
                h_str, r_str, t_str = parts

                # First-seen ordering: each new string gets the next free ID.
                if h_str not in ent2id:
                    ent2id[h_str] = len(ent2id)
                if t_str not in ent2id:
                    ent2id[t_str] = len(ent2id)
                if r_str not in rel2id:
                    rel2id[r_str] = len(rel2id)

                triples.append((ent2id[h_str], rel2id[r_str], ent2id[t_str]))

    # Inverse maps for going back to strings (used at inference time).
    id2ent = {i: s for s, i in ent2id.items()}
    id2rel = {i: s for s, i in rel2id.items()}

    return {
        "ent2id": ent2id,
        "rel2id": rel2id,
        "id2ent": id2ent,
        "id2rel": id2rel,
        "triples": triples,
        "triple_set": set(triples),
        "n_ent": len(ent2id),
        "n_rel": len(rel2id),
    }
