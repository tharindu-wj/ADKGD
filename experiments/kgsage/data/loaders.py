"""Generic KG loader for KGSAGE.

Loads any TSV-format KG that has train.txt / valid.txt / test.txt in a single
directory. The loader is format-agnostic — works for FB15K-237, WN18RR,
NELL-995, and any custom dataset in the same format.

Two output formats from the same data:

  1. PyTorch tensors (edge_index, edge_type) — used by RGCN message passing.
     edge_index has shape (2, num_edges): row 0 = head IDs, row 1 = tail IDs.
     edge_type has shape (num_edges,)    : relation ID per edge.
     This is the format `torch_geometric.nn.RGCNConv` expects.

  2. Python triples (h, r, t) — used by everything else (dataset audit,
     link prediction evaluation, etc.).

We deliberately keep this loader separate from the simple GAN's `experiments/gan/data.py`
because the KGSAGE pipeline has different needs:
  - PyG needs tensors built up front, not per-batch
  - We need head/tail counts per relation for the dataset-level audit
  - We need a clean train/valid/test split (the simple GAN merges all splits)

Vocab strategy: first-seen ordering. Train.txt is loaded first, so its entities
and relations get the lowest IDs. This matches the standard KGE convention and
means valid/test only-entities (if any) get higher IDs.
"""
import os

import torch


def load_kg(data_dir):
    """Load a KG from `data_dir/{train,valid,test}.txt`.

    Each line in those files is three tab-separated strings:
        head_string<TAB>relation_string<TAB>tail_string

    Returns a dict with everything Phase 1 needs:
      ent2id, rel2id     : string -> int (vocab; train-first ordering)
      id2ent, id2rel     : int -> string (inverse maps)
      triples_train      : list of (h, r, t) integer tuples — used by RGCN
      triples_valid      : list of (h, r, t) integer tuples — used by validation MRR
      triples_test       : list of (h, r, t) integer tuples — used by Test 1.1
      triple_set_train   : set of train triples (for negative sampling collision check)
      triple_set_all     : set of train+valid+test triples (for filtered MRR)
      edge_index         : (2, num_train_edges) torch.LongTensor — for RGCN
      edge_type          : (num_train_edges,)   torch.LongTensor — for RGCN
      n_ent, n_rel       : vocabulary sizes

    Notes:
      - Only TRAIN triples go into edge_index/edge_type. Valid/test triples
        must never leak into the encoder's message-passing graph.
      - We add no inverse edges. RGCN treats r as a directed edge type.
        Adding inverse edges would defeat the whole point of Phase 1 — we
        WANT to test whether the model learns anti-symmetric structure
        from a directed graph.
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

    # ─── Step 4: build edge_index / edge_type for RGCN ─────────────────
    # PyG convention: edge_index is a (2, E) LongTensor; row 0 is source
    # node, row 1 is destination node. edge_type is (E,) telling RGCN
    # which relation each edge represents.
    heads = [h for (h, r, t) in triples_train]
    tails = [t for (h, r, t) in triples_train]
    rels = [r for (h, r, t) in triples_train]

    edge_index = torch.tensor([heads, tails], dtype=torch.long)
    edge_type = torch.tensor(rels, dtype=torch.long)

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
            # add_to_vocab is always True in current usage, but the flag
            # is here for a future strict-vocab variant if we ever need it.
            if add_to_vocab:
                if h_str not in ent2id:
                    ent2id[h_str] = len(ent2id)
                if t_str not in ent2id:
                    ent2id[t_str] = len(ent2id)
                if r_str not in rel2id:
                    rel2id[r_str] = len(rel2id)

            triples.append((ent2id[h_str], rel2id[r_str], ent2id[t_str]))

    return triples
