"""Build concept pools and save them as a pickle cache.

A concept pool tells Phase 2 (training-time candidate sampling) which
entities are TYPE-COMPATIBLE with each relation's head or tail slot.

The 5-step build:

  STEP 1: Iterate the training triples to build the entity and relation
          vocabularies (string -> int IDs).
          (Integer IDs are faster for downstream pool lookups, set
           membership tests, and torch indexing.)

  STEP 2: Compute the dataset hash (sha256 of sorted training triples).
          (Used by Phase 2/3 to detect stale caches when the input KG
           changes.)

  STEP 3: From the adapter's entity_to_types, build an inverse index:
            type_to_entities[type] = {entities of that type}
          (Lets us answer "all entities of type X" in O(1).)

  STEP 4: For each (h, r, t) in training, accumulate the set of HEAD
          types and TAIL types observed for r. Together these form
          C_S(r) - the concept signature of r.

  STEP 5: For each r:
            headPool[r] = union of type_to_entities[t] for t in C_S(r).heads
            tailPool[r] = union of type_to_entities[t] for t in C_S(r).tails
          (Pools are TYPE-AWARE: an entity is in the pool if it shares ANY
           type with the observed heads/tails, not just if it co-occurred
           with the relation in training. This is CGSP's extension over
           plain usage-based sampling.)

Output is a single pickle dict consumed by Phase 2 and Phase 3.
"""
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import pickle


def iter_triples(triples_path):
    """Yield (head, relation, tail) string triples from a tab-separated file."""
    with open(triples_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                yield parts[0], parts[1], parts[2]


def build_vocab(triples_path):
    """Return (entity_to_id, relation_to_id, id_to_entity, id_to_relation).

    Order is deterministic: entities and relations are numbered by FIRST
    APPEARANCE in the file. This keeps IDs stable across runs as long as
    the file's line order doesn't change.
    """
    entity_to_id = {}
    relation_to_id = {}
    for h, r, t in iter_triples(triples_path):
        if h not in entity_to_id:
            entity_to_id[h] = len(entity_to_id)
        if t not in entity_to_id:
            entity_to_id[t] = len(entity_to_id)
        if r not in relation_to_id:
            relation_to_id[r] = len(relation_to_id)
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    return entity_to_id, relation_to_id, id_to_entity, id_to_relation


def compute_dataset_hash(triples_path):
    """SHA-256 of the sorted training triples. Used for cache invalidation."""
    h = sha256()
    lines = []
    with open(triples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if line:
                lines.append(line)
    for line in sorted(lines):
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build_pools(triples_path, entity_to_types):
    """Build the concept-pools state dict.

    Args:
      triples_path:     path to train.txt (TSV format).
      entity_to_types:  {entity_id_string: set of type strings} from an adapter.

    Returns:
      A dict suitable for save_pools(). Cardinality fields are EMPTY here -
      they get filled in by cardinality.classify_cardinality() before saving.
    """
    # STEP 1: build vocab.
    entity_to_id, relation_to_id, id_to_entity, id_to_relation = build_vocab(triples_path)
    n_entities = len(entity_to_id)
    n_relations = len(relation_to_id)

    # STEP 2: dataset hash (for cache invalidation).
    dataset_hash = compute_dataset_hash(triples_path)

    # STEP 3: inverse index type -> entities.
    type_to_entities = defaultdict(set)
    for entity, types in entity_to_types.items():
        if entity in entity_to_id:
            e_id = entity_to_id[entity]
            for t in types:
                type_to_entities[t].add(e_id)

    # STEP 4: observe per-relation head/tail types from training.
    # Also build the real_triple_set used by Phase 3 corruption validation.
    head_types_per_r = defaultdict(set)
    tail_types_per_r = defaultdict(set)
    real_triple_set = set()
    for h, r, t in iter_triples(triples_path):
        r_id = relation_to_id[r]
        head_types_per_r[r_id].update(entity_to_types.get(h, set()))
        tail_types_per_r[r_id].update(entity_to_types.get(t, set()))
        real_triple_set.add((entity_to_id[h], r_id, entity_to_id[t]))

    # STEP 5: build pools by unioning entities of each observed type.
    headPool = {}
    tailPool = {}
    pool_sizes = {}
    for r_id in range(n_relations):
        head_pool = set()
        for t in head_types_per_r[r_id]:
            head_pool.update(type_to_entities[t])
        tail_pool = set()
        for t in tail_types_per_r[r_id]:
            tail_pool.update(type_to_entities[t])
        headPool[r_id] = head_pool
        tailPool[r_id] = tail_pool
        pool_sizes[r_id] = {
            "head": len(head_pool),
            "tail": len(tail_pool),
            "head_frac": len(head_pool) / max(n_entities, 1),
            "tail_frac": len(tail_pool) / max(n_entities, 1),
        }

    return {
        "schema_version": 1,
        "dataset_hash": dataset_hash,
        "n_entities": n_entities,
        "n_relations": n_relations,
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
        "id_to_entity": id_to_entity,
        "id_to_relation": id_to_relation,
        "headPool": headPool,
        "tailPool": tailPool,
        "pool_sizes": pool_sizes,
        "real_triple_set": real_triple_set,
        # Cardinality fields get filled by classify_cardinality() before saving.
        "cardinality": {},
        "cardinality_stats": {},
        # Provenance - the orchestrator fills these too.
        "extraction_method": "schema_based",
        "source_files": {},
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "produced_by": "kg_corrupter v0.1",
    }


def save_pools(state, output_path):
    """Pickle the state dict to disk."""
    with open(output_path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pools(input_path):
    """Load and return the pickled state dict."""
    with open(input_path, "rb") as f:
        return pickle.load(f)
