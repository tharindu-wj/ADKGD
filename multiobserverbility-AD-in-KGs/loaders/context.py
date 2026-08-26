"""The definitions store: what every id in the graph MEANS, loaded once.

A dataset stores triples as opaque ids -- exact but unreadable. The labels
its definition files ship are what an agent can reason about. This module
loads the definition files a single time, keeps only the entities that
actually appear in the prepared graph, and translates between the two
languages:

    ids     -- used in files and run records, because they are exact
    labels  -- used by every tool and agent, because they mean something

Tools accept labels as INPUT, which requires labels to be unique. That is a
property of a dataset, not of this code -- so it is CHECKED at load, and a
dataset with colliding labels fails loudly here instead of resolving
ambiguously somewhere downstream.
"""
import json

from loaders import graph
from loaders.active import DATASET


class DatasetContext:
    """Triples plus the meaning of every id in them."""

    def __init__(self):
        if not DATASET.KG.exists():
            raise SystemExit(
                f"missing {DATASET.KG}. Run scripts/1_prepare_graph.py first.")

        self.triples = graph.load_triples(DATASET.KG)

        entity_ids = {e for h, r, t in self.triples for e in (h, t)}
        relation_ids = {r for h, r, t in self.triples}

        all_entities = _read_json(DATASET.ENTITY_DEFINITIONS)
        all_relations = _read_json(DATASET.RELATION_DEFINITIONS)
        types_of_entity = _read_json(DATASET.ENTITY_TYPES)
        all_types = _read_json(DATASET.TYPE_DEFINITIONS)

        #: entity id -> {"label", "description"} for OUR entities only
        self.entities = {}
        for entity_id in entity_ids:
            record = all_entities.get(entity_id, {})
            self.entities[entity_id] = {
                "label": record.get("label") or entity_id,
                "description": record.get("description") or "",
            }

        #: relation id -> {"label", "description"}
        self.relations = {}
        for relation_id in relation_ids:
            record = all_relations.get(relation_id, {})
            self.relations[relation_id] = {
                "label": record.get("label") or relation_id,
                "description": record.get("description") or "",
            }

        #: entity id -> its type LABELS (readable words), not type ids
        self.entity_types = {}
        for entity_id in entity_ids:
            labels = []
            for type_id in types_of_entity.get(entity_id, []):
                labels.append(all_types.get(type_id, {}).get("label") or type_id)
            self.entity_types[entity_id] = labels

        # Reverse maps, case-insensitive. Uniqueness is checked, not assumed.
        self._entity_id_by_label = _reverse_map(self.entities, "entity")
        self._relation_id_by_label = _reverse_map(self.relations, "relation")

    # ---- translating -----------------------------------------------------

    def entity_label(self, entity_id):
        return self.entities.get(entity_id, {}).get("label", entity_id)

    def relation_label(self, relation_id):
        return self.relations.get(relation_id, {}).get("label", relation_id)

    def find_entity(self, term):
        """Entity id for a label or id. None if unknown."""
        term = (term or "").strip()
        if term in self.entities:
            return term
        return self._entity_id_by_label.get(term.lower())

    def find_relation(self, term):
        """Relation id for a label or id. None if unknown."""
        term = (term or "").strip()
        if term in self.relations:
            return term
        return self._relation_id_by_label.get(term.lower())

    def triple_text(self, triple):
        """One triple as readable text: '<head> --<relation>-- <tail>'."""
        head, relation, tail = triple
        return (f"{self.entity_label(head)} "
                f"--{self.relation_label(relation)}-- "
                f"{self.entity_label(tail)}")

    def all_relation_labels(self):
        """Every relation's label, sorted -- for error messages and menus."""
        return sorted(info["label"] for info in self.relations.values())


def _reverse_map(items, what):
    """label (lowercased) -> id, refusing a dataset whose labels collide."""
    by_label = {}
    collisions = set()
    for item_id, info in items.items():
        key = info["label"].lower()
        if key in by_label:
            collisions.add(info["label"])
        by_label[key] = item_id
    if collisions:
        shown = ", ".join(sorted(collisions)[:5])
        raise SystemExit(
            f"{len(collisions)} {what} labels collide in this dataset "
            f"(e.g. {shown}). Tools accept labels as input, which needs them "
            f"unique -- disambiguate the labels in the dataset's definition "
            f"files before running.")
    return by_label


def _read_json(path):
    if not path.exists():
        raise SystemExit(f"missing {path} -- the dataset's definition files "
                         "are incomplete.")
    return json.loads(path.read_text(encoding="utf-8"))


#: loaded on first use, then shared -- parsing 11 MB of JSON once is enough
_shared = None


def get_context():
    global _shared
    if _shared is None:
        _shared = DatasetContext()
    return _shared
