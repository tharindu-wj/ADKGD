"""The definitions store: what every id in the graph MEANS, loaded once.

CoDEx stores triples as Wikidata ids (Q7604, P26). The ids are exact but
unreadable; the labels ("Leonhard Euler", "spouse") are what an agent can
reason about. This module loads the four definition files a single time,
keeps only the entities that actually appear in the prepared graph
(2,034 of 77,951), and translates between the two languages:

    ids     -- used in files and run records, because they are exact
    labels  -- used by every tool and agent, because they mean something

Labels are unique inside the CoDEx-S vocabulary (verified 26 Aug 2026,
see DESIGN.md), so translating a label back to its id is unambiguous.
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

        #: entity id -> its type LABELS ("human", "city"), not type ids
        self.entity_types = {}
        for entity_id in entity_ids:
            labels = []
            for type_id in types_of_entity.get(entity_id, []):
                labels.append(all_types.get(type_id, {}).get("label") or type_id)
            self.entity_types[entity_id] = labels

        # Reverse maps, case-insensitive. Safe because labels are unique here.
        self._entity_id_by_label = {
            info["label"].lower(): eid for eid, info in self.entities.items()}
        self._relation_id_by_label = {
            info["label"].lower(): rid for rid, info in self.relations.items()}

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
        """One triple as readable text: 'Leonhard Euler --spouse-- ...'."""
        head, relation, tail = triple
        return (f"{self.entity_label(head)} "
                f"--{self.relation_label(relation)}-- "
                f"{self.entity_label(tail)}")

    def all_relation_labels(self):
        """Every relation's label, sorted -- for error messages and menus."""
        return sorted(info["label"] for info in self.relations.values())


def _read_json(path):
    if not path.exists():
        raise SystemExit(f"missing {path} -- the CoDEx data folder is incomplete.")
    return json.loads(path.read_text(encoding="utf-8"))


#: loaded on first use, then shared -- parsing 11 MB of JSON once is enough
_shared = None


def get_context():
    global _shared
    if _shared is None:
        _shared = DatasetContext()
    return _shared
