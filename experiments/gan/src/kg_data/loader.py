"""Loader for KG datasets that follow the FB15k-237 file layout.

The loader takes a directory like::

    data/dummy_kg/
        train.txt              REQUIRED - tab-separated head, relation, tail
        valid.txt              OPTIONAL - validation split (may be empty)
        test.txt               OPTIONAL - test split (may be empty)
        entity_metadata.txt    OPTIONAL - entity_id <TAB> display_name <TAB> type
        relation_metadata.txt  OPTIONAL - relation_id <TAB> display_name

and returns a `KnowledgeGraph` object that contains everything downstream
code needs: vocabularies, integer-indexed triples, pretty-print helpers.

There is intentionally no adjacency tensor in this object. Callers that need
adjacency build it on demand from the triples via
`kg_data.adjacency.triples_to_adjacency_tensor`.

This format is FB15k-237-compatible: the same loader works on the dummy KG
and (later) on the full FB15k-237 dataset, with no code change at the call
site - just point at a different directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


# (head_index, relation_index, tail_index) - the canonical "triple" form
Triple = Tuple[int, int, int]


@dataclass
class KnowledgeGraph:
    """A loaded knowledge graph plus all the metadata downstream code uses.

    The dataclass holds TWO representations of the same triples:
      * `triples`     - the raw form from the TSV file (strings like
                        "/dummy/Alice", "/dummy/born_in"). Mostly for
                        debugging and round-tripping back to disk.
      * `triples_idx` - the integer-indexed form. This is what every
                        downstream piece of code (TRIC, dataset builder,
                        the model) actually consumes.

    The vocab maps (`entity_id_to_row`, `relation_id_to_channel`) are the
    bridge between those two forms.
    """
    # Raw string triples - original form from the TSV files
    triples:        List[Tuple[str, str, str]]   # training split
    valid_triples:  List[Tuple[str, str, str]]   # validation split (may be [])
    test_triples:   List[Tuple[str, str, str]]   # test split (may be [])

    # Integer-indexed triples - what downstream code consumes
    triples_idx:    List[Triple]
    triple_set_idx: Set[Triple]                  # set form, O(1) "exists?"

    # Vocabulary maps (string -> integer)
    entity_id_to_row:       Dict[str, int]       # e.g. {"/dummy/Alice": 0}
    relation_id_to_channel: Dict[str, int]       # e.g. {"/dummy/born_in": 0}

    # Pretty-print helpers (integer -> human-readable string)
    node_labels:    Dict[int, str]               # row_index -> display name
    node_types:     Dict[int, str]               # row_index -> type string
    relation_names: List[str]                    # channel_index -> display name

    dataset_name:   str = ""

    @property
    def num_nodes(self) -> int:
        """Number of unique entities in the training split."""
        return len(self.entity_id_to_row)

    @property
    def num_relations(self) -> int:
        """Number of unique relations in the training split."""
        return len(self.relation_id_to_channel)


def _read_triples_tsv(path: Path) -> List[Tuple[str, str, str]]:
    """Read tab-separated string triples from a file. Returns [] if missing."""
    if not path.exists():
        return []
    parsed_triples: List[Tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for raw_line in input_file:
            raw_line = raw_line.rstrip("\n")
            if not raw_line.strip():
                continue   # skip blank lines
            columns = raw_line.split("\t")
            if len(columns) != 3:
                raise ValueError(
                    f"{path}: expected 3 tab-separated fields, "
                    f"got {len(columns)} in line: {raw_line!r}"
                )
            parsed_triples.append((columns[0], columns[1], columns[2]))
    return parsed_triples


def _read_metadata_tsv(path: Path, expected_num_columns: int) -> List[List[str]]:
    """Read a tab-separated metadata file (any number of columns). Returns []
    if the file doesn't exist. Pads short rows with empty strings."""
    if not path.exists():
        return []
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for raw_line in input_file:
            raw_line = raw_line.rstrip("\n")
            if not raw_line.strip():
                continue
            columns = raw_line.split("\t")
            while len(columns) < expected_num_columns:
                columns.append("")
            rows.append(columns[:expected_num_columns])
    return rows


def load_kg(dataset_directory: str | Path) -> KnowledgeGraph:
    """Load a KG from a directory of TSV files (FB15k-237-compatible layout).

    Steps:
      1. Read the three triple files (train.txt, valid.txt, test.txt).
      2. Build entity & relation vocabularies in "first-seen" order from the
         training triples.
      3. Read optional metadata files (display names and entity types).
      4. Translate each training triple into integer-indexed form.
      5. Wrap everything in a KnowledgeGraph dataclass and return it.

    Parameters
    ----------
    dataset_directory : str or Path
        Path to a directory containing at least `train.txt`.

    Returns
    -------
    KnowledgeGraph - see the dataclass docstring for what's inside.
    """
    dataset_directory = Path(dataset_directory)
    if not dataset_directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_directory}")

    # ── Step 1: read the three triple split files ─────────────────────
    training_triples_strings   = _read_triples_tsv(dataset_directory / "train.txt")
    validation_triples_strings = _read_triples_tsv(dataset_directory / "valid.txt")
    test_triples_strings       = _read_triples_tsv(dataset_directory / "test.txt")
    if not training_triples_strings:
        raise ValueError(f"No training triples found at {dataset_directory / 'train.txt'}")

    # ── Step 2: build entity & relation vocabularies ──────────────────
    # First-seen ordering: row index 0 is the first entity that appears in
    # the training data, row 1 is the second, and so on. Same for relations.
    entity_ids_in_order:   List[str] = []
    relation_ids_in_order: List[str] = []
    entities_already_seen:  set = set()
    relations_already_seen: set = set()

    for head_string, relation_string, tail_string in training_triples_strings:
        for entity_string in (head_string, tail_string):
            if entity_string not in entities_already_seen:
                entities_already_seen.add(entity_string)
                entity_ids_in_order.append(entity_string)
        if relation_string not in relations_already_seen:
            relations_already_seen.add(relation_string)
            relation_ids_in_order.append(relation_string)

    entity_id_to_row       = {entity_id: row_index
                              for row_index, entity_id in enumerate(entity_ids_in_order)}
    relation_id_to_channel = {relation_id: channel_index
                              for channel_index, relation_id in enumerate(relation_ids_in_order)}

    # ── Step 3: read optional metadata ────────────────────────────────
    entity_metadata_rows = _read_metadata_tsv(
        dataset_directory / "entity_metadata.txt", expected_num_columns=3,
    )
    entity_id_to_display_name: Dict[str, str] = {}
    entity_id_to_type:         Dict[str, str] = {}
    for entity_id, display_name, entity_type in entity_metadata_rows:
        entity_id_to_display_name[entity_id] = display_name or entity_id
        entity_id_to_type[entity_id]         = entity_type or "unknown"

    relation_metadata_rows = _read_metadata_tsv(
        dataset_directory / "relation_metadata.txt", expected_num_columns=2,
    )
    relation_id_to_display_name: Dict[str, str] = {}
    for relation_id, display_name in relation_metadata_rows:
        relation_id_to_display_name[relation_id] = display_name or relation_id

    # Pretty-print helpers, keyed by integer index (the form models use)
    node_labels = {
        row_index: entity_id_to_display_name.get(entity_id, entity_id)
        for entity_id, row_index in entity_id_to_row.items()
    }
    node_types = {
        row_index: entity_id_to_type.get(entity_id, "unknown")
        for entity_id, row_index in entity_id_to_row.items()
    }
    relation_names = [
        relation_id_to_display_name.get(relation_id, relation_id)
        for relation_id in relation_ids_in_order
    ]

    # ── Step 4: integer-indexed triple form ───────────────────────────
    triples_idx: List[Triple] = [
        (
            entity_id_to_row[head_string],
            relation_id_to_channel[relation_string],
            entity_id_to_row[tail_string],
        )
        for head_string, relation_string, tail_string in training_triples_strings
    ]
    # Same content as triples_idx but as a set, for O(1) "exists?" checks.
    triple_set_idx: Set[Triple] = set(triples_idx)

    # ── Step 5: wrap and return ───────────────────────────────────────
    return KnowledgeGraph(
        triples=training_triples_strings,
        valid_triples=validation_triples_strings,
        test_triples=test_triples_strings,
        triples_idx=triples_idx,
        triple_set_idx=triple_set_idx,
        entity_id_to_row=entity_id_to_row,
        relation_id_to_channel=relation_id_to_channel,
        node_labels=node_labels,
        node_types=node_types,
        relation_names=relation_names,
        dataset_name=dataset_directory.name,
    )


def load_kg_union(dataset_directory: str | Path) -> KnowledgeGraph:
    """Load a KG with vocab and triple set built from the UNION of all three splits.

    The standard `load_kg` builds vocabularies and the triple set from
    `train.txt` only, which mirrors how knowledge-graph-completion code
    typically treats the three splits.

    For the ADKGD synthetic-anomaly task we need the opposite: the GAN
    sees one big graph (all 310k FB15k-237 triples), and the "is this
    triple real?" check during anomaly generation must consult the union
    so we never emit a triple that exists in train OR valid OR test.

    Behaviour vs `load_kg`:
      * `triples`        - still the training split (preserved so callers
                           that look at the raw training file still work).
      * `valid_triples`  - validation split (unchanged).
      * `test_triples`   - test split (unchanged).
      * `triples_idx`    - UNION of all three splits, integer-indexed.
      * `triple_set_idx` - set form of the union, for O(1) "exists?" checks.
      * `entity_id_to_row` / `relation_id_to_channel` - built from the
        UNION of strings appearing across all three files.
    """
    dataset_directory = Path(dataset_directory)
    if not dataset_directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_directory}")

    training_triples_strings   = _read_triples_tsv(dataset_directory / "train.txt")
    validation_triples_strings = _read_triples_tsv(dataset_directory / "valid.txt")
    test_triples_strings       = _read_triples_tsv(dataset_directory / "test.txt")
    if not training_triples_strings:
        raise ValueError(f"No training triples found at {dataset_directory / 'train.txt'}")

    union_triples_strings: List[Tuple[str, str, str]] = (
        list(training_triples_strings)
        + list(validation_triples_strings)
        + list(test_triples_strings)
    )

    # Build vocab from the UNION, first-seen order across train → valid → test.
    entity_ids_in_order:   List[str] = []
    relation_ids_in_order: List[str] = []
    entities_already_seen:  set = set()
    relations_already_seen: set = set()
    for head_string, relation_string, tail_string in union_triples_strings:
        for entity_string in (head_string, tail_string):
            if entity_string not in entities_already_seen:
                entities_already_seen.add(entity_string)
                entity_ids_in_order.append(entity_string)
        if relation_string not in relations_already_seen:
            relations_already_seen.add(relation_string)
            relation_ids_in_order.append(relation_string)

    entity_id_to_row       = {entity_id: row_index
                              for row_index, entity_id in enumerate(entity_ids_in_order)}
    relation_id_to_channel = {relation_id: channel_index
                              for channel_index, relation_id in enumerate(relation_ids_in_order)}

    entity_metadata_rows = _read_metadata_tsv(
        dataset_directory / "entity_metadata.txt", expected_num_columns=3,
    )
    entity_id_to_display_name: Dict[str, str] = {}
    entity_id_to_type:         Dict[str, str] = {}
    for entity_id, display_name, entity_type in entity_metadata_rows:
        entity_id_to_display_name[entity_id] = display_name or entity_id
        entity_id_to_type[entity_id]         = entity_type or "unknown"

    relation_metadata_rows = _read_metadata_tsv(
        dataset_directory / "relation_metadata.txt", expected_num_columns=2,
    )
    relation_id_to_display_name: Dict[str, str] = {}
    for relation_id, display_name in relation_metadata_rows:
        relation_id_to_display_name[relation_id] = display_name or relation_id

    node_labels = {
        row_index: entity_id_to_display_name.get(entity_id, entity_id)
        for entity_id, row_index in entity_id_to_row.items()
    }
    node_types = {
        row_index: entity_id_to_type.get(entity_id, "unknown")
        for entity_id, row_index in entity_id_to_row.items()
    }
    relation_names = [
        relation_id_to_display_name.get(relation_id, relation_id)
        for relation_id in relation_ids_in_order
    ]

    union_triples_idx: List[Triple] = [
        (
            entity_id_to_row[head_string],
            relation_id_to_channel[relation_string],
            entity_id_to_row[tail_string],
        )
        for head_string, relation_string, tail_string in union_triples_strings
    ]
    union_triple_set_idx: Set[Triple] = set(union_triples_idx)

    return KnowledgeGraph(
        triples=training_triples_strings,
        valid_triples=validation_triples_strings,
        test_triples=test_triples_strings,
        triples_idx=union_triples_idx,
        triple_set_idx=union_triple_set_idx,
        entity_id_to_row=entity_id_to_row,
        relation_id_to_channel=relation_id_to_channel,
        node_labels=node_labels,
        node_types=node_types,
        relation_names=relation_names,
        dataset_name=dataset_directory.name,
    )
