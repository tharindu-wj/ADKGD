"""Abstract base class for KB-family adapters.

A family adapter extracts ENTITY TYPE INFORMATION from a KG that belongs
to a specific knowledge-base family (Freebase, WordNet, YAGO, ...). The
adapter knows the family's conventions - Freebase relation paths, WordNet
synset offsets, YAGO rdf:type triples - and produces a STANDARDIZED OUTPUT
that downstream code consumes identically across families.

Each adapter produces two files per dataset:

  STEP 1: entity_types.tsv
          (NTriples-style: <entity>\\trdf:type\\t<type>; one row per
           (entity, type) pair. Multi-typed entities = multiple rows.
           UTF-8, LF line endings, no header.)

  STEP 2: entity_types_metadata.json
          (Provenance: source KG, adapter version, vocabulary, statistics.
           Useful for reviewers and reproducibility hashing.)

Subclasses implement:
  - supports(triples_path)        does this adapter recognise the format?
  - extract_types(triples_path)   family-specific parsing logic.
  - metadata()                    adapter-specific provenance fields.

The base class provides:
  - write_types_file()            standard NTriples-style TSV writer.
  - write_metadata_file()         standard JSON writer with shared schema.
"""
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from hashlib import sha256
import json


class BaseKBAdapter(ABC):
    """Contract every KB-family adapter implements.

    Attributes:
      family_name:       Human-readable family label (e.g., "Freebase").
      family_id_pattern: Regex describing the entity-ID convention; used
                         by supports() heuristics. Subclasses override.
      version:           Bump on breaking changes to extraction logic.
    """

    family_name: str = "Unknown"
    family_id_pattern: str = r".*"
    version: str = "1.0"

    @abstractmethod
    def supports(self, triples_path) -> bool:
        """Return True if this adapter can handle the input file format.

        Subclasses use heuristics (entity-ID regex, relation prefix style)
        on a sample of lines. Used by auto-detection helpers; explicit
        get_adapter("freebase") skips this check.
        """

    @abstractmethod
    def extract_types(self, triples_path, **kwargs):
        """Return mapping {entity_id: set of type strings}.

        kwargs are adapter-specific (e.g., granularity for WordNet).
        Subclasses document supported kwargs in their own docstrings.
        """

    @abstractmethod
    def metadata(self) -> dict:
        """Return adapter-specific provenance fields.

        Used by write_metadata_file() to fill the "adapter" section of
        entity_types_metadata.json.
        """

    def write_types_file(self, output_path, entity_to_types):
        """Write NTriples-style TSV: <entity>\\trdf:type\\t<type> per row.

        Output is sorted - same input always produces the same file (handy
        for git diffs and reproducibility hashing).
        """
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            for entity in sorted(entity_to_types):
                for type_label in sorted(entity_to_types[entity]):
                    f.write(f"{entity}\trdf:type\t{type_label}\n")

    def write_metadata_file(self, output_path, dataset_name,
                            source_kg_files, entity_to_types,
                            source_kg_citation=None,
                            type_namespace="Unknown"):
        """Write entity_types_metadata.json with the shared schema.

        Args:
          output_path:        where to write the .json file.
          dataset_name:       human-friendly name, e.g., "FB15K-237".
          source_kg_files:    {logical_name: path_on_disk} - we'll hash
                              and line-count each. Example:
                                {"train.txt": "data/FB15K-237/train.txt"}
          entity_to_types:    output of extract_types(); used for stats.
          source_kg_citation: human-readable citation for the source KG
                              (paper title or similar). Falls back to
                              dataset_name if not provided.
          type_namespace:     short description of where the types come
                              from, e.g., "Freebase top-level domains".
        """
        # Aggregate statistics - useful for reviewers and quick sanity checks.
        n_entities = len(entity_to_types)
        n_typed = sum(1 for types in entity_to_types.values() if types)
        multi_typed = sum(1 for types in entity_to_types.values() if len(types) > 1)
        total_assignments = sum(len(types) for types in entity_to_types.values())
        avg_types = total_assignments / max(n_entities, 1)
        all_types = sorted({t for types in entity_to_types.values() for t in types})

        # File hashes - reviewers can verify they have the right source files.
        source_kg_files_with_hash = {}
        for logical_name, path in source_kg_files.items():
            source_kg_files_with_hash[logical_name] = {
                "lines": _count_lines(path),
                "sha256": _sha256_file(path),
            }

        metadata = {
            "schema_version": 1,
            "dataset_name": dataset_name,
            "dataset_family": self.family_name,
            "source_kg": source_kg_citation or dataset_name,
            "source_kg_files": source_kg_files_with_hash,
            "adapter": {
                "name": self.__class__.__name__,
                "version": self.version,
                **self.metadata(),
            },
            "type_vocabulary": {
                "size": len(all_types),
                "namespace": type_namespace,
                "examples": all_types[:10],
            },
            "statistics": {
                "n_entities": n_entities,
                "n_typed_entities": n_typed,
                "coverage_fraction": n_typed / max(n_entities, 1),
                "avg_types_per_entity": avg_types,
                "multi_typed_count": multi_typed,
            },
            "produced_by": "kg_corrupter v0.1",
            "produced_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=False)
            f.write("\n")


def _count_lines(path) -> int:
    """Count lines in a file. Streams to avoid loading the whole file."""
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _sha256_file(path) -> str:
    """SHA-256 of a file's bytes, computed in chunks for memory safety."""
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
