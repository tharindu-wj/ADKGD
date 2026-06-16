"""FreebaseAdapter - extract entity types from Freebase-derived KGs.

In Freebase datasets (FB15K, FB15K-237, FB13, ...), entity IDs are
Freebase MIDs of the form /m/XXXXX (e.g., /m/0d3k14 = Bill Gates).
Type information for each entity is NOT distributed separately - it's
EMBEDDED IN THE RELATION PATHS that link entities.

Sample relations from FB15K-237:
  /people/person/nationality
  /location/country/capital
  /tv/tv_program/regular_cast./tv/regular_tv_appearance/actor   (compound)

The 3-step extraction:

  STEP 1: Parse each relation path -> (head_type, tail_type).
          (Split on "." for compound relations and use the first
           sub-path. head_type = first segment (the domain);
           tail_type = last segment (a property name used as a role
           marker - see note below).)

  STEP 2: For each (h, r, t) in the training set, accumulate types:
            entity_to_types[h] |= {head_type}
            entity_to_types[t] |= {tail_type}
          (Multi-typing is the NORM in Freebase: Bill Gates appears
           as head of /people/* relations AND /organization/* relations,
           ending up tagged with multiple types. The UNION over all
           relations is the entity's full type profile.)

  STEP 3: Emit (entity, type) pairs for every accumulated type.

The tail_type choice deserves a note: Freebase relation paths don't
explicitly encode the tail's CLASS (/people/person/nationality has no
"country" segment). We use the LAST segment as a ROLE MARKER, so
"nationality" becomes a type tag. This is intentional: tailPool[r]
in Phase 1.2 then contains exactly the entities that have ever appeared
as tail of r, which is what Phase 2's candidate-pool builder needs.

Covers: FB15K, FB15K-237, FB13, FB237 - any Freebase subset using
        MID identifiers and slash-path relations.
"""
from collections import defaultdict
import re

from .base import BaseKBAdapter


# Freebase MIDs look like /m/0d3k14, /m/02sjp, /m/abc_1, etc.
# Used by supports() to auto-detect from sample lines.
_MID_PATTERN = re.compile(r"^/m/[a-z0-9_]+$")


class FreebaseAdapter(BaseKBAdapter):
    """Parses Freebase relation prefixes to derive entity types.

    Multi-typing is supported (FB entities are often multi-typed via
    union over their co-occurring relations).
    """

    family_name = "Freebase"
    family_id_pattern = r"^/m/[a-z0-9_]+$"
    version = "1.0"

    def supports(self, triples_path) -> bool:
        """Heuristic: do the first 10 lines' entities look like Freebase MIDs?

        Sample-based detection - we don't need to read the whole file just
        to know whether this adapter applies.
        """
        sample_entities = []
        with open(triples_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    sample_entities.append(parts[0])
                    sample_entities.append(parts[2])
        if not sample_entities:
            return False
        # Require most (>=80%) to match the MID pattern - lets a stray
        # malformed line slip through without breaking detection.
        matches = sum(1 for e in sample_entities if _MID_PATTERN.match(e))
        return matches / len(sample_entities) >= 0.8

    def extract_types(self, triples_path, **kwargs):
        """Return {entity_id: set of types} derived from co-occurring relations.

        Single pass over the file. Memory cost is proportional to
        (n_entities x avg_types_per_entity), which is small for FB-scale.
        """
        entity_to_types = defaultdict(set)

        with open(triples_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                h, r, t = parts[0], parts[1], parts[2]
                head_type, tail_type = self._parse_relation(r)
                entity_to_types[h].add(head_type)
                entity_to_types[t].add(tail_type)

        # Return a plain dict so callers can't accidentally grow it by
        # lookup (defaultdict-on-lookup gotcha).
        return dict(entity_to_types)

    def metadata(self) -> dict:
        """FreebaseAdapter-specific provenance fields."""
        return {
            "extraction_method": (
                "Parse Freebase relation prefixes. "
                "head_type = first segment (domain); "
                "tail_type = last segment (role marker). "
                "Multi-typing via union over co-occurring relations."
            ),
            "extraction_params": {
                "compound_relation_handling": "split-on-dot-take-first-subpath",
                "head_type_source": "first segment of relation path",
                "tail_type_source": "last segment of relation path",
            },
        }

    def _parse_relation(self, relation):
        """Return (head_type, tail_type) for a Freebase relation path.

        Examples:
          /people/person/nationality                    -> ("people", "nationality")
          /location/country/capital                     -> ("location", "capital")
          /tv/tv_program/regular_cast./.../actor        -> ("tv", "actor")

        For compound relations (containing "."), only the FIRST sub-path
        is used. Freebase compounds chain through mediator nodes whose
        own type is less informative; the head's class comes from the
        first sub-path.
        """
        # First sub-path of a compound relation.
        primary = relation.split(".")[0]
        # Strip the leading "/" and split into segments.
        segments = primary.lstrip("/").split("/")
        if not segments:
            return ("unknown", "unknown")
        head_type = segments[0]
        # Last segment acts as a role marker - see module docstring.
        tail_type = segments[-1] if len(segments) > 1 else head_type
        return (head_type, tail_type)
