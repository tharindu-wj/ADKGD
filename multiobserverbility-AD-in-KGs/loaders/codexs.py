"""Where the CoDEx-S files are. Paths and names only -- no logic."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

NAME = "codex-s"

#: The ONLY thing phase 1 may know about this dataset. Subjects, never
#: attributes: naming "family ties" or "citizenship" here would whisper the
#: schema into norms that must not have seen it. Recorded verbatim in every
#: run file, so each run proves exactly what its phase 1 could see.
CARD = ("An encyclopedic knowledge graph about notable real people, "
        "organisations and places.")

#: the graph as shipped by CoDEx: tab-separated Wikidata ids, three splits.
#: Read by scripts/1_prepare_graph.py only -- tools never touch these.
TRIPLE_SPLITS = [
    DATA / "triples" / "codex-s" / "train.txt",
    DATA / "triples" / "codex-s" / "valid.txt",
    DATA / "triples" / "codex-s" / "test.txt",
]

#: definitions -- what every id MEANS. Labels, descriptions, types.
#: Safe for tools to read: these files contain no triples.
ENTITY_DEFINITIONS = DATA / "entities" / "en" / "entities.json"
RELATION_DEFINITIONS = DATA / "relations" / "en" / "relations.json"
ENTITY_TYPES = DATA / "types" / "entity2types.json"
TYPE_DEFINITIONS = DATA / "types" / "en" / "types.json"

#: written by scripts/1_prepare_graph.py -- the ONLY triple file tools read
KG = ROOT / "prepared" / "kg.tsv"

#: where scripts/2_run_root.py records its runs
RUNS = ROOT / "runs"
