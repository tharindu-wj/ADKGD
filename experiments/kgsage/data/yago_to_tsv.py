"""Convert YAGO 4.5 Turtle into KGSAGE-format train/valid/test.txt.

Reads YAGO 4.5 (the `-tiny` Wikipedia-aligned release is the intended input) and
emits the tab-separated  head <TAB> relation <TAB> tail  files that load_kg()
expects. Pure standard library -- no torch, no rdflib -- so it runs with base
Python straight after you unzip the download.

WHAT IT KEEPS
    YAGO 4.5 bundles three things: SHACL schema shapes, an rdfs:subClassOf
    taxonomy, and the actual entity facts. The facts follow one clean pattern:

        yago:Subject <TAB> schema:property <TAB> yago:Object <TAB> .

    We keep exactly the entity->entity facts (subject and object are both YAGO
    entities, predicate is a schema.org property) and drop everything else:
    taxonomy, SHACL shapes, LITERAL objects (dates, numbers, coordinates),
    Wikidata references, self-loops, and duplicates. KGSAGE only models
    entity-entity triples, so literal-valued properties (birthDate, area,
    population, ...) are intentionally excluded.

NAMESPACES + SCHEMA ARE DERIVED FROM THE FILE (not hardcoded)
    1. PREFIXES come from the file's own `@prefix name: <url> .` block. The
       entity and predicate namespaces are matched by their canonical URLs
       (yago-knowledge.org/resource/ and schema.org), so a release that renames
       the `yago:` / `schema:` prefix strings still works.
    2. FACT PROPERTIES come from the file's embedded SHACL `sh:path schema:X`
       declarations -- YAGO's authoritative property list. Kept relations are
       validated against it (catches allowlist drift).

SUBSAMPLING (why it matters -- READ THIS)
    Full YAGO is millions of entities. KGSAGE builds O(n_ent) structures --
    the membership sketch is [n_ent, 8192] bytes (~8 GB per million entities)
    and pool_masks (the per-relation type pools) is [2, n_rel, n_ent] -- so the
    raw graph will NOT fit. You must shrink YAGO to FB/WN scale (~15k-50k
    entities). Three knobs, applied in order:

      --relations r1 r2 ...   keep only these schema properties (focus the KG)
      --min_degree K          k-core: drop entities with < K neighbours,
                              repeatedly, until stable (densifies + shrinks)
      --max_entities N        if still too big, BFS from the highest-degree
                              entities to extract a connected N-entity subgraph

    A good first-run recipe for the -tiny release:
        --min_degree 5 --max_entities 30000
    which yields a dense, WN18RR-scale KG that the NeighbourhoodContextEncoder
    and the membership sketches handle.

USAGE
    python experiments/kgsage/data/yago_to_tsv.py \
        --in  data/YAGO-4.5.0.2-tiny.zip \
        --out data/YAGO4.5 \
        --min_degree 5 --max_entities 30000

  --in accepts a .ttl file, a .zip (reads its .ttl members), or a directory.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import zipfile
from collections import defaultdict, deque

# A Turtle term: a full <IRI>, a "quoted literal"(@lang|^^type)?, or a bare
# token (prefixed name / number / boolean). Quote-aware so a literal containing
# spaces or ';' is not mis-split. Tabs are whitespace, so tab-separated facts
# tokenise correctly.
_TERM = re.compile(r'<[^>]+>|"(?:[^"\\]|\\.)*"(?:@[\w-]+|\^\^\S+)?|[^\s;,]+')

# @prefix name: <url> .   (name may be empty, may contain '-')
_PREFIX = re.compile(r'@prefix\s+([\w-]*):\s*<([^>]+)>')

# sh:path <predicate>  -- declares a schema fact property inside a SHACL shape
_SH_PATH = re.compile(r'\bsh:path\s+([\w-]+):([\w-]+)')

# Canonical namespace URLs (stable across YAGO 4.x). The allowlist is anchored on
# these URLs, not on prefix strings, so it survives a prefix rename. The prefix
# NAMES that map to them are resolved from the input file at runtime.
ENTITY_NS_URL = "http://yago-knowledge.org/resource/"
PREDICATE_NS_URL = "http://schema.org/"


def _parse_statement(stmt):
    """Yield (subject, predicate, object) raw-term triples from one statement.

    Handles the ';' subject-shared form and the ',' object-list form within a
    single statement:  S P1 O1 ; P2 O2a , O2b
    """
    sections = stmt.split(";")
    head_terms = _TERM.findall(sections[0])
    if len(head_terms) < 3:
        return
    subject, predicate = head_terms[0], head_terms[1]
    for obj in head_terms[2:]:
        yield subject, predicate, obj
    for section in sections[1:]:
        terms = _TERM.findall(section)
        if len(terms) < 2:
            continue
        predicate = terms[0]
        for obj in terms[1:]:
            yield subject, predicate, obj


def _local_name(term):
    """schema:birthPlace -> birthPlace ; <http://.../Ulm> -> Ulm."""
    if term.startswith("<") and term.endswith(">"):
        inner = term[1:-1]
        for separator in ("#", "/"):
            if separator in inner:
                inner = inner.rsplit(separator, 1)[-1]
        return inner
    if ":" in term:
        return term.split(":", 1)[1]
    return term


def _in_namespace(term, ns_url, prefix_names):
    """True if term is in the given namespace, by full-IRI URL or by prefix name.

    prefix_names is the set of prefix strings (from the file's @prefix block)
    that map to ns_url -- so both  <http://schema.org/birthPlace>  and
    schema:birthPlace  are recognised, and a prefix rename is handled
    automatically.
    """
    if term.startswith("<") and term.endswith(">"):
        return term[1:-1].startswith(ns_url)
    if ":" in term:
        return term.split(":", 1)[0] in prefix_names
    return False


def _ttl_members(in_path):
    """Yield (name, text_stream) for each .ttl source under in_path.

    Supports a .zip, a single .ttl file, or a directory of .ttl files. Skips
    members named *schema* / *taxonomy* (the YAGO -tiny ships one bundled file).
    """
    def wanted(name):
        low = name.lower()
        return (low.endswith(".ttl")
                and "schema" not in low and "taxonomy" not in low)

    if in_path.lower().endswith(".zip"):
        archive = zipfile.ZipFile(in_path)
        for member in archive.namelist():
            if wanted(member):
                yield member, io.TextIOWrapper(archive.open(member),
                                               encoding="utf-8")
    elif os.path.isdir(in_path):
        for filename in sorted(os.listdir(in_path)):
            if wanted(filename):
                yield filename, open(os.path.join(in_path, filename),
                                     encoding="utf-8")
    else:
        yield os.path.basename(in_path), open(in_path, encoding="utf-8")


def _read_entity_entity_triples(in_path, keep_prefix, stats):
    """Parse the YAGO Turtle and return (triples, schema_props, ns_info).

    triples      : set of (head, relation, tail) readable-name entity triples
    schema_props : set of fact-property local names declared via SHACL sh:path
    ns_info      : (entity_prefixes, predicate_prefixes) resolved from @prefix
    """
    triples = set()
    prefixes = {}                 # prefix name -> namespace url (from @prefix)
    schema_props = set()          # local names declared via sh:path
    entity_prefixes = set()
    predicate_prefixes = set()

    def render(term):
        return term if keep_prefix else _local_name(term)

    def refresh_prefix_sets():
        entity_prefixes.clear()
        predicate_prefixes.clear()
        for name, url in prefixes.items():
            if url == ENTITY_NS_URL:
                entity_prefixes.add(name)
            elif url == PREDICATE_NS_URL:
                predicate_prefixes.add(name)

    for name, stream in _ttl_members(in_path):
        stats["files"] += 1
        print(f"  reading {name} ...", flush=True)
        with stream:
            for raw in stream:
                stats["lines"] += 1
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # 1. collect @prefix declarations (all precede the facts)
                if line.startswith("@prefix") or line.lower().startswith("prefix "):
                    match = _PREFIX.search(line)
                    if match:
                        prefixes[match.group(1)] = match.group(2)
                        refresh_prefix_sets()
                    continue
                if line.startswith("@"):
                    continue
                # 2. collect authoritative fact-property declarations from SHACL
                sh = _SH_PATH.search(line)
                if sh and sh.group(1) in predicate_prefixes:
                    schema_props.add(sh.group(2))
                # 3. facts are single-line statements ending in '.'
                if not line.endswith("."):
                    continue
                for subj, pred, obj in _parse_statement(line[:-1]):
                    stats["raw_triples"] += 1
                    if not (_in_namespace(subj, ENTITY_NS_URL, entity_prefixes)
                            and _in_namespace(obj, ENTITY_NS_URL, entity_prefixes)
                            and _in_namespace(pred, PREDICATE_NS_URL,
                                              predicate_prefixes)):
                        stats["dropped_non_fact"] += 1
                        continue
                    head, relation, tail = render(subj), render(pred), render(obj)
                    if head == tail:
                        stats["dropped_selfloop"] += 1
                        continue
                    triples.add((head, relation, tail))

    return triples, schema_props, (entity_prefixes, predicate_prefixes)


def _k_core(triples, min_degree):
    """Keep only entities that survive iterative degree>=min_degree pruning.

    Standard O(V+E) k-core peeling on the UNDIRECTED entity graph: an entity
    with fewer than min_degree distinct neighbours is removed, which may drop a
    neighbour below the threshold too, and so on. Returns the filtered triples.
    """
    if min_degree <= 0:
        return triples
    adjacency = defaultdict(set)
    for head, _, tail in triples:
        adjacency[head].add(tail)
        adjacency[tail].add(head)
    degree = {e: len(neighbours) for e, neighbours in adjacency.items()}
    queue = deque(e for e, d in degree.items() if d < min_degree)
    removed = set()
    while queue:
        entity = queue.popleft()
        if entity in removed:
            continue
        removed.add(entity)
        for neighbour in adjacency[entity]:
            if neighbour not in removed:
                degree[neighbour] -= 1
                if degree[neighbour] < min_degree:
                    queue.append(neighbour)
    return [(h, r, t) for (h, r, t) in triples
            if h not in removed and t not in removed]


def _cap_entities(triples, max_entities):
    """Extract a connected subgraph of at most max_entities entities.

    BFS outward from the highest-degree entities (so the kept subgraph is dense
    and connected, which the Phase-1 RGCN message passing needs). Returns the
    induced triples.
    """
    entities = {e for h, r, t in triples for e in (h, t)}
    if max_entities <= 0 or len(entities) <= max_entities:
        return triples
    adjacency = defaultdict(set)
    for head, _, tail in triples:
        adjacency[head].add(tail)
        adjacency[tail].add(head)
    seeds = sorted(entities, key=lambda e: -len(adjacency[e]))
    keep = set()
    frontier = deque()
    for seed in seeds:
        if len(keep) >= max_entities:
            break
        if seed in keep:
            continue
        frontier.append(seed)
        while frontier and len(keep) < max_entities:
            entity = frontier.popleft()
            if entity in keep:
                continue
            keep.add(entity)
            for neighbour in adjacency[entity]:
                if neighbour not in keep:
                    frontier.append(neighbour)
    return [(h, r, t) for (h, r, t) in triples if h in keep and t in keep]


def convert(in_path, out_dir, keep_prefix=False, seed=0,
            relations=None, min_degree=0, max_entities=0):
    stats = {"files": 0, "lines": 0, "raw_triples": 0,
             "dropped_non_fact": 0, "dropped_selfloop": 0}

    # ── 1. parse the Turtle into entity-entity triples ──
    triples, schema_props, (entity_prefixes, predicate_prefixes) = \
        _read_entity_entity_triples(in_path, keep_prefix, stats)

    if not entity_prefixes or not predicate_prefixes:
        print()
        print("  WARNING: could not resolve the entity/predicate namespaces from")
        print(f"  the @prefix block. Expected URLs:\n    {ENTITY_NS_URL}"
              f"\n    {PREDICATE_NS_URL}")
        print("  Found prefixes:", dict(list(range(0))))

    n_parsed = len(triples)

    # ── 2. subsampling to KGSAGE scale (relation whitelist -> k-core -> cap) ──
    if relations:
        wanted = set(relations)
        triples = [(h, r, t) for (h, r, t) in triples
                   if (r if not keep_prefix else _local_name(r)) in wanted]
    n_after_rel = len(triples)

    triples = _k_core(triples, min_degree)
    n_after_kcore = len(triples)

    triples = _cap_entities(triples, max_entities)
    n_after_cap = len(triples)

    # ── 3. 90/5/5 split with vocab completeness ──
    # sorted() before shuffle makes the split reproducible: a Python set's
    # iteration order varies across processes (randomised string hashing), so
    # sorting first pins the split to (seed) alone -- important for a citable
    # dataset.
    import random
    triples = sorted(set(triples))
    random.Random(seed).shuffle(triples)
    n = len(triples)
    n_train = int(0.9 * n)
    train = triples[:n_train]
    rest = triples[n_train:]

    train_ents = {h for h, r, t in train} | {t for h, r, t in train}
    train_rels = {r for h, r, t in train}
    keep_rest = [(h, r, t) for (h, r, t) in rest
                 if h in train_ents and t in train_ents and r in train_rels]
    dropped_oov = len(rest) - len(keep_rest)
    mid = len(keep_rest) // 2
    valid, test = keep_rest[:mid], keep_rest[mid:]

    # ── 4. write ──
    os.makedirs(out_dir, exist_ok=True)
    for split_name, rows in (("train", train), ("valid", valid), ("test", test)):
        with open(os.path.join(out_dir, f"{split_name}.txt"), "w",
                  encoding="utf-8") as f:
            for h, r, t in rows:
                f.write(f"{h}\t{r}\t{t}\n")

    all_ents = train_ents | {e for h, r, t in valid + test for e in (h, t)}

    # ── 5. schema validation: kept relations should be schema-declared ──
    kept_local = (train_rels if not keep_prefix
                  else {_local_name(r) for r in train_rels})
    unexpected = kept_local - schema_props

    print()
    print("=" * 64)
    print("  YAGO -> KGSAGE conversion summary")
    print("=" * 64)
    print(f"  source files read        : {stats['files']}")
    print(f"  lines scanned            : {stats['lines']:,}")
    print(f"  raw triples seen         : {stats['raw_triples']:,}")
    print(f"    dropped (not a fact)   : {stats['dropped_non_fact']:,}")
    print(f"    dropped (self-loop)    : {stats['dropped_selfloop']:,}")
    print(f"  unique entity-entity     : {n_parsed:,}")
    if relations:
        print(f"    after relation filter  : {n_after_rel:,}")
    if min_degree:
        print(f"    after k-core (>={min_degree})      : {n_after_kcore:,}")
    if max_entities:
        print(f"    after entity cap ({max_entities}) : {n_after_cap:,}")
    print(f"    dropped (oov in v/t)   : {dropped_oov:,}")
    print(f"  entities / relations     : {len(all_ents):,} / {len(train_rels)}")
    print(f"  split  train/valid/test  : {len(train):,} / {len(valid):,} / {len(test):,}")
    print()
    print(f"  entity prefixes    : {sorted(entity_prefixes)} -> {ENTITY_NS_URL}")
    print(f"  predicate prefixes : {sorted(predicate_prefixes)} -> {PREDICATE_NS_URL}")
    print(f"  schema fact-properties declared (sh:path) : {len(schema_props)}")
    if unexpected:
        print(f"  !! {len(unexpected)} kept relation(s) NOT declared in the schema:")
        for r in sorted(unexpected)[:15]:
            print(f"       - {r}")
    print()
    print(f"  written to               : {out_dir}/")
    if len(train_rels) < 3 or len(train) < 500:
        print()
        print("  WARNING: very few relations/triples survived. Loosen "
              "--min_degree / --max_entities, or check the @prefix block above.")
    return stats


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True,
                    help="YAGO .ttl file, a .zip, or a directory of .ttl files")
    ap.add_argument("--out", required=True,
                    help="output directory for train/valid/test.txt")
    ap.add_argument("--keep-prefix", action="store_true",
                    help="keep fully-qualified pfx:local names instead of "
                         "shortening to local names")
    ap.add_argument("--relations", nargs="*", default=None,
                    help="keep only these schema properties (local names, e.g. "
                         "nationality birthPlace spouse memberOf author)")
    ap.add_argument("--min_degree", type=int, default=0,
                    help="k-core: drop entities with fewer than this many "
                         "neighbours (0 = off). Densifies + shrinks the graph.")
    ap.add_argument("--max_entities", type=int, default=0,
                    help="cap the graph to a connected subgraph of at most this "
                         "many entities (0 = off). Applied after k-core.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    convert(args.in_path, args.out, keep_prefix=args.keep_prefix, seed=args.seed,
            relations=args.relations, min_degree=args.min_degree,
            max_entities=args.max_entities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
