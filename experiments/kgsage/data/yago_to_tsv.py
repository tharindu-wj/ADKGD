"""Convert YAGO 4.5 Turtle into KGSAGE-format train/valid/test.txt.

Reads YAGO 4.5 (the `-tiny` Wikipedia-aligned release is the intended input) and
emits the tab-separated  head <TAB> relation <TAB> tail  files that load_kg()
expects. Pure standard library -- no torch, no rdflib -- so it runs with base
Python straight after you unzip the download.

YAGO 4.5's single yago-tiny.ttl bundles three things: SHACL schema shapes
(multi-line ';' blocks at the top), an rdfs:subClassOf taxonomy (the bulk), and
the actual entity facts. The facts follow ONE clean pattern --

    yago:Subject <TAB> schema:property <TAB> yago:Object <TAB> .

-- one fact per line. We keep exactly that pattern via a namespace ALLOWLIST and
drop everything else (taxonomy, SHACL shapes, literal objects, wikidata refs,
self-loops, duplicates).

AUTHORITATIVE, VERSION-MATCHED DEFINITIONS (the two robustness upgrades)
  Rather than hardcoding prefix strings or a canonical property list, the
  converter derives both from the input file's OWN declarations -- which YAGO's
  construction pipeline emits authoritatively and which are guaranteed to match
  the exact release you downloaded:

  1. PREFIXES: parsed from the file's `@prefix name: <url> .` block. The entity
     and fact-property namespaces are identified by their canonical URLs
     (yago-knowledge.org/resource/ and schema.org), so the allowlist survives a
     release that renames the `yago:` / `schema:` prefix strings.

  2. SCHEMA PROPERTIES: parsed from the file's embedded SHACL `sh:path schema:X`
     declarations -- YAGO's authoritative list of fact properties. The surviving
     relation set is validated against it: any kept relation NOT declared in the
     schema is flagged (catches allowlist drift).

OUTPUT HYGIENE (KGSAGE data standards)
  - relation + entity strings are readable local names (schema:birthPlace ->
    birthPlace, yago:Albert_Einstein -> Albert_Einstein); --keep-prefix keeps the
    pfx:local form.
  - 90/5/5 split with VOCAB COMPLETENESS: valid/test triples whose entity or
    relation is unseen in train are dropped (DS-10).
  - DS-3 (inverse leakage) is not a concern on YAGO: it stores no inverse
    direction, so there are no (t, r_inv, h) pairs to leak across the split.

USAGE
    python experiments/kgsage/data/yago_to_tsv.py \
        --in  data/YAGO-4.5.0.2-tiny/yago-tiny.ttl \
        --out data/YAGO-4.5-tiny

  --in accepts a .ttl file, a .zip (reads its .ttl members), or a directory.

THEN run the v2-A gate to confirm rule-mined positives work on it:
    python -m kgsage.cli.verify_templates --dataset data/YAGO-4.5-tiny
"""
import argparse
import io
import os
import re
import zipfile

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
    subject, pred = head_terms[0], head_terms[1]
    for obj in head_terms[2:]:
        yield subject, pred, obj
    for sec in sections[1:]:
        terms = _TERM.findall(sec)
        if len(terms) < 2:
            continue
        pred = terms[0]
        for obj in terms[1:]:
            yield subject, pred, obj


def _local_name(term):
    """schema:birthPlace -> birthPlace ; <http://.../Ulm> -> Ulm."""
    if term.startswith("<") and term.endswith(">"):
        inner = term[1:-1]
        for sep in ("#", "/"):
            if sep in inner:
                inner = inner.rsplit(sep, 1)[-1]
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
        return low.endswith(".ttl") and "schema" not in low and "taxonomy" not in low

    if in_path.lower().endswith(".zip"):
        zf = zipfile.ZipFile(in_path)
        for member in zf.namelist():
            if wanted(member):
                yield member, io.TextIOWrapper(zf.open(member), encoding="utf-8")
    elif os.path.isdir(in_path):
        for fn in sorted(os.listdir(in_path)):
            if wanted(fn):
                yield fn, open(os.path.join(in_path, fn), encoding="utf-8")
    else:
        yield os.path.basename(in_path), open(in_path, encoding="utf-8")


def convert(in_path, out_dir, keep_prefix=False, seed=0):
    triples = set()
    prefixes = {}                 # prefix name -> namespace url (from @prefix block)
    schema_props = set()          # local names declared via sh:path (authoritative)
    stats = {"files": 0, "lines": 0, "raw_triples": 0,
             "dropped_non_fact": 0, "dropped_selfloop": 0}

    def render(term):
        return term if keep_prefix else _local_name(term)

    # Resolved lazily once the @prefix block has been read (it precedes all facts
    # in YAGO's serialisation). entity_prefixes/predicate_prefixes are the prefix
    # names whose URL matches the canonical entity / predicate namespace.
    entity_prefixes = set()
    predicate_prefixes = set()

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
                    m = _PREFIX.search(line)
                    if m:
                        prefixes[m.group(1)] = m.group(2)
                        refresh_prefix_sets()
                    continue
                if line.startswith("@"):
                    continue
                # 2. collect authoritative fact-property declarations from SHACL
                shp = _SH_PATH.search(line)
                if shp and shp.group(1) in predicate_prefixes:
                    schema_props.add(shp.group(2))
                # 3. facts are single-line statements ending in '.'
                if not line.endswith("."):
                    continue
                for s, p, o in _parse_statement(line[:-1]):
                    stats["raw_triples"] += 1
                    if not (_in_namespace(s, ENTITY_NS_URL, entity_prefixes)
                            and _in_namespace(o, ENTITY_NS_URL, entity_prefixes)
                            and _in_namespace(p, PREDICATE_NS_URL, predicate_prefixes)):
                        stats["dropped_non_fact"] += 1
                        continue
                    h, r, t = render(s), render(p), render(o)
                    if h == t:
                        stats["dropped_selfloop"] += 1
                        continue
                    triples.add((h, r, t))

    if not entity_prefixes or not predicate_prefixes:
        print()
        print("  WARNING: could not resolve the entity/predicate namespaces from")
        print(f"  the @prefix block. Expected URLs:\n    {ENTITY_NS_URL}\n    {PREDICATE_NS_URL}")
        print("  Found prefixes:", dict(list(prefixes.items())[:8]), "...")

    # ── 90/5/5 split with vocab completeness ──
    # sorted() before shuffle makes the split fully reproducible: a Python set's
    # iteration order varies across processes (randomised string hashing), so
    # list(set) would seed the shuffle from a different order each run. Sorting
    # first pins the split to (seed) alone -- important for a citable dataset.
    import random
    triples = sorted(triples)
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

    # ── write ──
    os.makedirs(out_dir, exist_ok=True)
    for split_name, rows in (("train", train), ("valid", valid), ("test", test)):
        with open(os.path.join(out_dir, f"{split_name}.txt"), "w", encoding="utf-8") as f:
            for h, r, t in rows:
                f.write(f"{h}\t{r}\t{t}\n")

    all_ents = train_ents | {h for h, r, t in valid + test} | {t for h, r, t in valid + test}

    # ── schema validation (Upgrade 2) ──
    # kept relations are LOCAL names; schema_props are LOCAL names. Compare.
    kept_local = train_rels if not keep_prefix else {_local_name(r) for r in train_rels}
    declared = schema_props
    unexpected = kept_local - declared                 # kept but not in schema
    produced = kept_local & declared                   # declared AND has e-e facts

    print()
    print("=" * 64)
    print("  YAGO -> KGSAGE conversion summary")
    print("=" * 64)
    print(f"  source files read       : {stats['files']}")
    print(f"  lines scanned           : {stats['lines']:,}")
    print(f"  raw triples seen        : {stats['raw_triples']:,}")
    print(f"    dropped (not a fact)  : {stats['dropped_non_fact']:,}")
    print(f"    dropped (self-loop)   : {stats['dropped_selfloop']:,}")
    print(f"  unique entity-entity    : {len(triples):,}")
    print(f"    dropped (oov in v/t)  : {dropped_oov:,}")
    print(f"  entities / relations    : {len(all_ents):,} / {len(train_rels)}")
    print(f"  split  train/valid/test : {len(train):,} / {len(valid):,} / {len(test):,}")
    print()
    print(f"  namespaces (from @prefix block, authoritative):")
    print(f"    entity prefixes    : {sorted(entity_prefixes)} -> {ENTITY_NS_URL}")
    print(f"    predicate prefixes : {sorted(predicate_prefixes)} -> {PREDICATE_NS_URL}")
    print(f"  schema fact-properties declared (sh:path) : {len(declared)}")
    print(f"    of which produced entity-entity facts   : {len(produced)}")
    if unexpected:
        print(f"  !! {len(unexpected)} kept relation(s) NOT declared in the schema "
              f"(allowlist drift):")
        for r in sorted(unexpected)[:15]:
            print(f"       - {r}")
    else:
        print(f"    kept relations all schema-declared      : YES")
    print()
    print(f"  written to              : {out_dir}/")
    print()
    print("  Next: confirm v2-A rule-mined positives work on this KG:")
    print(f"    python -m kgsage.cli.verify_templates --dataset {out_dir}")
    if len(train_rels) < 5 or len(triples) < 1000:
        print()
        print("  WARNING: very few relations/triples survived. Check that the")
        print("  @prefix block declared the expected namespace URLs above.")
    return stats


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True,
                    help="YAGO .ttl file, a .zip, or a directory of .ttl files")
    ap.add_argument("--out", required=True,
                    help="output directory for train/valid/test.txt")
    ap.add_argument("--keep-prefix", action="store_true",
                    help="keep fully-qualified pfx:local names instead of "
                         "shortening to local names")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    convert(args.in_path, args.out, keep_prefix=args.keep_prefix, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
