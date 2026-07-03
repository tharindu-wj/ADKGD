"""LEGACY ARM (B1a) TOOL — for contradiction-trained (context-distant)
checkpoints. It draws RAW single-shot generator samples (NOT the deployed
decode path: kgsage.inference adds known-true/self/pool masks + bounded
resample), and its ctx-dist column assumes the B1a far-from-context
objective. For A-ii checkpoints use `python -m kgsage.cli.inspect_gan_lp`.

Inspect a trained KGSAGE generator's corruptions for HUMAN evaluation.

Picks random real triples, corrupts an ENTITY slot with the trained generator
(a few diverse draws) plus one random corruption for calibration, and writes:

  1. a Markdown REPORT showing, for each triple, the NEIGHBOURHOOD of the head
     and the original slot value — so a human can judge type + context by
     STRUCTURE even when the entities are opaque IDs (e.g. FB15K-237 MIDs); and
  2. a CSV RATINGS SHEET (one row per corruption, blank rating columns) for
     scoring and later aggregation.

Everything comes from the checkpoint alone — it caches the real triples, the
id<->string maps, and the context table E' — so no dataset files (and no PyG)
are needed.

Run from the repo root:
  PYTHONPATH=experiments python -m kgsage.cli.inspect_corruptions \
      --ckpt experiments/kgsage/outputs/checkpoints/kgsage_fb15k237.pt \
      --num 20 --min_degree 10 --slot tail --seed 0

How to read the report (structure-only evaluation):
  * a relation is readable (e.g. /location/country/form_of_government) and tells
    you the slot's TYPE;
  * an entity's "fingerprint" = the relations it participates in = its type
    signature (a country participates in capital/contains/form_of_government...);
  * a good corruption is TYPE-VALID (fits the relation), PLAUSIBLE (could be a
    real fact), and FALSE (absent + contradicts the head's neighbourhood).
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# --- path bootstrap so `python -m kgsage.cli.inspect_corruptions` works ---
_EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import numpy as np
import torch
import torch.nn.functional as F

# Eval tool: correctness beats speed, and single-threading also dodges a local
# MKL/OpenMP crash on some Windows machines. Harmless on HPC.
torch.set_num_threads(1)

from kgsage import load_kg
from kgsage.inference import load_checkpoint

HEAD, TAIL = 0, 2  # slot ids (relation slot, 1, is intentionally not inspected)


# --------------------------------------------------------------------------- #
# Graph structure                                                             #
# --------------------------------------------------------------------------- #
def build_graph_index(triples):
    """From all (h, r, t), build adjacency + degree + type-validity tables.

    Returns a dict:
      out_edges[e]   : list of (r, t) with e as head
      in_edges[e]    : list of (h, r) with e as tail
      degree[e]      : out + in count
      rel_profile[e] : Counter of relations e touches (as head or tail)
      tail_count[(r,t)] / head_count[(r,h)] : how often a value fills that slot
      tail_pool[r] / head_pool[r] : the set of legal fillers for that slot
    """
    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    rel_profile = defaultdict(Counter)
    tail_count = Counter()
    head_count = Counter()
    tail_pool = defaultdict(set)
    head_pool = defaultdict(set)
    for h, r, t in triples:
        out_edges[h].append((r, t))
        in_edges[t].append((h, r))
        rel_profile[h][r] += 1
        rel_profile[t][r] += 1
        tail_count[(r, t)] += 1
        head_count[(r, h)] += 1
        tail_pool[r].add(t)
        head_pool[r].add(h)
    degree = {e: len(out_edges[e]) + len(in_edges[e])
              for e in set(out_edges) | set(in_edges)}
    return dict(out_edges=out_edges, in_edges=in_edges, degree=degree,
                rel_profile=rel_profile, tail_count=tail_count, head_count=head_count,
                tail_pool=tail_pool, head_pool=head_pool)


def short_rel(rel_string):
    """Compact form of a relation path for inline display (last segment)."""
    return rel_string.rstrip("/").split("/")[-1] or rel_string


def fingerprint(entity, gi, id2rel, top_k=6):
    """The entity's 'type signature': the relations it most participates in."""
    common = gi["rel_profile"][entity].most_common(top_k)
    return ", ".join(short_rel(id2rel[r]) for r, _ in common) or "(isolated)"


def neighbourhood_lines(entity, gi, id2ent, id2rel, cap):
    """Capped, formatted neighbour edges (out then in) for the report."""
    lines = []
    for r, t in gi["out_edges"][entity][:cap]:
        lines.append(f"    {id2ent[entity]} --{short_rel(id2rel[r])}--> {id2ent[t]}")
    for h, r in gi["in_edges"][entity][:max(cap - len(lines), 0)]:
        lines.append(f"    {id2ent[h]} --{short_rel(id2rel[r])}--> {id2ent[entity]}")
    deg = gi["degree"].get(entity, 0)
    if deg > len(lines):
        lines.append(f"    ... (+{deg - len(lines)} more)")
    return lines or ["    (no neighbours)"]


# --------------------------------------------------------------------------- #
# Readable entity names (optional: --labels file and/or --wordnet)            #
# --------------------------------------------------------------------------- #
def load_label_map(labels_file):
    """Optional TSV `entity_string<TAB>readable_name` -> dict (dataset-agnostic)."""
    mapping = {}
    if not labels_file:
        return mapping
    with open(labels_file, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping


def make_wordnet_resolver(enabled):
    """Best-effort resolver for numeric WordNet synset offsets (WN18RR).

    WN18RR entity IDs are bare offsets with NO part-of-speech, so we try POS in
    order (noun, verb, adj, adv) and take the first synset that resolves. This is
    best-effort — an offset that exists for several POS may resolve to the wrong
    one — but a readable synset name (e.g. `land_reform.n.01`) helps a human far
    more than a bare number. Returns a function, or None if disabled/unavailable.
    """
    if not enabled:
        return None
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        print("  [--wordnet] nltk/WordNet unavailable -> showing raw offsets. Enable with: "
              "pip install nltk && python -c \"import nltk; nltk.download('wordnet')\"", flush=True)
        return None
    import warnings
    cache = {}

    def resolve(ent):
        if ent not in cache:
            name = ent
            if ent.isdigit():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # nltk warns per wrong-POS probe
                    for pos in ("n", "v", "a", "r"):
                        try:
                            name = wn.synset_from_pos_and_offset(pos, int(ent)).name()
                            break
                        except Exception:
                            continue
            cache[ent] = name
        return cache[ent]
    return resolve


class ReadableEntities:
    """Dict-like id -> readable string: label map > WordNet resolver > raw. Lazy + cached.

    A drop-in replacement for id2ent, so every entity display in the report/CSV
    goes through it without touching the render functions.
    """
    def __init__(self, id2ent, label_map, wn_resolve):
        self.id2ent, self.label_map, self.wn_resolve, self.cache = id2ent, label_map, wn_resolve, {}

    def __getitem__(self, i):
        if i not in self.cache:
            s = self.id2ent[i]
            if s in self.label_map:
                r = self.label_map[s]
            elif self.wn_resolve is not None:
                r = self.wn_resolve(s)
            else:
                r = s
            self.cache[i] = r
        return self.cache[i]


# --------------------------------------------------------------------------- #
# Corruption generation                                                       #
# --------------------------------------------------------------------------- #
def model_corruption(payload, h, r, t, slot):
    """One SINGLE-SHOT generator draw for `slot` — matches kgsage.inference now
    (no retry, no fallback). Returns (value, confidence, failed).

    failed=True if the single pick is a self-loop or collides with a real triple,
    i.e. the case where inference keeps the ORIGINAL triple (used_original).
    """
    G, device, E = payload["generator"], payload["device"], payload["entity_context"]
    real_set, z_dim = payload["real_triple_set"], payload["z_dim"]

    h_t = torch.tensor([h], device=device)
    r_t = torch.tensor([r], device=device)
    t_t = torch.tensor([t], device=device)
    z = torch.randn(1, z_dim, device=device)
    with torch.no_grad():
        head_logits, _rel_logits, tail_logits = G(h_t, r_t, t_t, z, E)

    logits = (head_logits if slot == HEAD else tail_logits)[0]
    true_idx = h if slot == HEAD else t
    probs = torch.softmax(logits, dim=0)

    masked = logits.clone()
    masked[true_idx] = float("-inf")       # never re-emit the true value
    u = torch.rand_like(masked).clamp_(1e-10, 1 - 1e-10)
    gumbel = -torch.log(-torch.log(u))
    idx = int((masked + gumbel * 0.5).argmax())        # single shot, no retry
    candidate = (idx, r, t) if slot == HEAD else (h, r, idx)
    failed = (candidate[0] == candidate[2]) or (candidate in real_set)
    return idx, float(probs[idx]), failed


def random_corruption(h, r, t, slot, n_ent, real_set, rng, max_tries=200):
    """Uniformly random entity in the slot (ANY entity) — the B0-style baseline."""
    true_idx = h if slot == HEAD else t
    idx = true_idx
    for _ in range(max_tries):
        idx = int(rng.integers(0, n_ent))
        candidate = (idx, r, t) if slot == HEAD else (h, r, idx)
        if idx != true_idx and candidate[0] != candidate[2] and candidate not in real_set:
            return idx
    return idx


def context_distance(E, value, anchor):
    """1 - cos(E'[value], E'[anchor]) — high = far from the anchor's context."""
    return float(1.0 - F.cosine_similarity(E[value].unsqueeze(0), E[anchor].unsqueeze(0)))


def random_distance_baseline(E, pool, anchor, rng, k=50):
    """Average context distance of random TYPE-VALID fillers (a reference point)."""
    pool = list(pool)
    if not pool:
        return float("nan")
    idx = rng.integers(0, len(pool), size=min(k, len(pool)))
    values = torch.tensor([pool[i] for i in idx], device=E.device)
    cos = F.cosine_similarity(E[values], E[anchor].unsqueeze(0).expand(len(values), -1), dim=1)
    return float((1.0 - cos).mean())


def choose_slot(mode, rng):
    if mode == "tail":
        return TAIL
    if mode == "head":
        return HEAD
    return TAIL if rng.integers(0, 2) == 0 else HEAD


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def render_block(idx, h, r, t, slot, anchor, original_value, diags,
                 rand_avg, gi, id2ent, id2rel, cap):
    """Markdown for one triple: original + neighbourhoods + corruptions table."""
    slot_name = "TAIL" if slot == TAIL else "HEAD"
    L = [f"\n---\n\n## Triple {idx}  -  relation `{id2rel[r]}`\n",
         f"**Original:**  `{id2ent[h]}` --{short_rel(id2rel[r])}--> `{id2ent[t]}`   "
         f"**(corrupting {slot_name})**\n",
         f"**Head `{id2ent[h]}`**  (degree {gi['degree'].get(h, 0)})  "
         f"fingerprint: _{fingerprint(h, gi, id2rel)}_",
         "```"]
    L += neighbourhood_lines(h, gi, id2ent, id2rel, cap)
    L += ["```",
          f"**Original {slot_name.lower()} `{id2ent[original_value]}`**  "
          f"(degree {gi['degree'].get(original_value, 0)})  "
          f"fingerprint: _{fingerprint(original_value, gi, id2rel)}_",
          "```"]
    L += neighbourhood_lines(original_value, gi, id2ent, id2rel, cap)
    L += ["```",
          "",
          "| # | source | corrupted value | type-valid (count) | absent | ctx-dist (rand avg) | conf |",
          "|--:|--------|-----------------|--------------------|--------|---------------------|------|"]
    for i, d in enumerate(diags, 1):
        tv = f"YES ({d['count']}x)" if d["count"] > 0 else "**no**"
        ab = "YES" if d["absent"] else "**no**"
        cd = f"{d['ctx_dist']:.2f} ({rand_avg:.2f})" if rand_avg == rand_avg else f"{d['ctx_dist']:.2f}"
        cf = f"{d['conf']:.2f}" if d["conf"] is not None else "-"
        src = d["source"] + (" **FAIL**" if d.get("failed") else "")
        L.append(f"| {i} | {src} | `{id2ent[d['value']]}` | {tv} | {ab} | {cd} | {cf} |")
    L += ["", "Corrupted-value fingerprints:"]
    for d in diags:
        v = d["value"]
        L.append(f"- `{id2ent[v]}` (deg {gi['degree'].get(v, 0)}): _{fingerprint(v, gi, id2rel)}_")
    L.append("\n**Rating:**  type-valid? [ ]   plausible? [ ]   false? [ ]   notes: ______________")
    return L


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="experiments/kgsage/outputs/checkpoints/kgsage_fb15k237.pt")
    ap.add_argument("--data", default=None,
                    help="dataset dir (same one the checkpoint was trained on); required for --split != all")
    ap.add_argument("--split", choices=["all", "train", "valid", "test"], default="all",
                    help="which split's triples to corrupt. test/valid = HELD-OUT (unseen by the GAN)")
    ap.add_argument("--num", type=int, default=20, help="how many real triples to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relation", default=None, help="only sample triples with this relation string")
    ap.add_argument("--min_degree", type=int, default=0, help="min degree of the HEAD entity")
    ap.add_argument("--slot", choices=["tail", "head", "both"], default="tail")
    ap.add_argument("--corruptions", type=int, default=3, help="model draws per triple")
    ap.add_argument("--neighbours", type=int, default=8, help="max neighbour edges shown per entity")
    ap.add_argument("--out_dir", default="experiments/kgsage/outputs/inspect")
    ap.add_argument("--wordnet", action="store_true",
                    help="resolve numeric WordNet synset offsets (WN18RR) to readable names via nltk")
    ap.add_argument("--labels", default=None,
                    help="optional TSV `entity_string<TAB>readable_name` to display readable entities")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    print(f"Loading checkpoint {args.ckpt} ...", flush=True)
    payload = load_checkpoint(args.ckpt, device=device)
    id2ent, id2rel, rel2id = payload["id2ent"], payload["id2rel"], payload["rel2id"]
    n_ent, E, real_set = payload["n_ent"], payload["entity_context"], payload["real_triple_set"]
    print(f"  {n_ent:,} entities, {payload['n_rel']:,} relations, {len(real_set):,} real triples", flush=True)

    # Optional readable entity names (label file and/or WordNet offset resolver).
    # Wrapping id2ent makes every entity display in the report/CSV readable.
    label_map = load_label_map(args.labels)
    wn_resolve = make_wordnet_resolver(args.wordnet)
    if label_map or wn_resolve:
        modes = ([f"labels-file ({len(label_map):,})"] if label_map else []) + (["wordnet"] if wn_resolve else [])
        print(f"  entity names: {', '.join(modes)}", flush=True)
    id2ent = ReadableEntities(id2ent, label_map, wn_resolve)

    # Neighbourhoods come from ALL real triples (full context regardless of split).
    print("Building graph index ...", flush=True)
    gi = build_graph_index(list(real_set))

    # Sampling universe: a specific split (held-out) or all triples. Splits load
    # via load_kg; ids match the checkpoint because the vocab is built
    # deterministically (first-seen) from the same dataset.
    if args.split == "all":
        sample_universe = list(real_set)
    else:
        if not args.data:
            raise SystemExit("--split requires --data (the dataset the checkpoint was trained on)")
        kg = load_kg(args.data)
        if kg["n_ent"] != n_ent or kg["n_rel"] != payload["n_rel"]:
            raise SystemExit("--data does not match the checkpoint (n_ent/n_rel differ). "
                             "Pass the same dataset the checkpoint was trained on.")
        split_map = {"train": kg["triples_train"], "valid": kg["triples_valid"], "test": kg["triples_test"]}
        sample_universe = [tuple(x) for x in split_map[args.split]]
        held = " (HELD-OUT: unseen by the GAN)" if args.split in ("valid", "test") else ""
        print(f"  sampling from '{args.split}' split: {len(sample_universe):,} triples{held}", flush=True)

    # ---- eligible triples (relation + head-degree filters) ----
    rel_filter = None
    if args.relation is not None:
        if args.relation not in rel2id:
            examples = "\n  ".join(sorted(rel2id)[:10])
            raise SystemExit(f"relation {args.relation!r} not found. Example relations:\n  {examples}")
        rel_filter = rel2id[args.relation]
    eligible = [trip for trip in sample_universe
                if (rel_filter is None or trip[1] == rel_filter)
                and gi["degree"].get(trip[0], 0) >= args.min_degree]
    if not eligible:
        raise SystemExit("No triples match the filters (try lowering --min_degree).")
    print(f"  {len(eligible):,} eligible triples after filters", flush=True)

    pick = rng.integers(0, len(eligible), size=min(args.num, len(eligible)))
    sampled = [eligible[i] for i in pick]

    # ---- generate, diagnose, render ----
    md = [f"# KGSAGE corruption inspection\n",
          f"checkpoint: `{args.ckpt}`  |  split={args.split}  |  {len(sampled)} triples  |  "
          f"slot={args.slot}  |  min_degree={args.min_degree}  |  seed={args.seed}\n",
          "_Judge each corruption by structure: does the corrupted value's fingerprint fit "
          "the relation (type-valid), could it be real (plausible), and does it contradict the "
          "head's neighbourhood (false)?  A **FAIL** corruption is a self-loop/collision — "
          "single-shot inference would keep the original triple._"]
    csv_rows = []
    n_model_total = 0
    n_model_failed = 0

    for idx, (h, r, t) in enumerate(sampled, 1):
        slot = choose_slot(args.slot, rng)
        anchor = h if slot == TAIL else t
        original_value = t if slot == TAIL else h
        pool = gi["tail_pool"][r] if slot == TAIL else gi["head_pool"][r]
        count_table = gi["tail_count"] if slot == TAIL else gi["head_count"]
        rand_avg = random_distance_baseline(E, pool, anchor, rng)

        diags = []
        for _ in range(args.corruptions):
            val, conf, failed = model_corruption(payload, h, r, t, slot)
            diags.append(dict(source="model", value=val, conf=conf, failed=failed))
            n_model_total += 1
            n_model_failed += int(failed)
        diags.append(dict(source="random",
                          value=random_corruption(h, r, t, slot, n_ent, real_set, rng),
                          conf=None, failed=False))
        # attach diagnostics
        for d in diags:
            candidate = (h, r, d["value"]) if slot == TAIL else (d["value"], r, t)
            d["count"] = count_table[(r, d["value"])]
            d["absent"] = candidate not in real_set
            d["ctx_dist"] = context_distance(E, d["value"], anchor)

        md += render_block(idx, h, r, t, slot, anchor, original_value, diags,
                           rand_avg, gi, id2ent, id2rel, args.neighbours)

        slot_name = "TAIL" if slot == TAIL else "HEAD"
        for d in diags:
            csv_rows.append({
                "idx": idx, "head": id2ent[h], "relation": id2rel[r], "tail": id2ent[t],
                "slot": slot_name, "source": d["source"], "corrupted_value": id2ent[d["value"]],
                "type_valid": d["count"] > 0, "type_valid_count": d["count"],
                "generation_failed": d.get("failed", False),
                "absent_from_graph": d["absent"], "ctx_distance": round(d["ctx_dist"], 4),
                "ctx_distance_random_avg": round(rand_avg, 4) if rand_avg == rand_avg else "",
                "gen_confidence": round(d["conf"], 4) if d["conf"] is not None else "",
                "rating_type_valid": "", "rating_plausible": "", "rating_false": "", "notes": "",
            })

    fail_rate = (f"{n_model_failed}/{n_model_total} ({n_model_failed / n_model_total:.1%})"
                 if n_model_total else "n/a")
    md.insert(3, f"\n**Model single-shot failure rate:** {fail_rate}  "
                 f"(self-loop / collision -> inference keeps the original triple)\n")

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, f"corruptions_{args.split}_seed{args.seed}.md")
    csv_path = os.path.join(args.out_dir, f"corruptions_{args.split}_seed{args.seed}.csv")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nWrote:\n  report : {report_path}\n  ratings: {csv_path}", flush=True)
    print(f"  {len(sampled)} triples x ({args.corruptions} model + 1 random) = "
          f"{len(csv_rows)} corruptions to review", flush=True)
    print(f"  model single-shot failures (self-loop/collision -> would use original): {fail_rate}",
          flush=True)


if __name__ == "__main__":
    main()
