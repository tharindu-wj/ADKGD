"""Human-readable KGSAGE corruption-quality report (no plausibility scorer).

Samples a fraction of the graph, corrupts each triple with the trained
generator through the DEPLOYED decode path, resolves opaque IDs to readable
names via entity2text, and writes:

  <out>.md   config + model-free quality metrics + a readable sample of
             (original -> corrupted) pairs, for eyeballing;
  <out>.tsv  EVERY sampled pair (readable), ready to batch-feed an LLM for
             correctness / real-world fact-checking.

By design it computes NO plausibility/hardness score -- it hands you the two
triples in plain English so you (or an LLM) judge correctness. All automatic
numbers here are model-free and non-circular: type-validity, closed-world
falseness, diversity / mode-collapse, and slot distribution. A collapsed
generator (e.g. WN18RR at default settings) shows up as very low distinct-
entity coverage and entropy.

entity2text.txt and relation2text.txt are auto-detected inside --data, so the
usual call is just (repo root, PYTHONPATH=experiments):
  python -m kgsage.cli.quality_report \
      --ckpt experiments/kgsage/outputs/checkpoints/kgsage_fb15k237_all_s0.pt \
      --data data/FB15K-237 \
      --sample_frac 0.05 --out reports/quality_fb15k237.md
(pass --entity2text / --relation2text explicitly to override the auto-detected files.)
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

from kgsage.inference import load_checkpoint, generate_negatives  # noqa: E402


def _read_text_map(path):
    m = {}
    if path and Path(path).exists():
        with open(path, encoding="utf-8-sig") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    m[parts[0]] = parts[1]
    return m


def _read_triples(path):
    out = []
    if not Path(path).exists():
        return out
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3:
                out.append(tuple(p))
    return out


def _clean_rel(r):
    # readable-but-recognisable: drop leading / or _, keep the rest
    return r.lstrip("/_") if isinstance(r, str) else r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="dataset dir (train/valid/test.txt)")
    ap.add_argument("--entity2text", default=None,
                    help="TSV id<TAB>name; auto-detected as <data>/entity2text.txt if omitted")
    ap.add_argument("--relation2text", default=None,
                    help="TSV relation<TAB>text; auto-detected as <data>/relation2text.txt if omitted")
    ap.add_argument("--sample_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_shown", type=int, default=120, help="pairs listed in the .md (all go to the .tsv)")
    ap.add_argument("--out", default="kgsage_quality_report.md")
    args = ap.parse_args()

    payload = load_checkpoint(args.ckpt)
    e2g, r2g = payload["ent2id"], payload["rel2id"]
    id2e, id2r = payload["id2ent"], payload["id2rel"]
    real_all = payload["real_triple_set"]

    # auto-detect the text maps inside --data unless overridden
    if args.entity2text is None:
        cand = Path(args.data) / "entity2text.txt"
        args.entity2text = str(cand) if cand.exists() else None
    if args.relation2text is None:
        cand = Path(args.data) / "relation2text.txt"
        args.relation2text = str(cand) if cand.exists() else None

    ent_txt = _read_text_map(args.entity2text)
    rel_txt = _read_text_map(args.relation2text)
    named = len(ent_txt) > 0
    print(f"names: entity2text={'yes' if ent_txt else 'no'} "
          f"({len(ent_txt):,}), relation2text={'yes' if rel_txt else 'no'} "
          f"({len(rel_txt):,})")

    def ename(gid):
        s = id2e[gid]
        return ent_txt.get(s, s)

    def rname(gid):
        s = id2r[gid]
        return rel_txt.get(s, _clean_rel(s))

    # triple universe = all splits, mapped to GAN ids (skip anything out-of-vocab)
    triples, skipped = [], 0
    for split in ("train", "valid", "test"):
        for h, r, t in _read_triples(Path(args.data) / f"{split}.txt"):
            if h in e2g and t in e2g and r in r2g:
                triples.append((e2g[h], r2g[r], e2g[t]))
            else:
                skipped += 1
    if not triples:
        print("no in-vocab triples found under --data", file=sys.stderr)
        return 1

    # type pools from the TRAIN split (for the type-validity check)
    train_gan = [(e2g[h], r2g[r], e2g[t])
                 for h, r, t in _read_triples(Path(args.data) / "train.txt")
                 if h in e2g and t in e2g and r in r2g]
    head_pool, tail_pool = {}, {}
    for h, r, t in train_gan:
        head_pool.setdefault(r, set()).add(h)
        tail_pool.setdefault(r, set()).add(t)

    rng = np.random.default_rng(args.seed)
    n_sample = max(1, int(round(len(triples) * args.sample_frac)))
    idx = rng.choice(len(triples), size=n_sample, replace=False)
    sample = [triples[i] for i in idx]

    maps = {"id2ent": id2e, "id2rel": id2r, "ent2id": e2g, "rel2id": r2g}
    negatives, stats = generate_negatives(sample, payload, maps,
                                          rng=np.random.default_rng(args.seed + 1))

    # ---- model-free metrics over the whole sample ----
    rows = []
    type_valid = false_ct = null_ct = head_ct = tail_ct = 0
    replacements = []              # (relation, replacement_entity) for diversity
    for (h, r, t), (nh, nr, nt) in zip(sample, negatives):
        if nh == h and nt == t:
            null_ct += 1
            rows.append(((h, r, t), (nh, nr, nt), "none", None, None))
            continue
        slot = "head" if nh != h else "tail"
        repl = nh if slot == "head" else nt
        pool = head_pool.get(r, set()) if slot == "head" else tail_pool.get(r, set())
        tv = repl in pool
        isf = (nh, nr, nt) not in real_all
        type_valid += int(tv); false_ct += int(isf)
        head_ct += int(slot == "head"); tail_ct += int(slot == "tail")
        replacements.append((r, repl))
        rows.append(((h, r, t), (nh, nr, nt), slot, tv, isf))

    n = len(sample)
    emitted = n - null_ct
    repl_entities = [e for _, e in replacements]
    distinct = len(set(repl_entities))
    counts = Counter(repl_entities)
    total = sum(counts.values()) or 1
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values()) if counts else 0.0
    max_entropy = math.log2(distinct) if distinct > 1 else 1.0
    distinct_rel_pairs = len(set(replacements))

    def pct(x, d):
        return f"{100 * x / d:.1f}%" if d else "n/a"

    # ---- write the .tsv (every pair, for LLM fact-checking) ----
    out_md = Path(args.out)
    out_tsv = out_md.with_suffix(".tsv")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("idx\tchanged\ttype_valid\tis_false\t"
                 "orig_head\torig_relation\torig_tail\t"
                 "corr_head\tcorr_relation\tcorr_tail\n")
        for i, ((h, r, t), (nh, nr, nt), slot, tv, isf) in enumerate(rows):
            fh.write("\t".join(str(x) for x in [
                i, slot, tv, isf,
                ename(h), rname(r), ename(t),
                ename(nh), rname(nr), ename(nt)]) + "\n")

    # ---- write the .md report ----
    L = []
    L.append(f"# KGSAGE corruption-quality report\n")
    L.append(f"- **checkpoint:** `{Path(args.ckpt).name}`")
    L.append(f"- **dataset:** `{args.data}`  ({len(triples):,} in-vocab triples across all splits"
             + (f", {skipped:,} skipped" if skipped else "") + ")")
    L.append(f"- **sample:** {n_sample:,} triples ({args.sample_frac:.0%}), seed {args.seed}")
    L.append(f"- **names:** entities {'✓' if ent_txt else 'RAW IDS'}, "
             f"relations {'✓' if rel_txt else 'cleaned strings'}")
    L.append(f"- **full pair list for LLM fact-check:** `{out_tsv.name}`\n")

    L.append("## Quality at a glance (model-free — no plausibility scorer)\n")
    L.append("| metric | value | what it tells you |")
    L.append("|---|---|---|")
    L.append(f"| Type-valid | {pct(type_valid, emitted)} | replacement is a legal filler for the relation (higher = better) |")
    L.append(f"| Truly false | {pct(false_ct, emitted)} | not a real fact in any split (should be ~100%) |")
    L.append(f"| Null (kept original) | {pct(null_ct, n)} | generator found no valid corruption (should be ~0%) |")
    L.append(f"| Slot head / tail | {pct(head_ct, emitted)} / {pct(tail_ct, emitted)} | relation slot is never corrupted |")
    L.append(f"| Distinct-entity coverage | {pct(distinct, emitted)} ({distinct:,} unique) | LOW = mode collapse (the WN18RR failure) |")
    L.append(f"| Entropy of replacements | {entropy:.2f} / {max_entropy:.2f} bits | how evenly spread the fakes are |")
    L.append("")
    L.append("> **How to read plausibility:** this report does not score it. "
             "Feed `" + out_tsv.name + "` to an LLM (or fact-check by hand) asking, per row, "
             "whether the corrupted fact is *true*, *false*, or *unknown*. A good KGSAGE fake "
             "is **believable but false**; a *true* verdict is a false negative.\n")

    # most-repeated replacements (collapse tell)
    top = counts.most_common(8)
    if top:
        L.append("## Most-repeated replacements (a collapse tell)\n")
        L.append("| replacement entity | times |")
        L.append("|---|---|")
        for gid, c in top:
            L.append(f"| {ename(gid)} | {c} |")
        L.append("")

    L.append(f"## Sample corruptions (first {min(args.max_shown, len(rows))} of {len(rows):,})\n")
    for i, ((h, r, t), (nh, nr, nt), slot, tv, isf) in enumerate(rows[:args.max_shown]):
        if slot == "none":
            L.append(f"**{i}.** `({ename(h)}, {rname(r)}, {ename(t)})` — KEPT ORIGINAL (no corruption)\n")
            continue
        flags = f"{'type-valid ✓' if tv else 'type-INVALID ✗'} · {'false ✓' if isf else 'REAL ✗'}"
        L.append(f"**{i}.** [{slot} changed · {flags}]")
        L.append(f"- orig: `({ename(h)}, {rname(r)}, {ename(t)})`")
        L.append(f"- corr: `({ename(nh)}, {rname(nr)}, {ename(nt)})`\n")

    L.append("## Caveats\n")
    L.append("- **Falseness is closed-world:** 'truly false' only checks the KG's own splits. "
             "A corruption absent from the KG could still be a real-world fact (a false negative). "
             "That is exactly what the LLM / fact-check pass on the `.tsv` is for.")
    L.append("- **No hardness score here by design** — plausibility is left to your LLM / human judgement.")
    if not named:
        L.append("- **No entity2text was applied**, so triples show raw IDs. Pass `--entity2text` for readable names.")

    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"wrote {out_md}  ({n_sample:,} sampled, type-valid {pct(type_valid, emitted)}, "
          f"false {pct(false_ct, emitted)}, distinct-coverage {pct(distinct, emitted)}, "
          f"null {null_ct})")
    print(f"wrote {out_tsv}  (all {len(rows):,} pairs, for LLM fact-checking)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
