"""Score a recorded observation run against the answer key. Nothing is judged here.

    python scripts/4_evaluate_verdicts.py                    the newest run
    python scripts/4_evaluate_verdicts.py --run runs/run_...json

THE ONLY READER OF ground_truth.tsv. The agents never saw it; this script
checks what they decided, after every decision is frozen in the run file.

Two halves, kept strictly apart:

  OBJECTIVE   anomaly-verdicts vs the planted falsehoods. A planted triple
              is false by human verification, so calling it an anomaly is
              measurably right, whatever the caller's norms.

  VIEWPOINT   the composed picture: what each observer flagged, where they
              agree, and where the SAME fact got DIFFERENT verdicts. A norm
              disagreement on a true fact has no answer key by construction
              -- it is reported with both reasons, never scored.

A caution baked into the numbers: kind=real means the preparation did not plant the
triple, not that it is true -- the graph carries its source's own mistakes.
A "false positive" here may be a discovery.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import collections  # noqa: E402

from loaders import graph  # noqa: E402
from loaders.active import DATASET  # noqa: E402
from loaders.context import get_context  # noqa: E402
from tools._observers import OBSERVER_NAMES  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--run", default=None, help="run file; default is newest")
args = parser.parse_args()

path = Path(args.run) if args.run else None
if path and not path.is_absolute():
    path = ROOT / path
if path is None:
    found = sorted(DATASET.RUNS.glob("run_*_observers.json"))
    if not found:
        raise SystemExit("no observation runs yet. Run scripts/3_run_observers.py first.")
    path = found[-1]

run = json.loads(path.read_text(encoding="utf-8"))

if run["status"] == "invalid":
    raise SystemExit(f"{path.name} is INVALID -- its blindness proof failed. "
                     "Nothing in it is evidence.")
if run["status"] == "truncated":
    print(f"WARNING: {path.name} was TRUNCATED by the API, not by the agents.")
    print("  Missing verdicts below are a rate limit, not a decision.\n")

# ---- the answer key -------------------------------------------------------
truth = {}
for head, relation, tail, label, kind in (
        line.split("\t") for line in
        DATASET.TRUTH.read_text(encoding="utf-8").splitlines()):
    truth[(head, relation, tail)] = int(label)
planted_total = sum(truth.values())

ctx = get_context()
print(f"run: {path.name}")
print(f"{DATASET.NAME}: {len(truth)} triples, {planted_total} planted "
      f"verified-false")

# ---- OBJECTIVE: each observer vs the planted falsehoods --------------------
verdicts_by_agent = dict(zip(OBSERVER_NAMES, run["verdicts"]))

for name in OBSERVER_NAMES:
    judged = verdicts_by_agent[name]
    print("\n" + "=" * 70)
    print(f"  {name} -- OBJECTIVE")
    print("=" * 70)
    if not judged:
        print("  judged nothing.")
        continue

    by_verdict = collections.Counter(v["verdict"] for v in judged.values())
    print(f"  {len(judged)} judged: "
          + "  ".join(f"{k} {n}" for k, n in sorted(by_verdict.items())))

    flagged = {cid: v for cid, v in judged.items() if v["verdict"] == "anomaly"}
    hits = {cid: v for cid, v in flagged.items()
            if truth.get(tuple(v["triple"])) == 1}
    if flagged:
        print(f"  anomaly precision vs planted: {len(hits)}/{len(flagged)} "
              f"({len(hits) / len(flagged):.0%})  -- a flag on an unplanted triple may "
              f"still be a real error inherited from the source data")
    missed = [v for v in judged.values()
              if truth.get(tuple(v["triple"])) == 1 and v["verdict"] != "anomaly"]
    print(f"  planted seen but not flagged: {len(missed)}")
    for v in missed[:3]:
        print(f"    [{v['verdict']:>8}] {ctx.triple_text(tuple(v['triple']))[:56]}"
              f" -- {v['why'][:60]}")
    per_scanner = collections.Counter(v["scanner"] for v in judged.values())
    print(f"  candidates by assistant: {dict(per_scanner)}")

# ---- VIEWPOINT: composition and disagreement ------------------------------
verdict_of = {}
for name in OBSERVER_NAMES:
    for v in verdicts_by_agent[name].values():
        verdict_of.setdefault(tuple(v["triple"]), {})[name] = v

flags = {name: {t for t, by in verdict_of.items()
                if by.get(name, {}).get("verdict") == "anomaly"}
         for name in OBSERVER_NAMES}
agent_1, agent_2 = OBSERVER_NAMES
both_judged = [t for t, by in verdict_of.items() if len(by) == 2]

print("\n" + "=" * 70)
print("  COMPOSED -- the two viewpoints together")
print("=" * 70)
print(f"  flagged: {agent_1} {len(flags[agent_1])}, "
      f"{agent_2} {len(flags[agent_2])}, "
      f"union {len(flags[agent_1] | flags[agent_2])}, "
      f"agreement {len(flags[agent_1] & flags[agent_2])}")
union_hits = sum(1 for t in flags[agent_1] | flags[agent_2] if truth.get(t) == 1)
print(f"  union catches {union_hits} planted -- vs "
      f"{sum(1 for t in flags[agent_1] if truth.get(t)==1)} and "
      f"{sum(1 for t in flags[agent_2] if truth.get(t)==1)} alone")
print(f"  judged by BOTH: {len(both_judged)}")

disagreements = [(t, verdict_of[t]) for t in both_judged
                 if verdict_of[t][agent_1]["verdict"]
                 != verdict_of[t][agent_2]["verdict"]]
print(f"\n  THE DISAGREEMENT SET -- same fact, different verdicts: "
      f"{len(disagreements)}")
print("  (no answer key can settle a norm disagreement on a true fact --")
print("   reported, never scored)")
for triple, by in disagreements:
    planted = "planted-false" if truth.get(triple) == 1 else "unplanted"
    print(f"\n    {ctx.triple_text(triple)}   [{planted}]")
    for name in OBSERVER_NAMES:
        v = by[name]
        print(f"      {name}: [{v['verdict']:>12}] {v['why'][:76]}")
