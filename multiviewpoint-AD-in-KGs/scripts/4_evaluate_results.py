"""Validate what an agent run found. Nothing is scored here.

The agents chose, scripts/3_run_agentic_detector.py carried out their choice and wrote
the ranking into the run file. This reads that ranking and checks it against
the answer key -- so the numbers below describe what the agents ACTUALLY found,
not what a scorer would produce if run again now.

Each section reports, at every cut-off,

    precision@k = anomalies in the top k / k
    recall@k    = anomalies in the top k / all anomalies

plus a global AUC and AUPRC, which need no cut-off. Same shape as ADKGD's
Our_TopK%_RankingList.py, so the numbers sit beside its.

This is the ONLY step that reads ground_truth.tsv. No model is loaded and no
scorer is imported, so a run stays checkable long after its model is gone.

    python scripts/4_evaluate_results.py                       the newest run
    python scripts/4_evaluate_results.py --run runs/run_...json
    python scripts/4_evaluate_results.py --k 1 2 3 4 5 10 --show 15
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from loaders.active import DATASET
from utils import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=None, help="run file; default is the newest")
ap.add_argument("--k", type=float, nargs="+", default=[1, 2, 3, 4, 5],
                help="cut-offs, as a percentage of the graph")
ap.add_argument("--show", type=int, default=10, help="triples to list per section")
args = ap.parse_args()

path = Path(args.run) if args.run else None
if path and not path.is_absolute():
    path = ROOT / path
if path is None:
    found = sorted((ROOT / "runs").glob("run_*.json"))
    if not found:
        raise SystemExit("no runs yet. Run scripts/3_run_agentic_detector.py first.")
    path = found[-1]

run = json.loads(path.read_text(encoding="utf-8"))
health = run.get("health") or {}
if health.get("truncated"):
    why = ("the API refused a request (quota)" if health.get("quota_exhausted")
           else "a model call failed")
    print(f"WARNING: {path.name} is TRUNCATED -- {why}.")
    print(f"  {health.get('total_model_calls')} model calls completed; "
          f"{len(health.get('errors') or [])} failed.")
    for e in (health.get("errors") or [])[:3]:
        print(f"    {e.get('agent')}: {e.get('type')}")
    print("  Whatever is missing below is that failure, not a decision an agent")
    print("  made. Do not read this run as evidence about agent behaviour.\n")

findings = run.get("findings") or []
if not findings:
    # Two different causes, and blaming the wrong one sends the reader looking
    # for a bug in the file format when the agents simply never answered.
    if health.get("truncated"):
        raise SystemExit(
            f"{path.name} recorded no findings because the run was cut short "
            f"before the agents finished.\nThis is not an agent failure. "
            f"Wait for the quota window and run it again.")
    if "findings" not in run:
        raise SystemExit(
            f"{path.name} predates the change that made "
            f"3_run_agentic_detector.py record findings. Rerun it.")
    missing = [i for i, s in enumerate(run.get("specs") or [], 1) if not s]
    raise SystemExit(
        f"{path.name} recorded no findings: "
        f"agent{'s' if len(missing) != 1 else ''} "
        f"{', '.join(map(str, missing)) or '?'} never produced a usable spec, "
        f"so there was nothing to carry out.\nThe agents do sometimes stop "
        f"without answering; nothing retries them. Rerun "
        f"3_run_agentic_detector.py, or evaluate an earlier run with --run.")
goals = run.get("goals") or []

# ---- the answer key -------------------------------------------------------
truth = pd.read_csv(DATASET.TRUTH, sep="\t", header=None,
                    names=["head", "relation", "tail", "label", "kind"])
key = {(r.head, r.relation, r.tail): (int(r.label), r.kind)
       for r in truth.itertuples(index=False)}
n, total = len(truth), int(truth["label"].sum())

kinds = ", ".join(f"{int((truth.kind == k).sum())} {k}" for k in evaluate.KINDS)
print(f"run: {path.name}")
print(f"{DATASET.NAME}: {n} triples, {total} anomalies ({kinds})")
print(f"K% is a share of the GRAPH, not of the anomalies "
      f"-- 1% is {max(1, round(0.01 * n))} triples.")


def as_frame(ranked):
    """A saved ranking with the answer attached. Order is preserved."""
    rows = []
    for h, r, t, score in ranked:
        if (h, r, t) not in key:
            raise SystemExit(
                f"{path.name} ranks a triple that is not in "
                f"{DATASET.TRUTH.name}: {h} {r} {t}.\nThe run and the answer "
                "key come from different contaminations.")
        label, kind = key[(h, r, t)]
        rows.append((h, r, t, score, label, kind))
    return pd.DataFrame(rows, columns=["head", "relation", "tail",
                                       "score", "label", "kind"])


def section(title, subtitle, frame, own_pct=None, show_score=True,
            extra_row=None):
    """One block: the two summary numbers, the K% table, the worst triples."""
    labels = frame["label"].to_numpy()
    discovered = np.cumsum(labels)
    # Position in the ranking is the only signal here -- the saved score is for
    # reading, and the union has none at all.
    by_rank = np.arange(len(frame), 0, -1)

    print("\n" + "=" * 70)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 70)
    print(f"  AUC {roc_auc_score(labels, by_rank):.3f}   "
          f"AUPRC {average_precision_score(labels, by_rank):.3f}   "
          f"(random AUPRC {total / n:.3f})")

    print(f"\n  {'K%':>4}{'k':>7}{'found':>8}{'precision':>12}{'recall':>9}")
    # The agent's own budget joins the fixed cut-offs, marked, so its actual
    # operating point is visible beside the ones used for comparison.
    marks = [(pct, "") for pct in args.k]
    if own_pct is not None and own_pct not in args.k:
        marks.append((own_pct, "   <- this agent's own budget"))
        marks.sort()
    rows = []
    for pct, note in marks:
        k = max(1, min(len(frame), int(round(pct / 100.0 * n))))
        rows.append((pct, k, int(discovered[k - 1]), note))
    if extra_row is not None:
        rows.append(extra_row)
        rows.sort()
    for pct, k, found, note in rows:
        print(f"  {pct:>4g}{k:>7}{found:>8}{found / k:>11.1%}"
              f"{found / total:>9.1%}{note}")

    if args.show:
        cols = (["head", "relation", "tail"]
                + (["score"] if show_score else []) + ["kind"])
        print(f"\n  {args.show} most anomalous:")
        print(frame.head(args.show)[cols].to_string(index=False))


frames = [as_frame(f["ranked"]) for f in findings]
COORD = ["head", "relation", "tail"]

identical = len(frames) > 1 and all(
    frames[0][COORD].equals(f[COORD]) for f in frames[1:])
if identical:
    print("\n  NOTE: every agent produced the SAME ranking -- they chose the "
          "same scorer.\n  This run is a redundancy control, not two "
          "viewpoints. The union would add\n  nothing, so it is skipped, and "
          "the sections below are copies of one another.")

if len(frames) > 1 and not identical:
    # The combined ranking is BEST POSITION ACROSS THE AGENTS: a triple is
    # suspicious if EITHER agent ranks it that badly. That is the union
    # expressed as a ranking, so the same K% table applies. Ties break on the
    # mean position, so where the two agree decides who goes first.
    positions = [{(r.head, r.relation, r.tail): i
                  for i, r in enumerate(f.itertuples(index=False))}
                 for f in frames]
    coords = list(positions[0])
    best = np.array([min(p[c] for p in positions) for c in coords])
    mean = np.array([sum(p[c] for p in positions) / len(positions) for c in coords])
    order = np.lexsort((mean, best))

    union = pd.DataFrame(
        [(*coords[i], 0.0, *key[coords[i]]) for i in order],
        columns=["head", "relation", "tail", "score", "label", "kind"])

    # What a reviewer would actually be handed: the SET union of the two
    # flagged slices. Not a cut of the combined ranking -- the agents chose
    # different budgets, so no single cut-off reproduces it.
    flagged_sets = [
        set(f[COORD].head(finding["flagged"]).itertuples(index=False, name=None))
        for finding, f in zip(findings, frames)]
    both_flagged = set().union(*flagged_sets)
    found_here = sum(key[c][0] for c in both_flagged)
    extra = (round(100.0 * len(both_flagged) / n, 1), len(both_flagged), found_here,
             "   <- what the two agents actually flagged, together")

    section("BOTH AGENTS  (union)",
            "flagged if either agent ranks it that badly", union,
            show_score=False, extra_row=extra)

for finding, frame in zip(findings, frames):
    i = finding["agent"]
    goal = goals[i - 1] if len(goals) >= i else ""
    section(f"AGENT {i}", f"{finding['scorer']} -- {goal}", frame,
            own_pct=finding["budget"] * 100)

    # Semantic consistency: of what this agent flagged, how much involves the
    # relations it declared it was auditing? Reads the frame and the ranking,
    # never a label -- so it says whether the agent did what it SAID, which is
    # a different question from whether what it said was right.
    sem = finding.get("semantics")
    if sem:
        flagged = frame.head(finding["flagged"])
        in_scope = int(flagged["relation"].isin(sem["relations"]).sum())
        got = in_scope / len(flagged)
        # Against what a scorer that ignored the frame entirely would give:
        # the declared relations' share of the graph. Without this the raw
        # percentage says nothing -- 63% is strong on a rare relation and
        # meaningless on one that is 60% of the triples to begin with.
        base = float(truth["relation"].isin(sem["relations"]).mean())
        print("\n  declared frame")
        print(f"    in scope:   {', '.join(sem['relations'])}")
        for field in ("entities", "normal", "suspicious", "impossible"):
            if sem.get(field):
                print(f"    {field + ':':<12}{sem[field]}")
        print(f"  semantic consistency: {in_scope}/{len(flagged)} flagged "
              f"triples use a declared relation ({got:.1%})")
        print(f"    vs {base:.1%} if the flags ignored the frame "
              f"-- {100 * (got - base):+.1f} points. No labels involved.")
