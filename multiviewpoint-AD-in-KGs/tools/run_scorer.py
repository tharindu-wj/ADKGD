"""Tool: score every triple with one scorer and flag the worst slice.

THE ONLY SCORING PATH. No other code an agent can reach produces a number.

Returns a summary plus a few of the flagged triples, because an agent has to
see triples to judge them. It never returns a label, and it never says whether
a flag was right -- that is the evaluator's job, after the run.
"""
import numpy as np

from loaders import graph
from loaders.active import DATASET
from tools.scorers import neighbourhood, plausibility
from utils import detect

#: the menu. Adding a scorer here does NOT change the agent's tool list.
SCORERS = {s.NAME: s for s in (plausibility, neighbourhood)}

SHOW = 8


def run_scorer(scorer: str, budget: float = 0.10, model: str = "distmult") -> str:
    """Score every triple with one scorer and show you the worst of them.

    This is the only way to get a number out of this graph. Returns how many
    triples were flagged, the spread of scores across the whole graph and
    across the flagged slice, and a handful of the flagged triples themselves.

    Nothing it returns says whether a flag is CORRECT. There is no answer key
    here and none will be offered. Judging the triples it hands back is your
    job, and the reason it hands them back at all.

    You must call declare_semantics before this will answer you.

    Args:
        scorer: which scorer to run. Ask for one that exists; the error tells
            you the menu if you get it wrong.
        budget: the fraction of the graph to flag, above 0 and at most 0.5.
            This is a review cost -- every flagged triple is one someone would
            have to check.
        model: which trained model to use, for scorers that need one.
    """
    if scorer not in SCORERS:
        return (f"ERROR: no scorer '{scorer}'. "
                f"Available: {', '.join(sorted(SCORERS))}.")
    if not 0 < budget <= 0.5:
        return f"ERROR: budget must be between 0 and 0.5, got {budget}."

    mod = SCORERS[scorer]
    triples = graph.load_triples(DATASET.KG)

    if mod.NEEDS_MODEL:
        model_dir = DATASET.MODELS / model
        if not (model_dir / "trained_model.pkl").exists():
            return (f"ERROR: no trained model at models/{DATASET.NAME}/{model}. "
                    f"Either use a scorer that needs none, or ask for a model "
                    f"that exists.")
        values = mod.score(triples, model_dir=model_dir, kg_path=DATASET.KG)
    else:
        values = mod.score(triples)

    flagged, n = detect.flag_worst(values, mod.DIRECTION, budget)
    v = np.asarray(values, dtype=float)
    picked = [t for t, f in zip(triples, flagged) if f]

    lines = [
        f"{scorer}: scored {len(triples)} triples, flagged the worst {n} "
        f"({100 * n / len(triples):.1f}%).",
        f"  score spread over all triples: min {v.min():.3f}, "
        f"median {np.median(v):.3f}, max {v.max():.3f}",
        f"  among the flagged: min {v[flagged].min():.3f}, "
        f"max {v[flagged].max():.3f}",
        "",
        f"{min(SHOW, len(picked))} of the flagged triples, worst first:",
    ]
    order = np.argsort(v[flagged] if mod.DIRECTION < 0 else -v[flagged])
    for i in order[:SHOW]:
        h, r, t = picked[i]
        lines.append(f"  {h}\t{r}\t{t}\t({v[flagged][i]:.3f})")
    lines.append("")
    lines.append("Judge these yourself. Nothing here says whether they are right.")
    # The tool's own response is the last thing the model reads before choosing
    # what to do next, which makes it the strongest place to put this. Agents
    # were reliably stopping here, having scored but never handed anything in.
    lines.append("If this is the scorer you want, call submit_spec now -- "
                 "scoring is not deciding, and nothing is recorded until you do.")
    return "\n".join(lines)
