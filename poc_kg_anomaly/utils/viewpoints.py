"""The viewpoints. One function per way of looking at a triple.

CONTRACT
    viewpoint_<name>(ctx) -> list[float]
    One value per triple, in the SAME ORDER as ctx["triples"].

Every detector calls these, so the single-view and multi-view numbers are
computed by identical code and cannot drift apart.

DIRECTION declares which end of each scale is anomalous. It is written down
rather than remembered because getting it backwards silently turns a detector
into its own opposite -- AUC becomes 1-AUC and nothing raises an error.
"""
import collections
from pathlib import Path

import pandas as pd


def build_context(kg_path, model_dir=None, device="cpu"):
    """Load everything the viewpoints might need, exactly once.

    model_dir is optional: the neighbourhood viewpoint needs no model, and
    loading torch for it would be pure waste.
    """
    triples = [tuple(line.rstrip("\n").split("\t"))
               for line in open(kg_path, encoding="utf-8")]

    # N[x] = everything x links to, either direction, any relation. Ignoring
    # direction and relation type is deliberate: the question is "do these two
    # move in the same circles", not "is there a specific path".
    N = collections.defaultdict(set)
    for h, r, t in triples:
        N[h].add(t)
        N[t].add(h)

    ctx = {"triples": triples, "N": N, "model": None, "tf": None}

    if model_dir is not None:
        import torch
        from pykeen.triples import TriplesFactory
        ctx["tf"] = TriplesFactory.from_path(str(kg_path))
        _check_model_matches_graph(Path(model_dir), set(triples))
        model = torch.load(Path(model_dir) / "trained_model.pkl",
                           map_location=device, weights_only=False)
        ctx["model"] = model.to(device)
    return ctx


def _check_model_matches_graph(model_dir, kg_triples):
    """Refuse to score a graph the model was not trained on.

    Nothing else binds model/ to data/. Rerun 1_contaminate.py without
    rerunning 2_train.py and the old model scores a graph whose anomalies it
    never saw -- which is exactly the memorisation setup the whole protocol
    exists to avoid. Measured when it happened by accident: score-alone jumped
    to 90.6% precision and 100.0% recall, with no warning of any kind.

    It fails as a plausible number, not an obvious one, so it has to be an
    error rather than a printed caution.
    """
    from pykeen.triples import TriplesFactory

    saved_dir = model_dir / "training_triples"
    if not saved_dir.exists():
        raise SystemExit(f"{saved_dir} is missing -- cannot verify the model "
                         "was trained on this graph. Rerun 2_train.py.")
    tf = TriplesFactory.from_path_binary(str(saved_dir))
    i2e = {v: k for k, v in tf.entity_to_id.items()}
    i2r = {v: k for k, v in tf.relation_to_id.items()}
    trained_on = {(i2e[h], i2r[r], i2e[t])
                  for h, r, t in tf.mapped_triples.numpy().tolist()}

    missing = len(kg_triples - trained_on)
    extra = len(trained_on - kg_triples)
    if missing or extra:
        raise SystemExit(
            f"STALE MODEL: {model_dir} was trained on a different graph "
            f"({missing} triples in the data the model never saw, "
            f"{extra} the model saw that are not in the data).\n"
            "Rerun 2_train.py before detecting.")


def viewpoint_score(ctx):
    """What the trained MODEL thinks of the triple. LOW = anomalous.

    A fact that contradicts the rest of the graph cannot be fitted as well as
    one the graph supports, even though both were trained on as positives.
    """
    from pykeen.predict import predict_triples

    if ctx["model"] is None:
        raise RuntimeError("viewpoint 'score' needs a model -- "
                           "pass model_dir to build_context()")
    df = predict_triples(model=ctx["model"], triples=ctx["tf"]).process(
        factory=ctx["tf"]).df

    # Join on labels, never on row position: TriplesFactory may reorder or
    # deduplicate, and a positional join would silently mis-assign scores.
    lookup = {(h, r, t): s for h, r, t, s in zip(
        df["head_label"], df["relation_label"], df["tail_label"], df["score"])}
    missing = [x for x in ctx["triples"] if x not in lookup]
    if missing:
        raise RuntimeError(f"{len(missing)} triples got no score, "
                           f"e.g. {missing[0]}")
    return [lookup[x] for x in ctx["triples"]]


def viewpoint_neighbourhood(ctx):
    """Do the two endpoints already share connections? LOW = anomalous.

        chad locatedin africa   8 of chad's 9 links also touch africa -> 0.889
        chad locatedin europe   0 of chad's 9 links touch europe      -> 0.000

    Leave-one-out: the edge under test is dropped from both sides first, or a
    triple would appear in its own evidence and vouch for itself.

    No model involved. Pure counting.
    """
    N = ctx["N"]
    out = []
    for h, r, t in ctx["triples"]:
        nh = N[h] - {t}
        nt = N[t] - {h}
        out.append(len(nh & nt) / len(nh) if nh else 0.0)
    return out


#: name -> function. The dispatch table every detector binds to.
VIEWPOINTS = {
    "score": viewpoint_score,
    "neighbourhood": viewpoint_neighbourhood,
}

#: which viewpoints need a trained model (so callers know when to load one)
NEEDS_MODEL = {"score"}

#: -1 = LOW value is anomalous. +1 = HIGH value is anomalous.
DIRECTION = {
    "score": -1,
    "neighbourhood": -1,
}


def run(ctx, names=None):
    """Call the named viewpoints and hold every result in memory.

    Returns {name: [value per triple]}, all aligned to ctx["triples"].
    """
    names = list(VIEWPOINTS) if names is None else list(names)
    return {name: VIEWPOINTS[name](ctx) for name in names}


def to_rank(values, direction):
    """Position on a common 1..N scale, where rank 1 is the MOST anomalous.

    Ranking first is what makes viewpoints comparable at all: model scores run
    0.25-1.24 and neighbourhood support runs 0.0-1.0, so averaging the raw
    numbers would let one scale dominate for no principled reason.
    """
    return pd.Series(values).rank(ascending=(direction < 0))


def rank_average(results):
    """Average each triple's rank across viewpoints. LOW = anomalous.

    A triple only has to look bad to ONE viewpoint to sink; and a true fact
    that one viewpoint dislikes gets pulled back up by the others. Both halves
    matter -- the second is where the precision gain comes from.
    """
    ranks = {name: to_rank(vals, DIRECTION[name]) for name, vals in results.items()}
    return sum(ranks.values()) / len(ranks)
