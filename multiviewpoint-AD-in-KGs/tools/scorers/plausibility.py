"""What the trained embedding model thinks of a triple. LOW = anomalous.

A fact that contradicts the rest of the graph cannot be fitted as well as one
the graph supports, even though both were trained on as positives.

Evidence: a KGE model, trained by scripts/2_train_plausibility_scorer.py on this same graph.
"""
from pathlib import Path

NAME = "plausibility"
NEEDS_MODEL = True
DIRECTION = -1          # a LOW value is anomalous


def score(triples, model_dir=None, device="cpu", kg_path=None, **params):
    import torch
    from pykeen.predict import predict_triples
    from pykeen.triples import TriplesFactory

    if model_dir is None or kg_path is None:
        raise ValueError("plausibility needs model_dir and kg_path")
    model_dir = Path(model_dir)
    _check_model_matches_graph(model_dir, set(triples))

    tf = TriplesFactory.from_path(str(kg_path))
    model = torch.load(model_dir / "trained_model.pkl",
                       map_location=device, weights_only=False).to(device)
    df = predict_triples(model=model, triples=tf).process(factory=tf).df

    # Join on labels, never on row position. TriplesFactory reorders: measured,
    # 1272 of 1273 rows move, so a positional join mis-assigns nearly every score.
    lookup = {(h, r, t): s for h, r, t, s in zip(
        df["head_label"], df["relation_label"], df["tail_label"], df["score"])}
    missing = [x for x in triples if x not in lookup]
    if missing:
        raise RuntimeError(f"{len(missing)} triples got no score, e.g. {missing[0]}")
    return [lookup[x] for x in triples]


def _check_model_matches_graph(model_dir, kg_triples):
    """Refuse to score a graph the model was not trained on.

    Nothing else binds models/ to data/. Re-run 1_inject_anomalies.py without
    re-running 2_train_plausibility_scorer.py and the old model scores a graph whose anomalies it
    never saw -- the memorisation setup the whole protocol exists to avoid.
    Measured when it happened by accident: precision 90.6%, recall 100.0%,
    with no warning. It fails as a plausible number, not an obvious one.
    """
    from pykeen.triples import TriplesFactory

    saved = model_dir / "training_triples"
    if not saved.exists():
        raise SystemExit(f"{saved} is missing -- cannot verify the model was "
                         "trained on this graph. Re-run scripts/2_train_plausibility_scorer.py.")
    tf = TriplesFactory.from_path_binary(str(saved))
    i2e = {v: k for k, v in tf.entity_to_id.items()}
    i2r = {v: k for k, v in tf.relation_to_id.items()}
    trained_on = {(i2e[h], i2r[r], i2e[t])
                  for h, r, t in tf.mapped_triples.numpy().tolist()}

    missing, extra = len(kg_triples - trained_on), len(trained_on - kg_triples)
    if missing or extra:
        raise SystemExit(
            f"STALE MODEL: {model_dir} was trained on a different graph "
            f"({missing} triples the model never saw, {extra} it saw that are "
            "not in the data).\nRe-run scripts/2_train_plausibility_scorer.py before detecting.")
