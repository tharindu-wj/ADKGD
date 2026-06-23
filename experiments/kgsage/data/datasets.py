"""Known-dataset registry + dataset resolution.

This file is the ONE place to extend KGSAGE to new datasets. The pattern:

    1. Drop train.txt / valid.txt / test.txt into data/<NAME>/
       (Tab-separated:  head_string<TAB>relation_string<TAB>tail_string)

    2. Add an entry to KNOWN_DATASETS below with that dataset's recommended
       hyperparameters and decision-gate thresholds.

    3. Use the dataset by its short name everywhere:
          python -m kgsage.cli.audit_dataset --dataset <name>
          python -m kgsage.cli.train_encoder --dataset <name>
          python -m kgsage.cli.audit_embeddings --dataset <name> --ckpt ...

For one-off datasets that don't need a registry entry, pass a filesystem path
directly. `resolve_dataset()` distinguishes names from paths and returns a
unified config dict either way.

WHY DIFFERENT DEFAULTS PER DATASET:
  - FB15K-237: 237 relations → 30 RGCN bases is a good compression ratio.
  - WN18RR:    only 11 relations → 30 bases would be overparameterised; use 11.
  - WN18RR is also smaller and sparser → train more epochs to compensate.
  - NELL-995:  200 relations → similar to FB15K-237; same defaults work.
  - dummy_kg:  tiny smoke-test fixture → small dim, short epochs.

These defaults are tuned to match the encoder paper baselines (RGCN +
DistMult typically reports MRR ~0.30 on FB15K-237, ~0.45 on WN18RR).
"""
import os


# Generic defaults — used for custom paths that aren't in the registry.
_GENERIC_DEFAULTS = {
    "epochs": 200,
    "dim": 200,
    "n_layers": 2,
    "num_bases": 30,
    "batch_size": 2048,
    "lr": 1e-3,
    "margin": 1.0,
    "eval_every": 10,
    "expected_mrr": None,        # no MRR expectation for unknown datasets
    "antisym_min_pairs": None,   # no anti-sym audit threshold either
}


KNOWN_DATASETS = {
    # Freebase 15K-237 — Toutanova-Chen 2015 cleaned variant of FB15K.
    # The primary thesis dataset.
    "fb15k237": {
        "default_path": "data/FB15K-237",
        "n_relations": 237,
        # Encoder training defaults
        "epochs": 200,
        "dim": 200,
        "n_layers": 2,
        "num_bases": 30,
        "batch_size": 2048,
        "lr": 1e-3,
        "margin": 1.0,
        "eval_every": 10,
        # Decision-gate thresholds (override the global defaults)
        "expected_mrr": 0.30,        # Pass: MRR >= this
        "antisym_min_pairs": 30,     # Pass: at least this many anti-sym pairs
    },

    # WordNet 18 — RR variant. Smaller, sparser, only 11 relations.
    # Used as a sanity check and potential fallback if FB15K-237 fails Test 1.3.
    "wn18rr": {
        "default_path": "data/WN18RR",
        "n_relations": 11,
        "epochs": 300,               # sparser data → more epochs
        "dim": 200,
        "n_layers": 2,
        "num_bases": 11,             # only 11 relations; no benefit from >11 bases
        "batch_size": 1024,
        "lr": 1e-3,
        "margin": 1.0,
        "eval_every": 20,
        "expected_mrr": 0.45,        # WN18RR usually reports higher MRR than FB
        "antisym_min_pairs": 3,      # few relations → few possible pairs
    },

    # NELL-995 — h25 variant from Xiong et al. 2017.
    "nell995": {
        "default_path": "data/NELL-995",
        "n_relations": 200,
        "epochs": 200,
        "dim": 200,
        "n_layers": 2,
        "num_bases": 30,
        "batch_size": 2048,
        "lr": 1e-3,
        "margin": 1.0,
        "eval_every": 10,
        "expected_mrr": 0.40,
        "antisym_min_pairs": 30,
    },

    # Tiny synthetic fixture — quick smoke test.
    # Doesn't need to pass any decision gates; it's only for verifying
    # the pipeline plumbs together end-to-end.
    "dummy_kg": {
        "default_path": "data/dummy_kg",
        "n_relations": 3,
        "epochs": 20,
        "dim": 32,
        "n_layers": 2,
        "num_bases": 3,
        "batch_size": 64,
        "lr": 1e-3,
        "margin": 1.0,
        "eval_every": 5,
        "expected_mrr": 0.0,         # don't bother checking on dummy
        "antisym_min_pairs": 0,
    },
}


def resolve_dataset(name_or_path):
    """Look up a dataset by short name or treat it as a filesystem path.

    Returns a unified config dict regardless of input form:
      {
        "name"             : short name (or basename of the path)
        "path"             : directory containing train.txt etc.
        "n_relations"      : known or None
        "epochs"           : recommended default
        "dim"              : recommended default
        "n_layers"         : recommended default
        "num_bases"        : recommended default
        "batch_size"       : recommended default
        "lr"               : recommended default
        "margin"           : recommended default
        "eval_every"       : recommended default
        "expected_mrr"     : pass threshold for Test 1.1 (None = no check)
        "antisym_min_pairs": pass threshold for Test 1.3 (None = no check)
      }

    CLI args still override anything in this dict; resolve_dataset() supplies
    defaults, not constraints.

    Inputs:
      name_or_path : short name (like "fb15k237") or filesystem path

    Raises:
      ValueError if the input is neither a known name nor an existing directory.
    """
    # ─── Case 1: known short name ──────────────────────────────────────
    if name_or_path in KNOWN_DATASETS:
        config = dict(KNOWN_DATASETS[name_or_path])  # shallow copy
        config["name"] = name_or_path
        config["path"] = config.pop("default_path")
        return config

    # ─── Case 2: filesystem path ───────────────────────────────────────
    if os.path.isdir(name_or_path):
        config = dict(_GENERIC_DEFAULTS)
        config["name"] = os.path.basename(name_or_path.rstrip(os.sep))
        config["path"] = name_or_path
        config["n_relations"] = None  # discovered at load time
        return config

    # ─── Case 3: neither — give a useful error message ────────────────
    raise ValueError(
        f"Unknown dataset '{name_or_path}'.\n"
        f"  Known short names: {sorted(KNOWN_DATASETS.keys())}\n"
        f"  Custom paths must point to a directory containing "
        f"train.txt / valid.txt / test.txt."
    )
