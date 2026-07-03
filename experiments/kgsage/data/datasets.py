"""Known-dataset registry + dataset resolution.

A convenience lookup that maps a short dataset name to its directory. The
pattern to add a dataset:

    1. Drop train.txt / valid.txt / test.txt into data/<NAME>/
       (Tab-separated:  head_string<TAB>relation_string<TAB>tail_string)

    2. (Optional) add an entry to KNOWN_DATASETS below so the short name works.

    3. Use the dataset directory with the GAN trainer / ADKGD run, e.g.:
          python -m kgsage.cli.train_gan --data data/<NAME> --out <ckpt>.pt

For one-off datasets that don't need a registry entry, pass a filesystem path
directly. `resolve_dataset()` distinguishes names from paths and returns a
unified config dict either way. Training hyperparameters are CLI flags on the
trainer, not stored here.
"""
import os


KNOWN_DATASETS = {
    # Only datasets that exist on disk in this checkout are registered.
    # (nell995/kinship/yago/kg20c entries were removed: their directories are
    # not in the repo and resolve_dataset would silently "succeed" with a
    # 0-triple KG. The YAGO converter was deleted by the dev_gan_1 cleanup;
    # recover it from commit 982cb77 if that dataset returns.)
    "fb15k237":   {"default_path": "data/FB15K-237",  "n_relations": 237},
    "wn18rr":     {"default_path": "data/WN18RR",     "n_relations": 11},
    "fb15k_mini": {"default_path": "data/FB15K-mini", "n_relations": 213},
    "dummy_kg":   {"default_path": "data/dummy_kg",   "n_relations": 3},
}


def resolve_dataset(name_or_path):
    """Look up a dataset by short name or treat it as a filesystem path.

    Returns a unified config dict regardless of input form:
      {
        "name"        : short name (or basename of the path)
        "path"        : directory containing train.txt etc.
        "n_relations" : known or None (discovered at load time)
      }

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
        return {
            "name": os.path.basename(name_or_path.rstrip(os.sep)),
            "path": name_or_path,
            "n_relations": None,  # discovered at load time
        }

    # ─── Case 3: neither — give a useful error message ────────────────
    raise ValueError(
        f"Unknown dataset '{name_or_path}'.\n"
        f"  Known short names: {sorted(KNOWN_DATASETS.keys())}\n"
        f"  Custom paths must point to a directory containing "
        f"train.txt / valid.txt / test.txt."
    )
