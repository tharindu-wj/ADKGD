"""Fetch the published LibKGE ComplEx checkpoints + id-source datasets and run
the filtered-MRR loader gate (Option B stage B1).

Usage (from repo root, pytorch env):
    python -m kgsage.cli.fetch_lp --dataset fb15k237     # or wn18rr, or all

Downloads into experiments/kgsage/outputs/lp/ (gitignored except gate reports):
    <name>-complex.pt        the ICLR-2020 best-config checkpoint
    <name>/ (train/valid/test.txt)  the LibKGE archive the ids were built from
then runs lp_scorer's filtered-MRR gate against the published number.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))  # experiments/ on path -> kgsage importable

from kgsage.lp_scorer import ComplExScorer  # noqa: E402

BASE_MODELS = "http://web.informatik.uni-mannheim.de/pi1/iclr2020-models"
BASE_DATA = "http://web.informatik.uni-mannheim.de/pi1/kge-datasets"

# name -> (ckpt file, dataset archive stem, published test MRR)
SPECS = {
    "fb15k237": ("fb15k-237-complex.pt", "fb15k-237", 0.348),
    "wn18rr": ("wnrr-complex.pt", "wnrr", 0.475),
}


def _download(url: str, dst: Path) -> None:
    if dst.exists():
        print(f"  exists: {dst.name}")
        return
    print(f"  downloading {url}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst)  # noqa: S310 - fixed HTTP host, published artifacts


def fetch_and_gate(name: str, out_dir: Path, tolerance: float = 0.01) -> bool:
    ckpt_file, data_stem, expected = SPECS[name]
    ckpt = out_dir / ckpt_file
    tar = out_dir / f"{data_stem}.tar.gz"
    data_dir = out_dir / data_stem

    _download(f"{BASE_MODELS}/{ckpt_file}", ckpt)
    _download(f"{BASE_DATA}/{data_stem}.tar.gz", tar)
    if not data_dir.exists():
        print(f"  extracting {tar.name}")
        with tarfile.open(tar, "r:gz") as tf:
            tf.extractall(out_dir)  # noqa: S202 - trusted published archive

    scorer = ComplExScorer.from_libkge(ckpt, data_dir)
    print(f"  loaded: n_ent={scorer.ent_emb.shape[0]} n_rel={scorer.n_rel_base} "
          f"reciprocal={scorer.reciprocal}")
    res = scorer.filtered_mrr(data_dir, split="test")
    ok = abs(res["mrr"] - expected) <= tolerance
    print(f"  GATE {'PASS' if ok else 'FAIL'} [{name}]: "
          f"mrr={res['mrr']:.4f} expected={expected}±{tolerance} "
          f"hits@10={res['hits@10']:.4f}")
    report = out_dir / f"gate_{name}.json"
    import json
    report.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="all", choices=[*SPECS, "all"])
    ap.add_argument("--out_dir", default=str(_HERE.parents[1] / "outputs" / "lp"))
    args = ap.parse_args()

    names = list(SPECS) if args.dataset == "all" else [args.dataset]
    ok = True
    for name in names:
        print(f"[{name}]")
        ok = fetch_and_gate(name, Path(args.out_dir)) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
