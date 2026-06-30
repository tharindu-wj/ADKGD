"""Shared helpers for the KGSAGE learning probes (rung1 .. rung7).

Each probe is a tiny standalone script you run to SEE one part of the pipeline
work — rather than read an aggregate metric. Run them in order:

    python experiments/kgsage/probes/rung1_data.py
    python experiments/kgsage/probes/rung2_antisymmetry.py
    ...

They auto-detect a dataset + matching checkpoints:
  * if data/FB15K-237 exists      -> use it (the real run)
  * otherwise                     -> fall back to data/dummy_kg (always present)
Override the dataset by passing a directory as the first argument:
    python experiments/kgsage/probes/rung1_data.py data/WN18RR

Probes that need the ENCODER (rung3, rung4) require torch_geometric and the
encoder checkpoint — they print a clear [skip] message if those are absent
(e.g. on a laptop without PyG) and are meant to be run on the HPC.
"""
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[2]          # .../experiments
REPO = _EXP.parent                                  # repo root
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

try:                                                # readable output on Windows cp1252
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_ENC = REPO / "experiments/kgsage/outputs/fb15k237_encoder.pt"
_GAN_FB = REPO / "experiments/kgsage/outputs/checkpoints/kgsage_fb15k237.pt"
_GAN_DUMMY = REPO / "experiments/kgsage/outputs/checkpoints/kgsage_dummy.pt"


def banner(title):
    print("=" * 72)
    print("  " + title)
    print("=" * 72)


def hint(msg):
    print()
    print(">> LOOK FOR: " + msg)


def resolve_bundle():
    """Return (data_dir, encoder_ckpt|None, gan_ckpt|None) — paths only, no load."""
    override = sys.argv[1] if len(sys.argv) > 1 else None
    if override:
        data_dir = override
    elif (REPO / "data" / "FB15K-237").is_dir():
        data_dir = str(REPO / "data" / "FB15K-237")
    elif (REPO / "data" / "dummy_kg").is_dir():
        data_dir = str(REPO / "data" / "dummy_kg")
    else:
        raise SystemExit("No dataset under data/. Pass a dataset directory as arg 1.")

    enc = str(_ENC) if _ENC.is_file() else None
    gan = str(_GAN_FB) if _GAN_FB.is_file() else (str(_GAN_DUMMY) if _GAN_DUMMY.is_file() else None)
    return data_dir, enc, gan


def load_dataset():
    """Resolve the bundle AND load the KG. Returns (kg, data_dir, enc, gan)."""
    from kgsage.data.loaders import load_kg
    data_dir, enc, gan = resolve_bundle()
    print(f"[probe] dataset    : {data_dir}")
    print(f"[probe] encoder ckpt: {enc or '(none — rung3/4 will skip)'}")
    print(f"[probe] gan ckpt    : {gan or '(none — train one first)'}")
    return load_kg(data_dir), data_dir, enc, gan


def is_dummy(data_dir):
    return "dummy" in Path(data_dir).name.lower()
