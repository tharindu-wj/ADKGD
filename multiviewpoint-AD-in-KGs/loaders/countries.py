"""Where the Countries files are. Paths and names only -- no logic."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "countries"

NAME = "countries"

#: the clean graph, checked in so a run never depends on a download
SOURCE = ("train.txt", "valid.txt", "test.txt")

#: written by scripts/1_inject_anomalies.py
KG = DIR / "contaminated_kg.tsv"
TRUTH = DIR / "ground_truth.tsv"

#: one subfolder per embedding model, so several can coexist:
#: models/countries/distmult/, models/countries/complex/, ...
MODELS = ROOT / "models" / NAME
