"""CLI entry point: python -m kgsage.cli.audit_embeddings

Runs Test 1.2 — Mann-Whitney U test on the cosine similarities of symmetric
vs anti-symmetric relation embeddings from a trained encoder checkpoint.

Consumes the JSON output of `audit_dataset` (Test 1.3) to choose which
relation pairs to compare. Pass criterion: anti-symmetric pairs have
statistically lower cosine similarity than symmetric pairs (p < 0.05).
"""
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from kgsage.cli import _configure_utf8_stdout
_configure_utf8_stdout()

from kgsage.encoder.audit_embeddings import main

if __name__ == "__main__":
    raise SystemExit(main())
