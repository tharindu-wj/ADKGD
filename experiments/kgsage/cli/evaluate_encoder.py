"""CLI entry point: python -m kgsage.cli.evaluate_encoder

Runs Test 1.1 — filtered link prediction MRR on the test set of a known or
custom dataset. Pass criterion: MRR >= dataset's `expected_mrr` threshold
from kgsage.data.datasets (or the global default of 0.30 for unknown paths).
"""
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from kgsage.cli import _configure_utf8_stdout
_configure_utf8_stdout()

from kgsage.encoder.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
