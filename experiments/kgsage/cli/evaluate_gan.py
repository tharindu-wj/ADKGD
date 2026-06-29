"""CLI entry point: python -m kgsage.cli.evaluate_gan

Phase 3 generation-quality evaluation of a trained KGSAGE pair-aware GAN:
compares KGSAGE's generated contradictions against the rule-only baseline
(precision / recall / diversity + how much they actually differ).
Heavy lifting lives in kgsage.gan.evaluate.
"""
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from kgsage.cli import _configure_utf8_stdout
_configure_utf8_stdout()

from kgsage.gan.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
