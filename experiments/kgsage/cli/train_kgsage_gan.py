"""CLI entry point: python -m kgsage.cli.train_kgsage_gan

Trains the Phase 2 PAIR-AWARE KGSAGE Generator + Discriminator (the role-swap
contradiction generator). Heavy lifting lives in kgsage.gan.train_kgsage.

This is distinct from `python -m kgsage.cli.train_gan`, which trains the simple
single-slot-corruption GAN used by ADKGD's `--neg_source gan` baseline.
"""
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from kgsage.cli import _configure_utf8_stdout
_configure_utf8_stdout()

from kgsage.gan.train_kgsage import main

if __name__ == "__main__":
    raise SystemExit(main())
