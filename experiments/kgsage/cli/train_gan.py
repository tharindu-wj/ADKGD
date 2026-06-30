"""CLI entry point: python -m kgsage.cli.train_gan

Trains the KGSAGE GAN (KGSAGEGenerator + KGSAGEDiscriminator) that generates
single-slot-corruption negatives. Heavy lifting lives in kgsage.gan.train.
"""
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from kgsage.cli import _configure_utf8_stdout
_configure_utf8_stdout()

from kgsage.gan.train import main

if __name__ == "__main__":
    raise SystemExit(main())
