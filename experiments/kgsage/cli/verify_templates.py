"""CLI entry point: python -m kgsage.cli.verify_templates

v2-A decision gate: verify the rule-mined partner-template miner reproduces the
trusted Test 1.3 anti-symmetric classification on a given KG. Heavy lifting
lives in kgsage.gan.verify_partner_templates.
"""
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from kgsage.cli import _configure_utf8_stdout
_configure_utf8_stdout()

from kgsage.gan.verify_partner_templates import main

if __name__ == "__main__":
    raise SystemExit(main())
