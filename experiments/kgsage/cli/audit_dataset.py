"""CLI entry point: python -m kgsage.cli.audit_dataset

Thin wrapper around kgsage.data.audit_dataset.main(). Heavy lifting lives in
the module; this file exists so the command line invocation is one short
string rather than a long module path.

Adds `experiments/` to sys.path so `import kgsage` works without pip-installing
the package. When KGSAGE is eventually released as `pip install kgsage`, this
sys.path manipulation becomes unnecessary.
"""
import sys
from pathlib import Path

# Make `import kgsage` work whether invoked from repo root or from any cwd.
_EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]  # -> experiments/
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from kgsage.cli import _configure_utf8_stdout
_configure_utf8_stdout()  # fix Windows cp1252 stdout for the report Unicode chars

from kgsage.data.audit_dataset import main

if __name__ == "__main__":
    raise SystemExit(main())
