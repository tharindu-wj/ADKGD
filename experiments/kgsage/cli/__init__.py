"""KGSAGE command-line entry points.

Each CLI shim is a thin (5-line) wrapper around a function in the underlying
module. Library users call the modules directly (e.g. `kgsage.encoder.train`);
shell users call the CLI shims (e.g. `python -m kgsage.cli.train_encoder`).

Available commands:
  python -m kgsage.cli.audit_dataset      - Test 1.3 (dataset-level anti-symmetric audit)
  python -m kgsage.cli.train_encoder      - train Phase 1 encoder (RGCN + DistMult)
  python -m kgsage.cli.evaluate_encoder   - Test 1.1 (link prediction MRR)
  python -m kgsage.cli.audit_embeddings   - Test 1.2 (anti-symmetric signal in embeddings)
  python -m kgsage.cli.train_gan          - train the pair-aware KGSAGE role-swap GAN
                                            (KGSAGEGenerator + KGSAGEDiscriminator)
"""
import sys


def _configure_utf8_stdout():
    """Force UTF-8 stdout so report Unicode chars print correctly on Windows.

    Windows' cp1252 default encoding cannot encode the box-drawing characters
    and special symbols used in pretty-printed audit reports. This silently
    reconfigures stdout/stderr to UTF-8 if the runtime supports it.

    No-op on Linux/Mac where stdout is already UTF-8.
    No-op if the runtime doesn't support reconfigure (very old Python; pipes).
    Each CLI shim calls this once at import time.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError, OSError):
        # AttributeError: Python < 3.7 (not targeted; be safe anyway).
        # ValueError/OSError: stdout was replaced by a non-reconfigurable stream
        # (test runners, IDE consoles, redirected pipes — none need the fix).
        pass
