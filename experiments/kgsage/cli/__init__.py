"""KGSAGE command-line tools (run from repo root with PYTHONPATH=experiments).

The current flow, tool by tool:
  python -m kgsage.gan.train                    - train (dual-discriminator,
                                                  per-epoch snapshots)
  python -m kgsage.cli.knockout_eval            - snapshot SELECTION by
                                                  anchor-knockout J@10
  python -m kgsage.cli.gen_corruptions_csv      - stage-1 evaluation CSV from a
                                                  locked generator (7.3 + 7.4)
  python -m kgsage.cli.ego_from_csv             - ego graphs per CSV row (7.4)
  python -m kgsage.cli.ego_viz                  - single ego-graph renderer
  python -m kgsage.cli.fetch_lp                 - download the LP auditor
                                                  checkpoints + MRR gate
  python -m kgsage.cli.inspect_gan_lp           - LP score-gap diagnostics for
                                                  generated negatives
"""
import sys


def _configure_utf8_stdout():
    """Force UTF-8 stdout so report Unicode chars print correctly on Windows.

    No-op on Linux/Mac where stdout is already UTF-8, and wherever the
    runtime does not support reconfigure (test runners, redirected pipes).
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError, OSError):
        pass
