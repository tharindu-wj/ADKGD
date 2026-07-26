"""KGSAGE command-line tools (run from repo root with PYTHONPATH=experiments).

The current flow, tool by tool:
  python -m kgsage.gan.train
      Phase 2 - Adversarial Generator Training (dual-discriminator
      architecture, one snapshot per epoch).
  python -m kgsage.cli.knockout_eval
      Snapshot SELECTION by knockout J@10: lower = more anchor-specific.
  python -m kgsage.cli.gen_corruptions_csv
      Phase 3 - Corruption Generation: the stage-1 evaluation CSV of
      corruptions from a promoted generator (feeds 7.3 and 7.4).
  python -m kgsage.cli.ego_from_csv
      One ego-graph figure per CSV row (7.4).
  python -m kgsage.cli.format_for_llm
      Paste-ready triple blocks, corrupted + control (7.3).
  python -m kgsage.cli.gen_neighbourhood_context
      Neighbourhood-context case blocks for the contradiction judgement
      (7.4 semantic).
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
