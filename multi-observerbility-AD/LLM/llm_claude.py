"""The Claude backend - drives the loop with your Claude subscription, no API key.

HOW IT CONNECTS
---------------
`claude -p` ("print mode") runs one prompt and exits, using the Claude Code login
you already have. We send the prompt via stdin (avoids Windows quoting problems)
and read the reply from stdout. Each call costs one CLI startup (~a few seconds)
plus model time, and counts against your subscription's usage window.

THE CONTRACT
------------
Same as every backend (see llm_dummy.py for the full statement):

    claude_llm(messages) -> {"thinking", "tool", "args"}  or  {"thinking", "final_spec"}

Note there is no provider "tool use" API anywhere here. The agent loop owns the
tools; the model just says which one it wants, as JSON text. The SYSTEM_PROMPT
below is what teaches it the two shapes -- that prompt IS the integration.
"""

import glob
import json
import os
import re
import shutil
import subprocess

from LLM.build_system_prompt import build_prompt

#: Which model the CLI should use. Aliases work: "haiku" (fast/cheap),
#: "sonnet" (good default), "opus" (strongest).
CLAUDE_CLI_MODEL = "sonnet"


def find_claude_cli():
    """Return the path to the claude CLI, wherever this machine keeps it.

    Checked in order:
      1. `claude` on the PATH (standalone install -- the durable setup)
      2. the native installer's location under ~/.local/bin
      3. the newest copy bundled inside the VS Code extension

    If you install the standalone CLI later (https://claude.com/claude-code),
    option 1 will simply take over -- nothing here needs to change.
    """
    on_path = shutil.which("claude")
    if on_path:
        return on_path

    native = os.path.expanduser("~/.local/bin/claude.exe")
    if os.path.exists(native):
        return native

    # The VS Code extension ships its own claude.exe. Several versions may be
    # installed side by side; take the most recently modified one.
    pattern = os.path.expanduser(
        "~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude.exe"
    )
    bundled = glob.glob(pattern)
    if bundled:
        return max(bundled, key=os.path.getmtime)

    raise FileNotFoundError(
        "Could not find the claude CLI. Install Claude Code, or add `claude` to PATH."
    )


def claude_llm(messages):
    """Ask Claude for the next move. See the module docstring for the contract."""
    prompt = build_prompt(messages)   # shared with every backend -- see build_system_prompt.py

    # Full path + list argv: no shell, so Windows quoting can never bite us.
    # The prompt goes in via stdin for the same reason.
    result = subprocess.run(
        [find_claude_cli(), "-p", "--model", CLAUDE_CLI_MODEL],
        input=prompt, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {result.returncode}). Is Claude Code installed "
            f"and logged in? stderr:\n{result.stderr[:500]}"
        )

    # The model was told "JSON only", but be tolerant: grab from the first '{'
    # to the last '}' so stray prose or ```json fences don't break parsing.
    match = re.search(r"\{.*\}", result.stdout, re.DOTALL)
    if match is None:
        raise RuntimeError(f"No JSON in the model's reply:\n{result.stdout[:500]}")
    return json.loads(match.group(0))
