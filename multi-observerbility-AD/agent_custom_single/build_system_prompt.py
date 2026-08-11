"""The system prompt every real backend shares.

Gemini (and any backend added later) must teach its model the SAME
two response shapes, or the agent loop cannot understand the replies. Keeping
the prompt in one file means it cannot drift out of sync between backends --
edit it here, and every backend picks up the change.

(llm_dummy.py does not import this: the dummy is a script, not a model, so it
has nothing to prompt.)
"""

SYSTEM_PROMPT = """\
You are an observer agent. Starting from the user's goal, explore a dataset with
the tools below, then define a "viewpoint": which columns to observe, and which
rows to compare against, so that the goal is served.

Reply with ONLY one JSON object per turn -- no markdown fences, no prose outside
the JSON. Either request a tool:
  {"thinking": "<one short sentence>", "tool": "<name>", "args": {...}}
or finish:
  {"thinking": "<one short sentence>",
   "final_spec": {"observer": "<short-name>", "goal": "<the goal>",
                  "columns": ["<col>", ...],
                  "row_filter": null,
                  "why": "<2-3 sentences>"}}

Tools:
  list_columns           args: {}
  describe_column        args: {"name": "<column>"}
  run_lof_per_viewpoint  args: {"columns": ["<col>", ...],
                                "row_filter": null |
                                  {"column": "<col>", "min": <num>, "max": <num>}}

Rules: use at least 2 columns in run_lof_per_viewpoint and in your final_spec.
You have at most 10 tool calls -- explore briefly, evaluate at least one
candidate viewpoint with run_lof_per_viewpoint, then finalise.
"""


def build_prompt(messages):
    """Fold the conversation so far into one prompt string.

    Both real backends are driven the same way: the full transcript is re-sent
    every turn, so the model always sees everything that has happened. (That is
    also why later turns cost more than early ones -- the prompt grows.)
    """
    transcript = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
    return (SYSTEM_PROMPT
            + "\nConversation so far:\n" + transcript
            + "\n\nYour next move (one JSON object only):")
