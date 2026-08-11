"""The dummy LLM backend - a scripted stand-in, fully offline, always deterministic.

THE CONTRACT (every backend file implements exactly this)
---------------------------------------------------------
A backend is one function:  llm(messages) -> dict

`messages` is the conversation so far, a list of dicts with "role" and "content".
The returned dict must be ONE of two shapes:

    {"thinking": "...", "tool": "<name>", "args": {...}}    "run this tool for me"
    {"thinking": "...", "final_spec": {...}}                "I'm done, here is my viewpoint"

That contract is one of the project's two integration surfaces (the other is the
tool contract in tools/registry.py): the agent loop in orchestrator_custom.py --
and the LangChain orchestration beside it -- neither knows nor cares which backend
produced the reply. Adding a new backend (Ollama, OpenAI, anything) means writing
one new llm_<name>.py in this folder with one function of this shape.

WHY KEEP THE DUMMY AT ALL, NOW THAT A REAL BACKEND EXISTS
---------------------------------------------------------
It is the regression test. It runs with no network, no login and no randomness,
so if the loop breaks, the dummy tells you the LOOP broke -- not the model, not
the API, not your internet connection. Run it before blaming anything else.

HOW IT WORKS
------------
A real LLM reads `messages` and decides. The dummy only counts how many tool
results have come back so far and plays the next line of a fixed script. The
script includes one deliberate MISTAKE (a wrong column name) so learners can
watch the agent read the error text and correct itself -- a real model does
that genuinely; a script can only have the correction written in.
"""

DUMMY_SCRIPT = [
    {
        "thinking": "I should first find out what data exists.",
        "tool": "list_columns", "args": {},
    },
    {
        "thinking": ("The goal mentions impossible households. AveOccup (people per "
                     "household) sounds central -- let me check its range."),
        "tool": "describe_column", "args": {"name": "AveOccup"},
    },
    {
        "thinking": "Rooms per household also matters for 'impossible'. Checking it.",
        "tool": "describe_column", "args": {"name": "Rooms"},   # <-- wrong on purpose
    },
    {
        "thinking": "That errored -- the correct name is AveRooms. Trying again.",
        "tool": "describe_column", "args": {"name": "AveRooms"},
    },
    {
        "thinking": ("AveOccup reaches 1243 against a 99th percentile of 5.4, and "
                     "AveRooms reaches 141 against 10.4 -- impossible households exist. "
                     "First attempt: score all household columns together."),
        "tool": "run_lof_per_viewpoint",
        "args": {"columns": ["AveRooms", "AveBedrms", "AveOccup", "Population"]},
    },
    {
        "thinking": ("Those top rows are genuinely impossible (households of 1200+). "
                     "But Population measures SIZE, not consistency -- my goal is about "
                     "ratios that contradict each other. Loop: try ratios only."),
        "tool": "run_lof_per_viewpoint",
        "args": {"columns": ["AveRooms", "AveBedrms", "AveOccup"]},
    },
    {
        "thinking": ("Both runs surface the same impossible-household blocks, so the "
                     "simpler three-ratio viewpoint is enough. Finalising."),
        "final_spec": {
            "observer": "census-quality-auditor",
            "goal": "find census rows that cannot describe a real place",
            "columns": ["AveRooms", "AveBedrms", "AveOccup"],
            "row_filter": None,
            "why": ("These three are ratios over the same denominator (households), "
                    "so a block whose ratios disagree cannot be a real place. "
                    "Population was dropped: block size is not a consistency signal."),
        },
    },
]


def dummy_llm(messages):
    """Return the next scripted step. See the module docstring for the contract."""
    steps_done = sum(1 for m in messages if m["role"] == "tool_result")
    return DUMMY_SCRIPT[steps_done]
