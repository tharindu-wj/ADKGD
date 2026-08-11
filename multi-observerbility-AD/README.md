# Observer agent MVP — multi-viewpoint anomaly detection

An agent reads a one-sentence goal, explores a dataset through tools, and decides
**where to look** — which columns, which rows. Deterministic code then decides
**how unusual each entity is** (Local Outlier Factor). The agent never scores;
the statistics never choose. That split is the whole idea.

```
user sets goal ──> AGENT (LLM backend) ──> TOOLS (plain functions) ──> FINAL SPEC
                        ^                        |                        |
                        └── loop: evaluate ──────┘                        v
                            and ask again                 runs/run_<time>_<backend>.json
```

## Quick start

```bash
# Gemini is the default — just give a goal (quotes optional):
python agent_custom_single/orchestrator_custom.py "find neighbourhoods that do not fit their region"
python agent_custom_single/orchestrator_custom.py find blocks whose housing looks impossible

python agent_custom_single/orchestrator_custom.py --dummy     # offline scripted regression test — free, deterministic
```

With no goal given, the default goal is used: *"find census rows that cannot
describe a real place"*. `--dummy` always replays its fixed script and cannot
react to a custom goal; it exists to prove the **loop** still works without
spending quota or needing a network.

> Running with no arguments now **calls the API** against your free-tier quota —
> including VS Code's ▶ Run button, which passes no arguments. The banner's first
> line and the run filename (`_gemini` / `_dummy`) always tell you which ran.

> **If it is at the root you run it; if it is in a folder you import it.**
> `python tools/run_lof_per_viewpoint.py` fails — those files are libraries, not entry points.

## The files

| File | What it is |
|---|---|
| `agent_custom_single/` | **Agent 1 — the hand-written loop.** Owns its LLM backends: `orchestrator_custom.py` (entry point), `build_system_prompt.py`, `llm_dummy.py` (**read this first** — it states the backend contract), `llm_gemini.py` |
| `agent_adk_single/` | **Agent 2 — Google ADK.** `agent.py` defines `root_agent`; run with `adk run` / `adk web` |
| `agent_adk_multiple/` | **Agent 3 — planned.** ADK `ParallelAgent`: two observers with isolated branches |
| `tools/registry.py` | The tool index: name → function. **Read this second**: it is what both orchestrations bind to |
| `tools/<name>.py` | One file per tool, and nothing else |
| `data/active.py` | The dataset switch: tools import from here, so changing datasets is one import line |
| `data/california_housing.py` | The reference dataset (`NAME`, `ENTITY`, `DATA`, `COLUMN_MEANINGS`). A leaf — imports nothing from the project, loaded once and shared |
| `utils/save_run.py` | Writes `runs/*.json`. Lives outside the orchestrators so both write an identical schema |
| `baseline/multiviewpoint.ipynb` | The hand-built baseline viewpoints (per-viewpoint LOF on California housing) that agent-derived ones are compared against |
| `runs/` | One JSON per run: the final spec **plus the full agent trace** |
| `.env` | Your Gemini key (`GEMINI_KEY=...`) — gitignored, never commit it |

## How it works

**The tools** (in `tools/`, one file each) are plain functions that return **strings** —
text in, text out, because that is what a language model actually consumes.
Errors come back as `ERROR: ...` text too, so the agent can read them and correct
itself instead of crashing:

| Tool | File | Job |
|---|---|---|
| `list_columns()` | `tools/list_columns.py` | What data exists — names and meanings |
| `describe_column(name)` | `tools/describe_column.py` | Distribution of one column (median, p99, max) so the agent learns the scales |
| `run_lof_per_viewpoint(columns, row_filter)` | `tools/run_lof_per_viewpoint.py` | The statistical component: standardise → LOF(k=20) → score summary + top-5 rows |

**The loop** (`derive_viewpoint`) is ~30 lines: ask the backend what to do, run
the tool it names, feed the text result back, repeat — until the backend returns
a `final_spec` or `max_steps` runs out. Backend crashes and exhausted runs still
save their trace (`"status": "exhausted"`): a failed run is data, not garbage.

**The backend contract** — the entire integration surface — is one function per
backend returning one of two JSON shapes:

```python
{"thinking": "...", "tool": "<name>", "args": {...}}   # "run this tool for me"
{"thinking": "...", "final_spec": {...}}               # "I'm done — here is my viewpoint"
```

Adding a backend (Ollama, OpenAI, ...) = one new `agent_custom_single/llm_<name>.py` with one
function of that shape, plus one line in the `BACKENDS` dict in
`orchestrator_custom.py`.
No provider "tool use" API is needed anywhere — the loop owns the tools; the
model just names the one it wants.

## Backend setup

**Dummy** — nothing. It replays a fixed script (including one deliberate
wrong-column mistake, so you can watch the error-correction behaviour). It is
the regression test: if the dummy breaks, the *loop* broke, not a model.

**Gemini** — put `GEMINI_KEY=<your key>` in `.env` in the repo root (or set
`GEMINI_API_KEY` in the environment); free keys at aistudio.google.com/apikey.
The model is **pinned to `gemini-3.5-flash-lite`**, learned the hard way: the
`-latest` alias resolved to a model whose free tier allows only 20 requests per
*day* (~3 agent runs). Free quotas are per model, and the lite models have much
larger buckets. Rate-limit 429s mid-run are still normal; the backend reads
Google's `retryDelay` hint and waits, up to 3 retries.

## What a run file contains

```json
{
  "run_id": "20260808_203857",
  "backend": "gemini",
  "goal": "find census rows that cannot describe a real place",
  "status": "completed",            // or "exhausted" — ran out of steps / crashed
  "steps_taken": 6,
  "final_spec": { "observer": "...", "columns": [...], "row_filter": null, "why": "..." },
  "trace": [ {"step": 1, "thinking": "...", "tool": "...", "args": {}, "result": "..."} ]
}
```

The console truncates tool results for readability; the trace never does. One
file per run, never overwritten — so comparing runs (same goal, different
backends or repeats) is just reading `runs/`. Replaying a saved viewpoint needs
no LLM — and no backend either, since `tools/` imports nothing from any agent:

```python
from tools.run_lof_per_viewpoint import run_lof_per_viewpoint
run_lof_per_viewpoint(spec["columns"], spec["row_filter"])
```

## Observations so far (why the trace matters)

Same goal, same data, same prompt — different runs derive **different
viewpoints**, each with a coherent rationale: one Gemini run chose
`[AveRooms, AveBedrms]` (catches the 132-room Tahoe blocks but ranks the
1,243-person prison block 4,259th of 20,640); another chose
`[AveRooms, AveOccup]` (the reverse); Gemini examined all three ratios and kept
them all. Termination is the agent's own decision — it stops when its top-5
looks convincing, and nothing currently asks *what its viewpoint misses*.
Both behaviours are measurable from the run files, and both are findings, not
bugs: derivation variance and stopping criteria are part of what the research
studies.

## Two rules to keep in mind when extending

1. **`max_steps` (code) must stay ≥ the prompt's tool budget + 2** — the
   finalising reply consumes a step, and an error-recovery retry costs another.
   The prompt number is a request; `max_steps` is the wall.
2. **The top-5 rows in `tools/run_lof_per_viewpoint.py`'s output are a deliberate MVP relaxation.**
   Seeing what it found is great for learning — but an agent that sees *which*
   rows scored high can tune its viewpoint toward them. The research version
   returns aggregate diagnostics only; restore that lock when this graduates
   from MVP to experiment.
