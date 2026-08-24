# Multi-viewpoint anomaly detection in knowledge graphs

Finding wrong facts in a knowledge graph, from more than one point of view.

A fact is scored by several independent **scorers**, each using a different
kind of evidence. Which scorers to trust, and what to look for, is meant to be
decided by agents rather than fixed in advance. The research question is
whether two viewpoints disagreeing tells you something one viewpoint alone
cannot.

`SPEC.md` has the design. This file is how to run what exists.

---

## Setup

No `pip install -e`, no packaging. Scripts add the repo root to `sys.path`
themselves, so they work from any working directory.

```
python -m pip install pykeen pandas numpy google-adk
```

`google-adk` is only needed from step 3 on — steps 1 and 2 and every `check_*`
script run without it. It brings `google-genai` with it, which is what actually
talks to Gemini.

The agents also need a key. Put it in a `.env` beside this README:

```
GOOGLE_API_KEY=...
```

The name matters: `google-genai` reads `GOOGLE_API_KEY` or `GEMINI_API_KEY` and
nothing else, so a `.env` using any other spelling is loaded and then silently
ignored, and `adk web` comes up with no key.

On this machine:

```
C:/Users/thari/miniconda3/python.exe      has pykeen 1.11.1, torch 2.8, pandas
```

Windows note: the scripts set `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
`KMP_DUPLICATE_LIB_OK` before importing torch. Without them this machine
aborts inside oneDNN with no Python traceback. The guard is `win32`-only, so a
Linux run is not throttled to one thread.

---

## Run order

A script's name says what it is for. **Numbered scripts are the pipeline, and
the number is the order.** Unnumbered `check_*` scripts are test rigs: they
exercise one tool or one scorer directly, with no agent involved, and nothing
in the pipeline depends on them.

```
python scripts/1_inject_anomalies.py           build the fixture + the answer key
python scripts/2_train_plausibility_scorer.py  the model that scorer reads
python scripts/3_run_agentic_detector.py       run the agents, record what they found
python scripts/4_evaluate_results.py           score that run against the answer key

python scripts/check_profiler.py               read every profiler tool's output
python scripts/check_scorer_neighbourhood.py   one scorer, no model needed
python scripts/check_scorer_plausibility.py    the other, needs a trained model
```

### 1. Inject anomalies

Reads the clean graph and injects known errors, so there is something to detect
and an answer key to score against.

```
data/countries/{train,valid,test}.txt   ->   contaminated_kg.tsv   what detectors read
                                             ground_truth.tsv      what the evaluator reads
```

Two classes of injected fake, named for how they are **built**, not for how
hard they are:

| kind | how | example |
|---|---|---|
| `type_invalid` | tail from the OTHER relation's pool | `belize neighbor western_asia` |
| `type_valid` | tail from the SAME pool, wrong value | `honduras locatedin western_europe` |

```
python scripts/1_inject_anomalies.py
python scripts/1_inject_anomalies.py --ratio 0.15 --invalid-frac 0.3 --seed 7
```

Three guards print every run and **must all read 0**:

```
GUARDS  fake-but-actually-true 0   self-loops 0   duplicates 0
```

The first is the one that quietly ruins results — labelling a true fact as an
error means punishing a detector for being right.

### 2. Train the plausibility scorer's model

Trains a KGE model on the **contaminated** graph — errors included, as
positives.

```
python scripts/2_train_plausibility_scorer.py
python scripts/2_train_plausibility_scorer.py --model ComplEx --epochs 2000
```

Writes `models/countries/distmult/`. One folder per architecture, so several
can coexist and be compared.

**Why train on the errors.** Train on clean data and inject afterwards and you
measure memorisation: every real triple was seen, every fake was not, and a
detector separates *seen from unseen* rather than *true from false*. Training on
the dirty graph is also what real auditing looks like — you have one dirty
graph, not a clean one.

The script prints a score spread and four sample queries. Read them. A
collapsed spread (std near 0) means nothing downstream can work. In the top-3
answers, an obviously wrong *kind* of entity means the model has not learned
what the slot takes.

*Currently observed:* `neighbor` queries return the head itself near the top —
the model has learned a near-identity component for a symmetric relation.
Harmless when scoring existing triples, but it is why link-prediction metrics
from this model should not be quoted.

### Checking one scorer by hand

One script per scorer, so each can be read and run on its own. Both flag the
worst 10% and score that against the answer key.

```
python scripts/check_scorer_neighbourhood.py            no model needed, runs in a second
python scripts/check_scorer_plausibility.py             needs models/countries/distmult/
python scripts/check_scorer_plausibility.py --budget 0.05
```

| scorer | evidence | caught of 115 | precision |
|---|---|---|---|
| `neighbourhood` | the graph, counted directly | 97 | 76.4% |
| `plausibility` | the embedding model | 91 | 71.7% |

Both far above the 9.0% a random 127 flags would give.

`plausibility` refuses to run against a graph its model was not trained on:

```
STALE MODEL: models/countries/distmult was trained on a different graph
(115 triples the model never saw, 115 it saw that are not in the data).
```

Nothing else binds `models/` to `data/`. Re-contaminate without re-training and
the old model scores a graph whose errors it never saw — which reads as an
excellent result (measured once by accident: 90.6% precision, 100% recall)
rather than an obvious failure. So it raises.

---

## Layout

```
data/<name>/        dataset files only — sources and generated, no code
loaders/            where files are and how to read them
  active.py           the dataset switch: one import line
  countries.py        paths and names
  graph.py            read a TSV of triples
scripts/            numbered = the pipeline, in order; check_* = test rigs
tools/              what an agent is allowed to call
  scorers/            neighbourhood.py, plausibility.py — NOT registered as tools
agents/             the ADK tree, one file per job
  __init__.py         puts the repo root on sys.path — nothing else
  config.py           model, budget, state keys, tool sets
  parsing.py          getting JSON back out of the model's text
  root_agent.py       the goal-writing agent and its prompt
  viewpoint_agents.py the auditor factory, its prompt, and the gate
  agent.py            assembles the tree — what ADK imports
utils/              detect.py (flagging), evaluate.py (the only label reader)
models/             generated, gitignored
```

`agents/agent.py` is deliberately thin. ADK looks up `agents.agent` and reads
`root_agent` from it, so it is the front door — and a front door should show
the shape of the house, not its furniture. The two long prompts live beside the
agents that use them.

Imports run one way only:

```
scripts/  ->  agents/, loaders/, tools/scorers/, utils/
agents/   ->  tools/  ->  tools/scorers/
```

An agent sees a fixed set of tools — `list_relations`, `describe_relation`,
`sample`, `run_scorer` — however many scorers exist. **Scorers are the menu
`run_scorer` picks from, never tools in their own right.** Registering them
individually would change both agents' tool lists every time one was added, and
the two agents must differ only in the goal they are given.

No script names a dataset. They read `from loaders.active import DATASET` and
use `DATASET.KG`, `DATASET.TRUTH`, `DATASET.MODELS`. Switching dataset is one
import line in `loaders/active.py`.

---

## The one rule

**`ground_truth.tsv` is read by the evaluator, and by nothing else.**

Not by a scorer, not by a tool, not by an agent. A detector that has seen the
answer key has not detected anything. Once there is code in `tools/` and
`agents/`, this is checkable rather than promised:

```
grep -rnI --exclude-dir=.adk "ground_truth" tools/ agents/   # must return nothing
```

`-I` and `--exclude-dir=.adk` are not cosmetic. `adk web` writes a binary
session database under `agents/.adk/`, and grep reports a match inside it —
which turns the one check this project relies on into a false positive that a
reader learns to ignore. The directory is gitignored for the same reason.

---

## Status

| | |
|---|---|
| `1_inject_anomalies.py`, `2_train_plausibility_scorer.py` | done |
| `neighbourhood` + `plausibility` scorers | done |
| `check_*` test rigs | done |
| agent-facing tools, `run_scorer` | done |
| root and viewpoint agents, on ADK | done |
| `declare_semantics` + the gate on `run_scorer` | done |
| `3_run_agentic_detector.py`, `4_evaluate_results.py` | done |
| a harness that loops seeds | not built — `--seed` is a flag nothing iterates |
| reviewer validation | not built — still the gate on everything else |

The pipeline is deterministic apart from the agents: same `--seed`, same output
bytes, verified across separate processes. The agents are not — two runs of
`3_run_agentic_detector.py` on one graph can pick different scorers, which is a result to
measure rather than a bug to fix, and `PLAN.md` tracks what has been seen.
