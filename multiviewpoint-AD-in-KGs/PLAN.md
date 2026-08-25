# Plan

What exists, measured against the architecture in `SPEC.md`, and what is left.

Diagrams — three views of the same system:

| view | what it draws | link |
|---|---|---|
| architecture | components, and the fences between them | <https://claude.ai/code/artifact/1a73cfb1-00fa-4f10-8787-3b587263a33f> |
| pipeline | the scripts, in the order they run | <https://claude.ai/code/artifact/bc114933-2731-48db-b52b-4effc7b474c2> |
| orchestration | one `runner.run()`, at runtime | <https://claude.ai/code/artifact/edc90ce5-c1db-45e6-9843-0252be2d7b74> |

The architecture diagram was redrawn against the code and now shows only what
exists. The pipeline and orchestration diagrams predate `declare_semantics` and
the gate, so neither shows them yet.

2,227 lines of Python across 30 files.

---

## Against the architecture diagram

| box | state | where |
|---|---|---|
| `contaminate` (seed 1 … 20) | **done** | `scripts/1_inject_anomalies.py` |
| `contaminated graph` | **done** | `data/countries/` |
| `ground truth`, sealed | **done** | read by the evaluator alone |
| profiler tools — `list_relations`, `describe_relation`, `sample` | **done** | `tools/`, `utils/profile.py` |
| Root Agent — writes both goals | **done** | `agents/root_agent.py` |
| Viewpoint Agent 1 / 2 | **done** | `agents/viewpoint_agents.py`, ADK `ParallelAgent` |
| `run_scorer` — the tool an agent calls | **done** | `tools/run_scorer.py` |
| scorer — `plausibility` | **done** | `tools/scorers/` |
| scorer — `neighbourhood` | **done** | `tools/scorers/` |
| flags A / flags B | **done** | saved as `findings` in the run file |
| overlap + union | **done** | union section in `4_evaluate_results.py` |
| `evaluator` | **done** | `scripts/4_evaluate_results.py` |
| label firewall | **done, enforced** | the grep passes |
| **semantics 1 / 2**, private and fenced | **done** | `tools/declare_semantics.py` → `sem_a` / `sem_b` |
| **domain knowledge (RAG)** | **not built** | deferred: Countries' entity names are already meaningful |
| **seeds 1 … 20 harness** | **not built** | `--seed` works; nothing loops it |

## Against the derivation diagram

| step | state |
|---|---|
| profile → root | **done** — the root queries the profiler, then writes goals |
| root → two goals | **done** |
| goal → perspective | **done** — `declare_semantics` records it, and the gate makes it precede scoring |
| perspective → viewpoint | **done** — the spec is `{scorer, budget, why, summary}` |

## The pipeline as it runs today

```
1_inject_anomalies.py           ->  contaminated_kg.tsv + ground_truth.tsv
2_train_plausibility_scorer.py  ->  models/countries/distmult/
3_run_agentic_detector.py       ->  runs/run_<stamp>_adk.json
                                    goals, semantics, specs, findings, trace
4_evaluate_results.py           ->  top-K%, worst triples, semantic consistency
```

A script's name says what it is for: **numbered scripts are the pipeline and the
number is the order**; unnumbered `check_*` scripts are test rigs that exercise
one part directly and nothing depends on them. `check_agent_turns.py` is the
newest: it prints every model response with its finish_reason, which is the
only way to see a turn that ended without a tool call.

---

## What the runs show so far

### Everything previously recorded here was measured through a broken instrument

Read this before trusting any number below it.

The free Gemini tier allows **15 requests per minute per model**. This tree
needs **19** (root 4, each viewpoint 7–8). Runs therefore had their last
requests refused with a 429 — and until 25 Aug 2026 nothing recorded that. A
run file's trace records tool CALLS, so these two are indistinguishable in it:

    an agent that ran a scorer and then chose to stop
    an agent whose next request the API refused

Every earlier finding on this page read the second as the first. Instrumenting
it (`agents/telemetry.py`, and `scripts/check_agent_turns.py` to watch a run
live) showed **31 model responses of which ZERO returned nothing** — the agents
were never failing to answer. Two prompt rewrites were spent on a behaviour
that does not exist.

Worse, whether a run survived depended on **how long it happened to take**: a
slow run spans more than one quota window and completes; a fast one does not.
So run-to-run comparisons were partly comparisons of wall-clock luck.

**Consequently the following are UNMEASURED, not established:**

- "the agents do not reliably diverge"
- "an agent often fails to answer at all" (the 7/12 spec-capture rate)
- any comparison between prompt versions, including the claim that
  `submit_spec` made capture worse — it was quota-starved, and in the first
  clean run since, **both** agents called it successfully

None can be checked retroactively: every run predating telemetry lacks the
`health` block that would say whether it was truncated.

### What is measured

A run now records `health` and carries a third status, `truncated`, distinct
from `incomplete`. `4_evaluate_results.py` refuses to read a truncated run as
evidence. Numbers gathered from here on are trustworthy; numbers from before
are not.

**Seven runs on 25 Aug 2026, one truncated by quota and excluded. The six that
completed:**

| | result |
|---|---|
| frames captured (`declare_semantics`) | **12 / 12** |
| specs captured (`submit_spec`) | **12 / 12** |
| specs that came via the tool, not the text fallback | **12 / 12** |
| runs where the two agents chose **different** scorers | **6 / 6** |
| budgets chosen | only **0.10** and **0.15** |
| model calls per run | 18–19 |

This overturns three things this page previously asserted:

- *"An agent often fails to answer at all"* (7/12) — **false.** 12/12 when the
  run is not quota-starved. The failures were refused requests.
- *"Divergence is not reliable"* — **overturned, then partly restored.** Six of
  six split, one agent taking `plausibility` and the other `neighbourhood`. A
  seventh clean run afterwards gave `neighbourhood` twice, so it is 6/7, not
  6/6. Divergence is common here, not guaranteed — which is what §2.5 said
  should be measured rather than forced.
- *"The agents pick deep budgets, 10–30%"* — **much tighter now**: nothing above
  15%. Still above the 1–5% ADKGD reports at, but not wildly.

**What this does NOT establish.** Four things changed between the old runs and
these: telemetry (no behavioural effect), the four missing tool descriptions,
`submit_spec`, and the prompt wording around it. The improvement cannot be
attributed to any one of them, and the old runs cannot be re-classified because
they have no `health` block. The narrower budgets are *consistent* with
`run_scorer` and `submit_spec` now describing a budget as review cost, but that
is a hypothesis, not a measurement.

Six runs is also six. It is enough to retire a claim that something "often"
fails when it now never does; it is not enough to claim divergence is reliable.

**The budget is still deeper than the convention.** 10–15% of the graph against
the 1–5% ADKGD reports at. At 10% a run flags 127 triples to catch 91 of 115 —
79.1% recall at 71.7% precision, where the same scorer at 5% gives 84.4%
precision. So the agents are buying recall with review cost, deliberately or
not, and now at least the tools tell them that is the trade.

**Their scorer choices contradict §2.4 of `SPEC.md`.** They pair containment
with `plausibility` and adjacency with `neighbourhood`; §2.4 measured the
opposite to be better on both counts. This one has held across every run, old
and new, which makes it the most durable observation on this page — and the
sharpest open question, since the frame an agent declares is currently unable
to influence what its scorer flags at all (`sem["relations"]` is read only by
the evaluator's print).

**Four of five tools sent the model an empty description.** `list_relations`,
`describe_relation`, `sample` and `run_scorer` had *module* docstrings, which
ADK never reads, and no *function* docstring — so ADK transmitted 0 characters
for each. Only `declare_semantics` had one, which made its geography worked
example the sole tool guidance in the system. Fixed 25 Aug 2026; every run
before that date was made by agents inferring four tools from their names.

---

## Order of remaining work

| # | build | why here | effort |
|---|---|---|---|
| ~~0~~ | ~~the quota ceiling~~ | **Done 25 Aug 2026.** `agents/config.py` now passes `HttpRetryOptions` to the model. 429 was always in google-genai's retriable codes, but retry is OFF unless you pass options, and the defaults (1,2,4,8s) are too shallow for a window the API says needs ~53s. Waits are now 10/20/40/70s. Verified by burning 14 of 15 quota requests and then running: 19 calls, 5 retried, **run completed** in 126.6s where it would previously have been truncated. Costs nothing when under the limit. | done |
| **1** | **reviewer validation** | With no human downstream, the agent's judgement is the only thing steering the loop. Hand it flagged triples with no labels, ask "is this fact true?", compare to the answer key. **90%+ and the loop has a judge; 60% and the rest is built on noise.** | ~70 lines |
| 2 | portability blockers | Guard `corrupt()` on an empty pool (crashes on Nations, 20/20 seeds); refuse a collapsed scorer spread; report the tie block at the budget cut; truncate the "unknown relation" error; derive `KINDS` from the data. None changes a Countries result. | ~40 lines total |
| 3 | a second dataset | Nations or FB15k-237, to find out what else only works here. Blocked on #2. | ~2 h |
| 4 | neutralise the prompts | The geography in `ROOT_INSTRUCTION`, `VIEWPOINT_INSTRUCTION` and `declare_semantics`. Then ablate: does the root still write a sound goal when its worked example is not the answer? | ~1 h + runs |
| 5 | seed harness | Loop contamination → train → agents → evaluate over N seeds. Only worth building after #0, or it measures the quota. | ~90 lines |
| 6 | more scorers | Two is a menu a `for` loop can exhaust; "the agent chose well" stays indistinguishable from luck until it is bigger. | ~80 each |
| 7 | domain KB | Only matters on a graph whose entity names are opaque. Not Countries. | ~60 lines |

**#0 was fixed the same day it was found.** Trimming calls could never have
worked: two viewpoints at 6-7 each is 12-14 before the root does anything, so
the tree cannot fit under a 15/minute ceiling — it has to wait instead. A run
now records `retries` and `seconds`, because a retry that SUCCEEDS never
reaches the error callback, and without that a run at the ceiling would just
quietly get slower. **A run with retries > 0 is a run that would have been
truncated before**; one with retries = 0 got lucky on timing.

What has landed beside this list:

- **Tool descriptions.** Four of five tools were sending the model 0 characters.
  All five now transmit, and the four new ones are deliberately dataset-neutral.
  This also delivered the old item 2 for free — `run_scorer` and `submit_spec`
  now both describe the budget as a review cost.
- **`submit_spec`.** The spec is handed in by a tool call rather than scraped
  from the agent's last message. Measured over six clean runs: **12/12 specs,
  all of them via the tool**, none falling back to text-scraping.
- **Telemetry.** `agents/telemetry.py` records model calls and errors;
  `scripts/check_agent_turns.py` shows every response's finish_reason live.

- **The private semantics store and its gate.** `declare_semantics` writes a
  frame to `sem_a` / `sem_b`, and a `before_tool_callback` refuses `run_scorer`
  to any agent that has not written one — so a frame is a commitment made before
  the evidence, not a description of it. This was item 4 on the old list.
- **A label-free metric.** Because a frame names its relations, the evaluator
  can report what share of an agent's flags actually used them, against the base
  rate. It has a sign and it moves: one run scored **&minus;21.8 points** (an
  agent flagging the relation it said it was *not* auditing), a later one
  **+16.3**. No labels are involved in either.
- `.env` now says `GOOGLE_API_KEY`, not `GEMINI_KEY`. ADK loads `.env` fine, but
  the `google-genai` client underneath only reads `GOOGLE_API_KEY` or
  `GEMINI_API_KEY` — the invented name was silently ignored, so `adk web` came
  up without a key. `adk web` works now; `3_run_agentic_detector.py` is unaffected, its
  `find_key()` already accepted all three spellings.
- Each viewpoint's answer now carries a `summary` field, so what the agent found
  is readable in the `adk web` chat pane rather than only in the events trace.
  Nothing reads that field — it is for the human.

---

## Open risks

**Two scorers is a small menu.** Until it grows, agent choice cannot be
distinguished from luck.

**The injected anomalies match the KGE negative sampler.** Uniform random tail
swaps are what the model was trained to reject, so `plausibility` has an
advantage here it would not have against real KG errors.

**Countries is too small to justify the architecture.** An LLM could read all
1,273 triples directly. Scorers earn their place only when the graph cannot be
read end to end.

***n* = 1 dataset — and it is worse than it sounds.** This used to say
"everything measured holds on Countries and nothing says it holds elsewhere".
An audit on 25 Aug 2026 pointed the pipeline at PyKEEN's Nations and found the
problem is not that results might not transfer. **The pipeline does not run.**

| what breaks | where | how |
|---|---|---|
| **`1_inject_anomalies.py` crashes** | `corrupt()` | `other[r] = all_tails - own[r]` is empty for 6 of 55 Nations relations, so `rng.integers(0)` raises. **20 of 20 seeds die.** |
| **`type_invalid` stops meaning anything** | the taxonomy | it means "a tail from another relation's pool", which equals *wrong type* only because Countries' two pools are perfectly disjoint. On Nations every entity is a nation, so 100% of "type_invalid" corruptions are type-CORRECT. On FB15k-237 the other-pool is ~99% of all entities, i.e. uniform random. |
| **`neighbourhood` becomes a constant** | the scorer | all 1,992 Nations triples score exactly 1.0, std 0.0. `flag_worst` then breaks the tie by row order and returns rows 0–198, which the evaluator scores and reports as a detection. Nothing checks for a collapsed spread, though `2_train_plausibility_scorer.py` already does exactly that check for the model. |

So Countries is not one datapoint among many possible ones: several design
decisions are only coherent on a graph with exactly two disjoint-vocabulary
relations. `loaders/active.py` calls itself "the dataset switch: one import
line" — it is nine lines with no logic, and it does not make this work.

**The prompts are written for Countries too.** `VIEWPOINT_INSTRUCTION` asserts
*"You know what a country is, what a continent is"* as the world knowledge to
use, and `ROOT_INSTRUCTION`'s only `good` example is a border audit — which the
root has been observed paraphrasing back as its goal. Until that example is
neutral, **"the root writes goals from structure alone" is untestable.**
