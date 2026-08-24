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

1,775 lines of Python across 27 files.

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
one scorer or one tool directly, without an agent, and nothing depends on them.

---

## What the runs show so far

Eight runs are now in `runs/`, all on the same graph with the same prompts.

**Divergence is not reliable, in either direction.** Some runs give two
different scorers, some give the same one twice. Nothing in the design forces
either outcome, and `SPEC.md` says that should be measured rather than forced —
which makes it a result, not a defect, and the first thing a seed harness would
turn into a rate instead of an anecdote.

**An agent often fails to answer at all.** Counted across every run on disk:

| artifact | how it is captured | landed |
|---|---|---|
| the frame | a **tool call** (`declare_semantics`) | **12 / 12** |
| the spec | scraped from the agent's **final message** | **7 / 12** |

Two runs produced no usable spec from either agent — they declared a frame, ran
a scorer, then stopped without answering, and the run recorded nothing to
evaluate. Nothing retries. Every artifact written by a tool call has landed;
the one scraped from free text is the only one that goes missing.

**The agents pick deep budgets.** 10-20% of the graph here, against the 1-5%
ADKGD reports at, and earlier runs went to 30%. High recall, poor precision: at
20% one agent hit 94.8% recall at 42.7% precision, where the same scorer at 5%
gives 84.4% precision. Nothing in the prompt frames the budget as a review-cost
decision.

**Their scorer choices contradict the measured evidence.** They pair
containment with `plausibility` and adjacency with `neighbourhood`; §2.4 of
`SPEC.md` measured the opposite to be better on both counts.

---

## Order of remaining work

| # | build | why here | lines |
|---|---|---|---|
| **0** | **reviewer validation** | With no human downstream, the agent's judgement is the only thing steering the loop. Hand it flagged triples with no labels, ask "is this fact true?", compare to the answer key. **90%+ and the loop has a judge; 60% and the rest is built on noise.** Still not done, still the gate. | ~70 |
| 1 | seed harness | Loop contamination → train → agents → evaluate over N seeds and report a distribution. Turns "the agents diverged once" into a rate. | ~90 |
| 2 | budget guidance in the prompt | Tell the agent a budget is review cost. Cheapest fix for the biggest gap between agent and ADKGD numbers. | ~10 |
| 3 | more scorers (TRIC family) | Two scorers is a menu a `for` loop can exhaust; "the agent chose well" stays indistinguishable from luck until it is bigger. | ~80 each |
| 4 | domain KB | Only matters on a graph whose entity names are opaque. Not Countries. | ~60 |

Steps 1 and 2 are cheap and would sharpen everything already built.

Nothing on this list has been built yet. What has landed beside it:

- **The private semantics store and its gate.** `declare_semantics` writes a
  frame to `sem_a` / `sem_b`, and a `before_tool_callback` refuses `run_scorer`
  to any agent that has not written one — so a frame is a commitment made before
  the evidence, not a description of it. This was item 4 on the old list.
- **A label-free metric.** Because a frame names its relations, the evaluator
  can report what share of an agent's flags actually used them, against the base
  rate. One run has already scored **&minus;21.8 points** — an agent flagging the
  relation it declared it was *not* auditing.
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

***n* = 1 dataset.** Everything measured holds on Countries and nothing says it
holds elsewhere.
