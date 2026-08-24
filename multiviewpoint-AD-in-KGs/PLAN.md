# Plan

What exists, measured against the architecture in `SPEC.md`, and what is left.

Diagrams — three views of the same system:

| view | what it draws | link |
|---|---|---|
| architecture | components, and the fences between them | <https://claude.ai/code/artifact/1a73cfb1-00fa-4f10-8787-3b587263a33f> |
| pipeline | the scripts, in the order they run | <https://claude.ai/code/artifact/bc114933-2731-48db-b52b-4effc7b474c2> |
| orchestration | one `runner.run()`, at runtime | <https://claude.ai/code/artifact/edc90ce5-c1db-45e6-9843-0252be2d7b74> |

The architecture sketch is the only one that draws the RAG store and the private
semantics — neither is built. The other two draw the code as it stands.

1,519 lines of Python across 21 files.

---

## Against the architecture diagram

| box | state | where |
|---|---|---|
| `contaminate` (seed 1 … 20) | **done** | `scripts/1_contaminate.py` |
| `contaminated graph` | **done** | `data/countries/` |
| `ground truth`, sealed | **done** | read by the evaluator alone |
| profiler tools — `list_relations`, `describe_relation`, `sample` | **done** | `tools/`, `utils/profile.py` |
| Root Agent — writes both goals | **done** | `agents/agent.py` |
| Viewpoint Agent 1 / 2 | **done** | ADK `ParallelAgent`, branch-isolated |
| `run_scorer` — the tool an agent calls | **done** | `tools/run_scorer.py` |
| scorer — `plausibility` | **done** | `tools/scorers/` |
| scorer — `neighbourhood` | **done** | `tools/scorers/` |
| flags A / flags B | **done** | saved as `findings` in the run file |
| overlap + union | **done** | union section in `6_evaluate.py` |
| `evaluator` | **done** | `scripts/6_evaluate.py` |
| label firewall | **done, enforced** | the grep passes |
| **semantics 1 / 2**, private and fenced | **not built** | deferred |
| **domain knowledge (RAG)** | **not built** | deferred: Countries' entity names are already meaningful |
| **seeds 1 … 20 harness** | **not built** | `--seed` works; nothing loops it |

## Against the derivation diagram

| step | state |
|---|---|
| profile → root | **done** — the root queries the profiler, then writes goals |
| root → two goals | **done** |
| goal → perspective | **partly** — the agent reasons, but the perspective is not recorded separately |
| perspective → viewpoint | **done** — the spec is `{scorer, budget, why}` |

## The pipeline as it runs today

```
1_contaminate.py   ->  contaminated_kg.tsv + ground_truth.tsv
2_train.py         ->  models/countries/distmult/
5_run_agents.py    ->  runs/run_<stamp>_adk.json   goals, specs, findings, trace
6_evaluate.py      ->  top-K% and the worst triples, per agent and combined
```

`3_detect_*.py` and `4_check_profiler.py` are test rigs, not pipeline steps —
they exercise one scorer or one tool directly, without an agent.

---

## What the runs show so far

The two runs now in `runs/`, same prompts, same graph:

| | run 104432 | run 105412 |
|---|---|---|
| agent 1 | plausibility @ 10% | plausibility @ 20% |
| agent 2 | neighbourhood @ 20% | neighbourhood @ 10% |

**Divergence is not reliable, in either direction.** These two both diverged,
and the budgets swapped between them. An earlier run — no longer in `runs/` —
gave the same scorer twice. So across the three observed: two split, one did
not, and nothing in the design forces either outcome. `SPEC.md` §2.5 says that
should be measured rather than forced, which makes this a result and not a
defect — and it is the first thing a seed harness would turn into a rate
instead of an anecdote.

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
| 4 | private semantics stores | Only matters once an agent should remember its own reasoning across steps. | ~60 |
| 5 | domain KB | Only matters on a graph whose entity names are opaque. Not Countries. | ~60 |

Steps 1 and 2 are cheap and would sharpen everything already built.

Nothing on this list has been built yet. Two small things landed beside it:

- `.env` now says `GOOGLE_API_KEY`, not `GEMINI_KEY`. ADK loads `.env` fine, but
  the `google-genai` client underneath only reads `GOOGLE_API_KEY` or
  `GEMINI_API_KEY` — the invented name was silently ignored, so `adk web` came
  up without a key. `adk web` works now; `5_run_agents.py` is unaffected, its
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
