# Plan

What exists, measured against the architecture in `SPEC.md`, and what is left.

Diagrams: <https://claude.ai/code/artifact/1a73cfb1-00fa-4f10-8787-3b587263a33f>

---

## Against the architecture diagram

| box | state |
|---|---|
| `contaminate` (seed 1 … 20) | **done** — `--seed` works; no multi-seed harness yet |
| `contaminated graph` | **done** |
| `ground truth`, sealed | **done** |
| profiler tools — `list_relations`, `describe_relation`, `sample` | **in progress** |
| Root Agent — writes both goals | pending |
| Viewpoint Agent 1 / 2 | pending |
| semantics 1 / 2, private and fenced | pending |
| domain knowledge (RAG) | pending |
| scorer — `plausibility` | **done** |
| scorer — `neighbourhood` | **done** |
| `run_scorer` — the tool an agent actually calls | pending |
| flags A / flags B | pending |
| overlap + union | pending |
| `evaluator` | **done** — `utils/evaluate.py` |
| label firewall | **done, enforced** — the grep passes |

## Against the derivation diagram

Nothing built. `profile -> goals -> perspective -> viewpoint` has no code behind
any arrow. The two scorers exist as the *menu* a viewpoint would pick from,
but nothing picks yet.

## What the current detect scripts are

`scripts/3_detect_neighbourhood.py` and `scripts/3_detect_plausibility.py` are
**not in the diagram**. They call a scorer directly and skip the agent layer.

That is deliberate — a scorer gets validated before an agent is allowed to
choose it — but it means the numbers so far (97 and 91 caught of 115) are
*scorer* results, not *system* results. Nothing has chosen anything yet.

---

## Order of work

| # | build | why here | lines |
|---|---|---|---|
| **0** | **reviewer validation** | With no human downstream, the agent's judgement is the only thing steering the loop. Give it the 127 flagged triples with no labels, ask "is this fact true?", compare to the answer key. **90%+ and the loop has a judge; 60% and steps 3-6 are built on noise.** Needs nothing else to exist. | ~70 |
| 1 | profiler tools + a script to read their output | What the root reasons from. No LLM, pure counting. | ~200 |
| 2 | `run_scorer` + `tools/registry.py` | The fixed four-tool surface an agent sees | ~80 |
| 3 | domain KB | So an agent can judge `chad locatedin europe` at all | ~60 |
| 4 | one viewpoint agent | Prove one works before two | ~120 |
| 5 | root + second agent | The experiment | ~90 |
| 6 | flags / overlap / union + seed harness | The measurement | ~80 |

Step 0 is the gate. It needs only the 127 triples
`3_detect_plausibility.py` already produces, and if it comes back weak, steps
3-6 change shape entirely.

---

## Open risks

**Two scorers is a small menu.** A dozen configurations is something a `for`
loop can exhaust, so "the agent chose well" stays hard to separate from luck
until more scorers exist. Add the TRIC-derived scorers before running the
selection experiment.

**The injected anomalies match the KGE negative sampler.** Uniform random tail
swaps are what the model was trained to reject, so `plausibility` has an
advantage here it would not have against real KG errors.

**Countries is too small to justify the architecture.** An LLM could read all
1,273 triples and find the errors directly. Scorers earn their place only when
the graph is too large to read.

***n* = 1 dataset.** Everything measured holds on Countries and nothing says it
holds elsewhere.
