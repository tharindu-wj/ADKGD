# Software design — the observability-point pipeline on CoDEx

A living document: **DECIDED** is settled, **PROPOSED** is my current suggestion
awaiting agreement, **OPEN** needs a decision before code.

Target architecture: <https://claude.ai/code/artifact/3a99a9c7-9adf-4372-92c1-12ca1079ea49>
(the technical redraw of the 26/02 whiteboard sketch — numbering ①–⑦ below
matches it). Predecessor project: `../multiviewpoint-AD-in-KGs/`, whose
mechanics we port, not its tools.

---

## 1. The data, as measured (26 Aug 2026)

`data/` is the CoDEx repository layout, CoDEx-S slice:

| file | contents | measured |
|---|---|---|
| `triples/codex-s/{train,valid,test}.txt` | the graph, tab-separated ids | 36,543 triples, 2,034 entities, 42 relations |
| `triples/codex-s/{valid,test}_negatives.txt` | **hand-verified FALSE triples** | 3,655 |
| `entities/en/entities.json` | id → label, description, wiki url | 77,951 total; **2,034/2,034** of ours labelled, 2,019 described |
| `relations/en/relations.json` | id → label, description | 71 total; **42/42** of ours labelled + described |
| `types/entity2types.json` | entity id → type ids | **2,034/2,034** of ours typed |
| `types/en/types.json` | type id → label, description | 3,443 types |

A fully resolved triple: `Q7604 P1412 Q188` → *Leonhard Euler —languages
spoken, written, or signed— German* (head type: human).

**Label collisions inside the codex-s vocabulary: zero** — entity labels and
relation labels are both unique. Verified 26 Aug.

---

## 2. Layer ①/③ — dataset context (the definitions store)

**DECIDED — labels only, ids in artifacts.** No agent ever sees a bare
`Q…`/`P…` id. Every tool resolves ids to labels on output and accepts labels on
input (unique, so unambiguous — re-verify this invariant if the slice ever
changes). Run files and state store *both* id and label: ids for exactness,
labels for the human reading the run.

**DECIDED — one loader, loaded once.** `loaders/context.py` reads the four
JSON files a single time per process, immediately subsets 77,951 entities to
the 2,034 in the graph, and builds the id↔label indexes. ~11 MB parse, once.

**DECIDED — tools read prepared artifacts only, never `data/` raw.** The
negatives files ARE the answer key now. The firewall therefore moves up a
level: step-1 output (`contaminated_kg.tsv`, `ground_truth.tsv`) is the only
triple source any tool may touch, and the firewall grep gains a pattern:
nothing in `tools/` or `agents/` may name `_negatives`. Definitions
(`entities/relations/types` JSONs) are safe for tools — they contain no triples.

---

## 3. The tool surface (PROPOSED — under discussion)

Two families, same split as the gate rule that already works: **context tools
return facts and are never gated; pipeline tools move the audit forward and
are ordered.**

### Context tools — ungated, read-only, both agents + root

| tool | returns | replaces |
|---|---|---|
| `describe_dataset()` | totals + all 42 relations: label, triple count, distinct heads/tails | `list_relations` |
| `describe_relation(name)` | Wikidata description + stats (cardinality, symmetry, top tails **as labels**) + 3 example triples | old profiler + definitions, merged |
| `lookup(term)` | entity: label, description, types (as labels), degree. relation: label, description, usage count | **new** — the definitions store made queryable |
| `sample(relation?, n)` | up to 10 resolved triples | same, now labelled |

### Pipeline tools — ordered, caller-keyed, writes are tool calls

| # | tool | who | contract |
|---|---|---|---|
| ② | `assign_observability_point(agent, scope, goal, norms)` | **root only** | validates scope relations exist and agent name is real; writes `op_a`/`op_b` state. Root must place BOTH before the parallel phase starts. |
| ④ | `select_scope(relations)` | sub agent | must be a subset of its OP's scope; registers the subgraph, returns its size. Prerequisite for candidates — this is the gate, relocated. |
| ⑤ | `get_candidates(page)` | sub agent | serves the top-N of the agent's scope subgraph, ~10 per page, resolved to labels, with scores. Refuses until `select_scope`. |
| ⑥ | `submit_verdicts(verdicts)` | sub agent | batch of `{triple, verdict ∈ anomaly/ok/out_of_scope, why}`; rejects triples it never served; accumulates findings. |

Rationale carried from the predecessor, measured there:
- **every artifact the run needs is written by a tool call** (tool-call writes
  landed 12/12; final-message scraping lost 5/12 in the same conditions);
- tools must have **function** docstrings — module docstrings transmit 0 chars
  to the model (four tools silently shipped empty for weeks);
- validation errors return as text so the agent self-corrects in-loop.

### PROPOSED — scores are precomputed, not computed in the loop

`2_train` scores the **entire contaminated graph** once and writes
`scores.npy` beside the model. `get_candidates` = load array → mask to scope →
top-N → page. No model, no torch, no 40-second load inside an agent turn, and
scoring is byte-identical across runs. The stale-model guard extends to the
score file (hash of the graph it scored).

This also enforces the diagram's redline mechanically: scores computed against
the FULL graph — the scope only *selects* — because the file is written before
any scope exists.

---

## 4. Contamination (①-adjacent, feeds everything)

**PROPOSED.** `1_prepare` builds
`contaminated_kg = train ∪ valid ∪ test ∪ sample(negatives, ratio)` and writes
`ground_truth.tsv` with `kind = verified_false`. No `corrupt()` — CoDEx's
negatives are type-consistent, human-checked false facts (*Mariah Carey
—spouse— Sean Penn*), which retires the "anomalies match the KGE negative
sampler" risk outright.

**OPEN** — additionally inject a slice of synthetic corruptions as a second
`kind`, for continuity with the Countries numbers? My lean: not in v1; one
honest anomaly family first.

---

## 5. Ported unchanged from `multiviewpoint-AD-in-KGs`

- `agents/telemetry.py` + `health` block + `truncated` status (the quota
  lesson: a refused request must never read as agent behaviour)
- `HttpRetryOptions` on the model (10/20/40/70s waits)
- `.env` key naming (`GOOGLE_API_KEY`), win32 thread guards
- `loaders/active.py` dataset-switch pattern
- scripts naming: numbered = pipeline, `check_*` = test rigs
- ParallelAgent branch isolation + state-key discipline + the
  `include_contents="none"` rule

---

## 6. Open questions (the current brainstorm)

1. **Who sets N (the reading budget)?** The diagram says N = reading capacity,
   absolute. Options: fixed config (simplest, comparable across runs) / root
   sets it per OP (one more thing to validate) / agent chooses under a cap.
   My lean: **config default, agent may lower, never raise.**
2. **Is the OP's scope binding?** `select_scope ⊆ OP.scope` (my lean — the OP
   is an assignment, and drift would blur whose findings are whose) vs
   advisory.
3. **Verdict vocabulary.** `anomaly / ok / out_of_scope` — is `unsure` a
   fourth verdict or is that what `ok` + a low-confidence `why` means? My
   lean: add `unsure`; forcing a binary call manufactures false confidence.
4. **Quota arithmetic.** ~4 context calls + 1 scope + N/10 pages + N/10
   verdict batches per agent. N=60 → ~35 model calls/run vs 15/min free tier.
   Retry absorbs it, but a run becomes ~3 minutes. Acceptable, or argue for
   N=30 in v1?
5. **Where does the root's context come from?** `describe_dataset` +
   `describe_relation` + `lookup` may be enough (labels are meaningful). The
   type vocabulary (3,443 types) is loaded but unexposed — add a
   `common_types()` tool later if root goals come out too vague.

---

## 7. Not yet designed (deliberately)

Evaluation (⑦'s scoring side): verified-false triples give objective
precision/recall per agent and for the union; norm disagreements on true facts
have no key and are reported, not scored. The evaluator redesign gets its own
section once the tool surface is agreed.

---

## 8. Milestone 1 (DECIDED 26 Aug) — root + context tools only

Scope: everything upstream of the sub agents. Produce OPs; consume nothing.

| piece | what | ~lines | needs API? |
|---|---|---|---|
| `scripts/1_prepare_graph.py` | merge train+valid+test → `kg.tsv` (no negatives yet) | 30 | no |
| `loaders/context.py` | 4 JSONs → graph-vocab subset, id↔label maps | 100 | no |
| `tools/` context ×4 | `describe_dataset` · `describe_relation` · `lookup` · `sample` | 180 | no |
| `tools/assign_observability_point.py` | root's only write; validates scope + agent name | 60 | no |
| `agents/` | root only + ported telemetry/retry/config | 150 | run only |
| `scripts/check_context.py`, `scripts/2_run_root.py` | eyeball rig · the run | 120 | 2nd only |

Excluded on purpose: training, scorer, subgraph/candidates/verdicts, sub
agents, contamination, evaluator.

Exit question this milestone answers: **asked for two OPs with differing norms
over a shared scope, does the root produce coherent ones?** (Self-written
frames converged 44/44; root assignment is the fix under test.)
Run cost ~5–7 model calls.

### Milestone 1 — built and measured (26 Aug 2026)

All 17 files built; every offline check passes (parse, ADK discovery, all 5
tool descriptions transmit, firewall grep clean, OP-tool validation incl. the
punctuation-proof norms guard). Two live runs, both `completed`, 4–6 model
calls, 8–10s — nowhere near quota.

**Exit question — does the root produce two coherent, differing norms?**

| | run 002523 | run 002553 |
|---|---|---|
| OPs placed | 2/2, via tool calls | 2/2 |
| scopes | disjoint partition (biographic vs geographic) | **overlapping** (3 shared relations) |
| norms differ | yes — symmetry-stance vs geo-consistency | yes — logical-consistency vs reciprocity/topology |
| "anomalous even if true" stance | partially (unreciprocated spouse = anomaly) | no |

Verdict: **the machinery works and the norms guard holds, but the full
same-scope/different-norms pair has not yet appeared** — the root leans toward
partitioning by relation, and neither run produced a clean norm of the
"flag it even if correctly recorded" kind. n=2. Options if this persists:
strengthen the instruction (ask for shared scope explicitly rather than
calling it "most valuable"), or have the tool enforce scope overlap. Decide
after more runs, not from two.

Also observed: run 1's norm for sub_agent_1 mentions "places of birth", which
is not in that agent's scope — the tool validates scope names, not norm prose.
Acceptable; the sub agent's out_of_scope verdict handles stray prose later.

---

## 9. Blind norms — the two-phase observability point (brainstormed 26 Aug)

**The principle.** Like people: a person's sense of what a normal relationship
looks like is formed by their background BEFORE they meet the community they
judge. The data teaches them the local vocabulary and where their values
apply; it does not supply the values. So:

    PHASE 1  who I am      norms from world knowledge alone
                           sees: DATASET_CARD (domain, 2-3 sentences)
                           never: relations, counts, samples
            -- the gate, inverted --
    PHASE 2  where I look  inspect the dataset, map norms onto its
                           vocabulary -> scope (relevant relations)

**Why structural, not prompted — our own evidence.** Both Milestone-1 runs
produced norms soaked in dataset vocabulary ("diplomatic relations",
"spouses", even "degree distributions" — data statistics, explicitly ruled
out). A root that browses first reverse-engineers "what would be anomalous
given this schema". Same failure family as frames-after-scores; same fix:
ordering enforced in code.

**Generalised gate rule (replaces the v1 rule).** A tool is blocked while it
could contaminate a commitment not yet made. Phase 1: context tools blocked.
Phase 2: context open, scoring (later) still gated behind scope.

**PROPOSED — personas solve the convergence problem.** Blind self-derived
norms would reconverge (44/44 precedent). The root — seeing ONLY the card,
needing no tools at all — assigns each sub agent a differing PERSONA (a
stance, e.g. legal formalist vs descriptive empiricist); each sub agent then
articulates its OWN norms from its persona. The root is the circumstance that
makes people different, not the author of their views.

**"Clearly show the separation" = three proofs.**
1. Code: per-agent phase gate on the context tools.
2. Trace: run script mechanically verifies first-context-call > norms-commit,
   per run, and marks violations invalid (like the firewall grep).
3. Artifact: blind norms contain no dataset vocabulary by construction, so
   they are PORTABLE across datasets — re-run phase 2 on another graph with
   the same norms. Norms that transfer are proof they never came from one.

**Build delta.** DATASET_CARD in the loader (domain + subject kinds, never
attribute kinds); root loses all tools, `assign_perspective` x2 with the
differing guard; sub agents get `declare_semantics` (blind, first) then
context tools then `select_scope`; OP splits into persona/norms/scope, each
stamped with call order.

**OPEN.**
1. Card grain — proposed line: name the domain and subjects, never attributes.
2. Personas: root-generated per run (agentic, variable) vs a fixed pair in
   config (reproducible, less agentic)? Lean: root-generated, seed-harness
   later measures variance.
3. Does phase 2 allow `sample()`? Seeing instances teaches vocabulary but also
   leaks "what is common" — norms are already fixed by then, so yes, allow.

**DECIDED (26 Aug) — card delivery.** The card is a `CARD` constant in the
dataset loader, injected into agent instructions like `{DATASET.NAME}` --
never the typed user prompt (unowned, leakable, unprovable) and never a tool
(costs calls, skippable). The run script pins the trigger message to a fixed
"Prepare the audit." so the card is the only domain channel, and records the
card verbatim in the run JSON as provenance. Residual hole, accepted: in
`adk web` a human can type schema into the chat; the scripted pipeline is the
measured path. Card text: "An encyclopedic knowledge graph about notable real
people, organisations and places." -- "encyclopedic" and "notable" kept
deliberately (they anchor the right world-knowledge prior); subjects only,
never attributes.

---

## 10. The pipeline on real data — worked example (all triples real)

**STEP 0 — the card** (the only thing phase 1 may see):
"An encyclopedic knowledge graph about notable real people, organisations and
places."

**PHASE 1a — root assigns personas** (sees the card, has no tools):

    sub_agent_1  FORMALIST   "a relationship is defined by its rules --
                              mutuality, exclusivity, consistency of record.
                              A rule violation is an anomaly even when every
                              fact in it is accurate."
    sub_agent_2  EMPIRICIST  "only a factually false claim can be wrong.
                              Unusual or incomplete arrangements that really
                              happened are not your concern."

**PHASE 1b — blind norms.** A peek is refused by the gate:

    -> describe_dataset()
    <- ERROR: you have not formed your view yet. Declare what YOU consider
       a normal relationship before looking at any data.

    formalist:  "a marriage is mutual by definition -- a record of an
                 inherently two-way bond held by one party only is anomalous
                 EVEN IF the underlying fact is real"
    empiricist: "a claim is anomalous only when false in the world;
                 incomplete but real relationships pass"

Note the vocabulary: "marriage", "two-way bond" -- world words. Neither agent
knows the dataset calls anything `spouse`.

**PHASE 2 — the gate lifts; norms map onto the actual vocabulary.**
`describe_dataset()` reveals the 42 relations; each agent selects the scope
its norms apply to:

    formalist:  spouse, unmarried partner, sibling, diplomatic relation
    empiricist: spouse, unmarried partner, sibling, child

Same slice, different reasons -- shared-scope/different-norms by construction.

**PHASES 5-6 — judged candidates** (later milestones; the triples are real):

| candidate | truth | formalist | empiricist |
|---|---|---|---|
| Mariah Carey --spouse-- Sean Penn | verified false | anomaly | anomaly |
| Russell Brand --spouse-- Katy Perry | TRUE, but the graph's one unreciprocated spouse edge | **anomaly** (mutuality violated) | **ok** (really married) |
| Katharine McPhee --spouse-- David Foster | true, both ways | ok | ok |

**7 — composed final list:**

    A AND B agree   Mariah Carey --spouse-- Sean Penn     <- scores both agents
    A only          Russell Brand --spouse-- Katy Perry   <- THE DISAGREEMENT SET
    B only          (empty here)

The disagreement row is the architecture's product: a true fact, anomalous
from one observability point, unremarkable from another.

**Portability, one line:** hand the formalist's norms to the Countries graph
and phase 2 maps them to `neighbor` (borders are mutual). Same norms, new
dataset, new scope -- the viewpoint never came from either dataset.

---

## 11. Milestone 2 — implementation plan (blind setup, ①–④)

Rebuild of the setup stage on the §9 design. Ends at scopes selected; scorer,
candidates and verdicts stay out of scope.

| # | piece | change | ~lines |
|---|---|---|---|
| 1 | `loaders/codexs.py` | add `CARD` (the §9-grain text, exactly as in §10) | +6 |
| 2 | `tools/assign_perspective.py` | NEW, replaces `assign_observability_point` (deleted): root's only tool; validates agent name, non-empty persona, differing-personas `_essence` guard; writes `persona_1/2` | 80 |
| 3 | `tools/declare_semantics.py` | NEW, sub agent, phase 1b: `(normal, anomalous, lets_pass)`; caller-keyed → `norms_1/2`; NOT validated against the dataset (it is blind); immutable once set; cross-agent identical-norms guard | 90 |
| 4 | `tools/select_scope.py` | NEW, sub agent, phase 2: `(relations, why)`; requires own norms first; validates labels via context; writes `scope_1/2` with ids+labels | 70 |
| 5 | `agents/phase_gate.py` | NEW: `before_tool_callback` — no norms yet → only `declare_semantics` allowed, context tools refused with the teaching error; norms set → context + `select_scope` open, re-declaration refused | 60 |
| 6 | `agents/root_agent.py` | REWRITE: no context tools — `[assign_perspective]` only; instruction = card + "two genuinely differing personas" | 60 |
| 7 | `agents/sub_agents.py` | NEW factory (twin discipline as before): instruction = card + persona via `{persona_N}` state templating + the two-phase contract; tools = `[declare_semantics, select_scope]` + context tools; gate + telemetry callbacks | 110 |
| 8 | `agents/config.py` | update keys (`PERSONA/NORMS/SCOPE_KEYS`), tool lists, budgets | ~30 Δ |
| 9 | `agents/agent.py` | tree becomes `SequentialAgent(root, ParallelAgent(sub_1, sub_2))` | ~15 Δ |
| 10 | `scripts/2_run_setup.py` | replaces `2_run_root.py`: runs the tree; prints personas/norms/scopes; records the card verbatim; **ordering proof** — per agent, verify from the trace that `declare_semantics` precedes the first context call, print BLINDNESS VERIFIED or mark the run invalid | 140 |
| 11 | `scripts/check_gate.py` | NEW rig, no API: gate blocks/opens correctly, immutability, persona guard, scope-requires-norms | 80 |

Unchanged: `1_prepare_graph.py`, `check_context.py`, the four context tools,
`context.py`, `telemetry.py`, `graph.py`, `active.py`.

Order: 1–2 + 6 (root testable alone, ~3 calls) → 3–5 → 11 (offline gate
proof) → 7–9 → 10 → offline suite (parse, ADK load, declaration transmission,
firewall, check_context, check_gate) → 2–3 live runs.

Quota: root ~3 + each sub agent ~5–7 → **~15–17 calls/run**, at the ceiling;
retry absorbs it.

Exit questions this milestone answers:
1. Do blind norms come out free of dataset vocabulary and data-statistics
   language? (Milestone 1's did not -- that is the regression test.)
2. Do persona-derived norms actually differ, or reconverge despite personas?
3. Does phase-2 mapping choose sensible, overlapping scopes?

### Milestone 2 — built and measured (26 Aug 2026)

All 11 pieces built. Offline: 25/25 gate checks pass; tree loads; all 7 tool
descriptions transmit; no cross-key leak; no schema words in the root's
instruction; firewall clean. Two live runs, both `completed`:

| | run 072336 | run 072441 |
|---|---|---|
| calls / time | 10 / 12.3s | 13 / 44.4s (4 retries -- ceiling absorbed) |
| **blindness proof** | **VERIFIED both agents** (norms at #2/#3, first data call at #4/#5) | **VERIFIED both agents** |
| personas | structural formalist vs empirical realist | structural formalist vs empirical historian |
| norms differ | yes -- and BOTH carry the "may flag what is factually true" stance | yes |
| scope overlap | 6 relations shared (spouse, sibling, diplomatic relation, citizenship, birth, death) | agent 1's scope (spouse, sibling, diplomatic relation) is a SUBSET of agent 2's 14 |

Exit questions:
1. **Blind norms free of dataset vocabulary?** Yes, and mechanically proven
   per run. Norms speak in world/ontology terms ("cardinality", "birthplaces")
   -- no codex-s labels, no data statistics. The Milestone-1 regression
   (norms soaked in schema) is gone.
2. **Do persona-derived norms differ?** Yes, sharply: the formalist flags
   structural violations "regardless of real-world plausibility" (= flags
   TRUE facts); the realist flags falsehoods "even if the graph schema is
   formally unbroken". The worked example's target pair, produced unprompted.
3. **Sensible overlapping scopes?** Yes -- substantial overlap both runs,
   including the disagreement-relevant symmetric relations. The
   Russell Brand one-way spouse edge falls in BOTH agents' scopes in both
   runs: the disagreement case is live.

Observed, for the seed harness later: the root's persona AXIS was
formalist-vs-external-truth in both runs. Within-run difference is what the
design needs and it is strong; across-run persona variance is unmeasured.
