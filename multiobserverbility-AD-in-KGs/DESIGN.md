# Software design — the observability-point pipeline on CoDEx

A living document: **DECIDED** is settled, **PROPOSED** is my current suggestion
awaiting agreement, **OPEN** needs a decision before code.

Target architecture: <https://claude.ai/code/artifact/3a99a9c7-9adf-4372-92c1-12ca1079ea49>
(the technical redraw of the 26/02 whiteboard sketch — numbering ①–⑦ below
matches it). Worked-run transcript — the same pipeline as the LLMs actually
saw it, every prompt and output verbatim from run_20260826_082940:
<https://claude.ai/code/artifact/69a217f6-6016-4f11-a861-ced5563edb0c>. Predecessor project: `../multiviewpoint-AD-in-KGs/`, whose
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

---

## 12. Finding anomalies — norm-shaped candidate generators (brainstormed 26 Aug)

**The dilemma.** Option 1 (agent reads all 36,543 triples) dies on context.
Option 2 (one link predictor shortlists for every goal) dies on a fact:
DistMult is mathematically symmetric -- score(h,r,t) = score(t,r,h) -- so the
formalist's first signal, a mutual bond recorded one-way, is invisible to it
IN PRINCIPLE. One scorer cannot serve different norms, and forcing it would
reopen the predecessor's seam (norms with no causal path into detection).

**DECIDED-pending-agreement: Option 3.** A small menu of CANDIDATE
GENERATORS, each norm-shaped. Phase 2 extends: norms -> scope -> generators.
Generators FIND (deterministic, graph-scale); the agent JUDGES (reading-budget
scale, by its norms). The LLM never scores or ranks -- it removes and
explains (INV-3 preserved).

Menu v1, grounded in the graph (probed 26 Aug):

| generator | mechanism | serves | measured yield |
|---|---|---|---|
| `implausible_links` | KGE low score, precomputed | "false in the world" norms | 3,655 planted negatives to find |
| `reciprocity_gaps` | one-way edges on mostly-symmetric relations | mutuality norms | **191 real**: 180 diplomatic, 10 unmarried partner, 1 spouse (Russell Brand) |
| `multiplicity_outliers` | heads with >1 value on typically-unique relations | cardinality norms | **9 real**: double causes of death (Fuller, Cole...) |
| `type_clashes` | entity types vs relation's typical types | typing norms | 0 here -- the slice is type-clean; absence is still a findable answer |

Unservable norm, recorded honestly: anachronisms -- no date relations in
codex-s; judge-time world knowledge only.

**Bonus finding.** The 9 double-causes-of-death are a second LIVE
disagreement family: formalist flags multiplicity, realist judges medical
coherence and may pass. With Russell Brand that is two real
true-fact-disagreement families before M3 is built.

**M3 deltas.** `get_candidates` -> `find_candidates(generator, ...)`: one
tool, menu behind it (fixed tool list as the menu grows). Recall ceiling =
union of chosen generators. OPEN: gate generator choice behind a commitment
tool, or inline `why` only? Lean: no third gate -- norms are already frozen,
generator choice is instrumentation; record choice + why in the run file.

**§12 in plain words.** Each agent is a judge who can only read a small stack
of files; the generators are its assistants. An assistant is dumb but fast --
plain code that sweeps all 36,543 triples in seconds and knows exactly one
kind of suspicious. The judge picks the assistants that match its OWN norms
(the link predictor suits a "false facts" norm; the reciprocity query suits a
"mutual bonds recorded one-way" norm -- the link predictor cannot see that
kind at all), reads their shortlists a page at a time, and gives each
candidate a verdict by its norms: anomaly / ok / out of scope / unsure, with
one line of why. **Code finds, the agent judges** -- code covers the whole
graph but has no viewpoint; the agent has the viewpoint but cannot cover the
graph.

---

## 13. Milestone 3 — implementation plan (find and judge, ⑤–⑥)

Ends at verdicts recorded per agent. Evaluation/composition (⑦) is M4.

| # | piece | change | ~lines |
|---|---|---|---|
| 1 | `scripts/1_prepare_graph.py` | EXTEND: after merging, sample verified negatives (`--ratio`, `--seed`) into `prepared/kg.tsv` and write `prepared/ground_truth.tsv` (label, kind=verified_false). Guards: a negative must not already be in the graph, no duplicates, all-zeros line printed | +60 |
| 2 | `scripts/2_train_scorer.py` | NEW: PyKEEN (DistMult default) on the contaminated graph; win32 thread guards; saves model AND `prepared/scores.npy` scoring every triple, plus a manifest (graph hash, model) so a stale score file refuses loudly | 120 |
| 3 | `tools/generators/` | NEW package, one file per assistant, shared contract `find(scope_ids, ctx) -> [(triple, note)]`, all deterministic: `implausible_links` (reads scores.npy + manifest check) · `reciprocity_gaps` (one-way edges where symmetry ≥ 50%) · `multiplicity_outliers` (>1 value on typically-single-valued relations) · `type_clashes` (types outside the relation's dominant types) | 4 × ~60 |
| 4 | `tools/find_candidates.py` | NEW, the ONE agent-facing finder: `find_candidates(generator, why, page)` — requires the caller's scope; restricts every generator to it; pages of 10, labels + per-candidate note ("no reverse edge recorded"); assigns stable ids (c1, c2…); records what was served per agent; total capped at the reading budget N | 110 |
| 5 | `tools/submit_verdicts.py` | NEW: batch of `{id, verdict ∈ anomaly/ok/out_of_scope/unsure, why}`; rejects ids never served to the caller; accumulates `verdicts_N` across batches | 90 |
| 6 | `agents/sub_agents.py` + `config.py` | EXTEND: phase 3 in the instruction (pick assistants matching YOUR norms, read, judge); budgets: `READING_BUDGET_N = 30`, `PAGE_SIZE = 10`, tool budget → ~16 | ~40 Δ |
| 7 | `scripts/3_run_audit.py` | RENAME+EXTEND `2_run_setup.py`: same tree, now runs through verdicts; run file adds generators chosen (with why), candidates served, verdicts; blindness proof unchanged | +80 |
| 8 | `scripts/check_generators.py` | NEW rig, no API: every generator on the full graph and on a sample scope — counts, examples, determinism. The eyeball checkpoint before any agent touches them | 90 |

Unchanged: context store, the four data tools, `assign_perspective`,
`declare_semantics`, `select_scope`, phase gate (candidate/verdict ordering is
enforced inside the new tools: no scope -> no candidates; not served -> no
verdict).

Order: 1 → 8a (generators on the clean merge, pre-contamination sanity) → 2 →
3–5 → 8 (full rig) → 6–7 → offline suite → live runs.

Quota: setup ~10–13 + per agent ~7–9 (generators + 3 pages + 3 verdict
batches) → **~28–32 calls/run**, two quota windows; retry absorbs.

OPEN before build: negatives `--ratio` (lean: ~500 planted ≈ 1.4%, keeps
reading budgets meaningful) · N=30 v1 (agent may lower, never raise) ·
`implausible_links` model: DistMult is fine for falsehood-hunting even though
symmetric — its blindness to direction is exactly why reciprocity_gaps exists.

Exit questions:
1. Do agents pick generators that MATCH their norms (formalist → reciprocity/
   multiplicity, realist → implausible_links)?
2. Do the two agents give DIFFERENT verdicts on the same true fact
   (Russell Brand, double causes of death)?
3. Does the realist actually catch planted verified-false candidates the
   generator surfaces?

### Milestone 3 — built and measured (26 Aug 2026)

All pieces built; 36/36 gate checks; firewall clean. One handoff bug found
live and fixed: select_scope's confirmation still said "you are done" from
M2, and both auditors obediently stopped before phase 3 -- the tool response
is the last thing the model reads, and it steers.

**The scorer finding.** DistMult trained ON the contaminated graph cannot
separate CoDEx's hard negatives: AUC 0.617 (dim 64/100ep) vs 0.617 (dim
128/500ep) -- not undertraining. Training memorizes the planted triples as
positives; type-consistent negatives break no regularity. Per-generator
reachability of the 500 planted: multiplicity_outliers 70 (29% precision),
reciprocity_gaps 31 (14%), type_clashes 6, implausible_links top-100: **0**.
Structural generators carry detection; the KGE alone would have been blind.

**First full audit run (075904): everything worked.**
- Blindness verified both agents; 14 calls, 11.8s.
- Generator<->norm match EXACT: formalist -> reciprocity_gaps ("topological
  symmetry"), realist -> implausible_links ("empirical plausibility").
- Formalist: 10/10 anomaly -- ALL true facts (one-way diplomatic edges),
  flagged per its norms "regardless of real-world plausibility". The
  true-but-anomalous class, en masse.
- Realist: 7 anomaly / 3 ok; **5 of 7 anomalies are planted falsehoods**
  (Mendelssohn/Tocqueville/Obruchev US citizenship, Sennett born Moscow),
  each with correct world-knowledge reasoning. One planted missed (Lieberman
  -- an honest reviewer error, the thing reviewer-validation measures). Two
  flagged "true" facts may be REAL Wikidata errors (Illich US citizenship)
  -- kind=real means un-planted, not verified-true.
- **The scope rescued the weak scorer**: implausible_links top-100 globally
  holds 0 planted, but the realist's biographical SCOPE filtered the noise
  -- its served page held 6 planted in 10. norms -> scope -> generator
  composition turned AUC 0.617 into 71% verdict precision.

Not yet observed: the same candidate judged differently by both agents (the
scopes and pages did not intersect this run). That is M4's union/disagreement
report, plus more runs.

---

## 14. Milestone 4 — built and measured (26 Aug 2026)

Built: the evaluator (sole reader of ground_truth), the reciprocity
interleave fix (a page is now a cross-section, not the front of the largest
queue), and the second-opinion phase (each auditor judges the other's flags
BLIND -- neutral note, no verdicts, no hint another auditor exists; reviewer
agents resolve to their principal's stores). 42/42 offline checks.

**Run 082940 (30 calls, 99.8s, 3 retries absorbed, blindness verified):**

| | formalist | empiricist |
|---|---|---|
| judged | 15 (10 own + 5 second-opinion) | 20 (10 own + 10 second-opinion) |
| anomaly precision vs planted | 7/10 | 7/12 |
| planted seen but missed | 0 | 0 |

**Composed:** union 15 flags, agreement 7 -- and the agreement set IS the
planted set (all 7 planted that were served, caught by both). The
disagreement set has 8 members, all unplanted, in exactly the two designed
shapes:

1. TRUE facts the formalist flags and the empiricist passes: one-way
   diplomatic edges ("lacks the reciprocal link" vs "Cape Verde maintains
   normal diplomatic relations with the EU"). The Russell Brand structure,
   live, on real triples.
2. Facts the empiricist flags and the formalist DECLINES to judge:
   occupation errors ("Goethe was a writer, not a composer") met with a
   principled out_of_scope ("my norms evaluate schema compliance, not
   factual accuracy"). The abstention verdict working as designed -- and
   these unplanted flags are candidate REAL Wikidata errors.

The worked example's three-row structure (§10) is realised: agreement =
the verified-false; disagreement = the norm-dependent; and no answer key is
asked to settle what norms legitimately dispute.

M4 exit questions: (1) same-fact dual verdicts -- YES, 8. (2) numbers --
70%/58% precision, zero misses on served planted; union adds breadth (15 vs
10/12 flags) though both caught the same 7 planted, BECAUSE second opinions
propagate every flag to both judges. (3) disagreements read as viewpoints --
yes: coherent norm-grounded reasons on both sides, n=1 run.

**Naming (DECIDED 26 Aug).** The sub agents are OBSERVERS, not auditors.
"Auditor" implied an external standard of correctness; the architecture's
claim is the opposite -- anomaly is a property of the standpoint, and an
observer sees according to who it is. It is also the thesis's own word: the
project is multi-observerbility, the predecessor multi-observer, and an
observability point is where an observer stands. Code names: observer_1/2,
observer_1_reviewer for the second-opinion phase. Old run files store agent
names positionally, so they still evaluate; their stored "agent" fields keep
the historical name.

**Naming, completed (26 Aug).** The whole vocabulary is now the observer
family: the tree is `observation`, one run is an observation, scripts are
`3_run_observers.py` / `4_evaluate_verdicts.py`, run files are
`run_*_observers.json`, the ADK app is `multi_observer`, and the trigger
message is "Begin the observation." `agents/reviewer_agents.py` holds the
returning observers (second opinions). "Audit"/"auditor" survive only in
historical sections of this document and in old run filenames, which the
evaluator still accepts via --run.

**Tool naming, tightened (26 Aug).** `sample` -> `show_examples` (actionable,
matches the describe_ family), `lookup` -> `explain_term` (says what it looks
up), `find_candidates` -> `find_suspects`, and the generators package ->
`tools/scanners/` -- they scan the whole graph for one kind of suspicious
each and hand the observer suspects to judge. "Scanner" kept over "scorer
assistant" deliberately: three of the four produce no scores at all, only the
KGE-backed one does, and calling them scorers would smuggle back the idea
that the machinery ranks while the observer rubber-stamps. The observer is
the only judge; scanners only point.

**Plain-word scanner names (26 Aug).** The scanner menu and the inspection
tool are model-facing vocabulary -- the observer TYPES these as arguments --
so academic words became plain ones: reciprocity_gaps -> `one_way_links`,
multiplicity_outliers -> `too_many_values`, type_clashes -> `odd_types`
("odd", not "wrong": scanners point, they do not judge), implausible_links ->
`unlikely_facts`, and show_examples -> `inspect_triples`, completing the
inspection family (describe_dataset, describe_relation, explain_term,
inspect_triples). Old run files keep historical scanner names in their
records; the evaluator only displays them.

---

## 15. Dataset-leak audit and purge (27 Aug 2026)

Principle enforced: dataset-specific content lives ONLY in the dataset's
loader module (`loaders/codexs.py`: CARD, paths, NEGATIVE_SPLITS) and in what
the inspection tools return from the loaded data at runtime. A 5-lens audit
(model-facing strings, agent instructions, comments, behavioral hardcoding,
plus a synthetic-dataset switch test) found and fixed:

- **Model-facing (7)**: `describe_dataset` hardcoded a CARD-like sentence
  (now interpolates `DATASET.CARD`); `declare_semantics` hinted the domain
  ("marriages, careers, places" -- deleted; "people or places" -> "entities");
  `describe_relation`/`explain_term` said "Wikidata description" (-> "the
  description shipped with the dataset"); `find_suspects` used
  "Alice --spouse-- Bob" (-> "<head> --<relation>-- <tail>");
  `unlikely_facts`' stale error hardcoded paths (-> loader values).
- **Behavioral (6)**: negatives paths moved into the loader as
  `NEGATIVE_SPLITS`; label uniqueness is now CHECKED at load (a colliding
  dataset fails loudly) instead of assumed from one dataset's audit; all
  three rigs derive their probes from the loaded dataset -- check_gate picks
  its scope relation by the scanner's own criterion, so the rigs port with
  the switch.
- **Human-facing (12)**: scanner and loader docstrings de-specified
  (mechanisms stay, anecdotes point to this document, which keeps them).

Verified after: transmitted tool descriptions contain no dataset words
(checked via FunctionTool declarations), all rigs pass, prepare runs with
guards at zero. Note for future audits: the workflow's limit-killed verify
agents were mislabeled "refuted" -- treat a null verdict as UNVERIFIED.

### §15 closure — the two remaining audit items (27 Aug 2026)

**Empirical switch test: PASSED.** A scratch copy of the machinery was
pointed at a synthetic "machines and parts" dataset (9 entities, 3 relations,
its own CARD, its own negatives file) by editing only loaders/active.py plus
a new 25-line loader module. Results: 1_prepare ran with guards at zero;
describe_dataset opened with the MACHINES card; all 42 gate checks passed
with probes the rig derived itself (it found "linked with" as its gappy
relation); the scanners ran deterministically (one_way_links caught the
planted one-way edge); the ADK tree loaded; and a word-boundary sweep of
every instruction and every transmitted tool description found ZERO
people-domain words reaching any agent. The one caution for future sweeps:
"persona" contains "person" -- use word boundaries.

**Re-baseline run (027_ era, purged text): complete**, blindness verified,
20 calls, 6 retries absorbed. Notable: this run caught 0 planted falsehoods
-- the observers' scanner and scope choices simply never reached them
(observer_1 flagged 12 structural one-ways, all unplanted; observer_2 passed
18 of 20). Against the earlier run's 5/7 precision this is the variance the
seed harness exists to measure: per-run objective performance swings on
which observability points the root deals out. The disagreement set (10)
remains rich either way.
