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
