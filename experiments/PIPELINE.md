# KGSAGE Pipeline — Locked Design

**KGSAGE** (Knowledge Graph Semantic Anomaly GEnerator) — an adversarially-trained negative sampling framework for generating type-coherent, semantically plausible KG anomalies. Implements the CGSP framework (Tong et al. 2026, DAMI) inside ADKGD's experiment harness.

Definitive reference for the locked design. Update only when scope changes are explicitly agreed.

## Scope

Locked deliverable: replace the legacy Gumbel-Softmax GAN inside `experiments/gan/` with the KGSAGE implementation that generates Category-5 semantic anomalies. Validate on FB15K-237 first; extend to WN18RR and YAGO 4.5 after FB validates.

| Aspect | Decision |
|---|---|
| Method | KGSAGE — REINFORCE + concept-aware sampling + cardinality weighting (CGSP framework) |
| First dataset | FB15K-237 (in-place rewrite of existing code) |
| Future datasets | WN18RR, YAGO 4.5 (added after FB validates) |
| Anomaly focus | Category 5 (type-coherent, semantically wrong) |
| Stage C consumer | ADKGD bridge (runtime negative sampling) |
| Bulk injector | **Parked** — defer to future work |
| Versioning | **Single version** — no parallel subdirectory for the prior approach |
| Codebase home | `experiments/gan/` reorganised into three phase folders |
| Out of scope | Cat 3 cardinality / Cat 4 logical anomalies; universal/zero-shot corrupter |

### Exemplar

Source positive: `(Bill Gates, /people/person/nationality, USA)`
Target anomaly: `(Bill Gates, /people/person/nationality, UK)` — type-coherent, factually wrong, requires multi-hop reasoning to detect.

## Architecture: three phases under `experiments/gan/`

Each phase = one CGSP module from Tong et al. 2026 = one subfolder. Dependencies flow one direction: **concept → adversarial → corruption**.

```
experiments/gan/
├── data.py                              ← SHARED utility (triple loading, vocab)
│
├── concept/                             ← PHASE 1 — Concept Module
│   ├── __init__.py
│   ├── adapters/
│   │   ├── __init__.py                  ←   get_adapter() factory
│   │   ├── base.py                      ←   BaseKBAdapter ABC
│   │   └── freebase.py                  ←   FreebaseAdapter (Phase 1 only)
│   ├── concept_pools.py                 ← builds headPool, tailPool
│   ├── cardinality.py                   ← classifies relations as 1-1/1-N/N-1/N-N
│   └── preprocess.py                    ← Phase 1 orchestrator
│
├── adversarial/                         ← PHASE 2 — Adversarial Module
│   ├── __init__.py
│   ├── generator.py                     ← candidate-scoring G (was gan_model.py)
│   ├── discriminator.py                 ← internal TransE D
│   ├── candidate_pool.py                ← per-positive candidate builder
│   └── train.py                         ← REINFORCE training loop
│
├── corruption/                          ← PHASE 3 — Corruption Module
│   ├── __init__.py
│   ├── api.py                           ← KGCorrupter primitive
│   ├── infer.py                         ← inference internals (was corrupt_triples.py)
│   └── adkgd_bridge.py                  ← ADKGD consumer (moved from gan/ root)
│
└── outputs/                             ← cache + artefacts (gitignored)
    ├── concept_pools/
    │   └── FB15K-237.pkl
    ├── checkpoints/
    │   └── FB15K-237_kgsage.pt
    └── logs/
        └── FB15K-237_kgsage_training.json
```

### Single-rule per folder

- `concept/` — anything about CONCEPTS / TYPES / SCHEMA
- `adversarial/` — anything about LEARNING / TRAINING / GAN INTERNALS
- `corruption/` — anything about USING the trained G to produce anomalies
- `data.py` at gan/ root — shared utilities used by multiple phases

## Phase 1 — Concept Module

**Role:** extract schema and type information from a KG; produce standardised pools and cardinality for downstream phases.

**Input:** `data/FB15K-237/{train,valid,test}.txt`
**Output (per dataset):**
- `data/<DATASET>/entity_types.tsv` — user-facing, NTriples-style TSV
- `data/<DATASET>/entity_types_metadata.json` — provenance + stats
- `experiments/gan/outputs/concept_pools/<DATASET>.pkl` — internal cache

### Files in `concept/`

| File | Purpose |
|---|---|
| `adapters/base.py` | `BaseKBAdapter` abstract base class |
| `adapters/freebase.py` | Parses Freebase relation paths; covers any FB15K-family dataset |
| `concept_pools.py` | Builds `headPool[r]`, `tailPool[r]` from triples + types |
| `cardinality.py` | Classifies each relation as 1-1, 1-N, N-1, or N-N |
| `preprocess.py` | CLI orchestrator: runs adapter + pools + cardinality + writes outputs |

### Output formats

**`entity_types.tsv`** — three tab-separated columns, UTF-8, LF, no header. Multi-typed entities → multiple rows. Matches YAGO's `yagoTypes.tsv` convention.

```
/m/0d3k14    rdf:type    people
/m/0d3k14    rdf:type    organization
/m/09c7w0    rdf:type    location
```

**`entity_types_metadata.json`** — provenance and statistics:

```json
{
  "schema_version": 1,
  "dataset_name": "FB15K-237",
  "dataset_family": "Freebase",
  "source_kg": "FB15K-237 (Toutanova & Chen, 2015)",
  "source_kg_files": {
    "train.txt": {"lines": 272115, "sha256": "..."},
    "valid.txt": {"lines": 17535, "sha256": "..."},
    "test.txt":  {"lines": 20466, "sha256": "..."}
  },
  "adapter": {
    "name": "FreebaseAdapter",
    "version": "1.0",
    "extraction_method": "Parse Freebase relation prefixes; majority vote",
    "extraction_params": { "...": "..." }
  },
  "type_vocabulary": {
    "size": 76,
    "namespace": "Freebase top-level domains",
    "examples": ["people", "location", "organization", "..."]
  },
  "statistics": {
    "n_entities": 14541,
    "n_typed_entities": 14538,
    "coverage_fraction": 0.9998,
    "avg_types_per_entity": 1.74,
    "multi_typed_count": 8203
  },
  "produced_by": "kg_corrupter v1.0",
  "produced_at": "2026-07-01T14:30:00Z"
}
```

**`concept_pools.pkl`** — internal cache for Phase 2/3:

```python
{
  "schema_version":     1,
  "dataset_name":       str,
  "dataset_hash":       str,                       # sha256 of sorted training triples
  "n_entities":         int,
  "n_relations":        int,
  "headPool":           Dict[int, Set[int]],       # relation_id → entity_ids
  "tailPool":           Dict[int, Set[int]],
  "pool_sizes":         Dict[int, Dict[str, int]],
  "cardinality":        Dict[int, str],            # "1-1" | "1-N" | "N-1" | "N-N"
  "cardinality_stats":  Dict[int, Dict[str, float]],
  "entity_to_id":       Dict[str, int],
  "relation_to_id":     Dict[str, int],
  "id_to_entity":       Dict[int, str],
  "id_to_relation":     Dict[int, str],
  "extraction_method":  "schema_based",
  "source_files":       { "...": "..." },
  "real_triple_set":    Set[Tuple[int, int, int]], # for Phase 3 validation
  "produced_at":        str,
  "produced_by":        str,
}
```

### Phase 1 individual validation gate

Run on the login node, FB15K-237 only:

```
python -m experiments.gan.concept.preprocess --dataset FB15K-237
```

Pass criteria:
- `entity_types.tsv` produced with ≥99% entity coverage
- Spot-check 20 random entities — types match intuition
- `concept_pools.pkl` loads without error
- Every relation has non-empty `headPool[r]` and `tailPool[r]`
- Cardinality classifications look right (e.g., `/people/person/nationality` → `N-1`)
- `dataset_hash` matches a recomputed hash of the train.txt sorted

## Phase 2 — Adversarial Module

**Role:** train the GAN (Generator + internal Discriminator) to produce hard, type-coherent negatives via REINFORCE.

**Input:** `data/FB15K-237/train.txt` + `outputs/concept_pools/FB15K-237.pkl`
**Output:**
- `outputs/checkpoints/FB15K-237_kgsage.pt` — trained G + D weights
- `outputs/logs/FB15K-237_kgsage_training.json` — loss curves, baseline, reward

### Files in `adversarial/`

| File | Purpose |
|---|---|
| `generator.py` | MLP candidate scorer (replaces vocab-wide Gumbel G) |
| `discriminator.py` | Internal TransE D (shares embeddings with G) |
| `candidate_pool.py` | Build N_S=64 candidate sets per positive at training time |
| `train.py` | REINFORCE loop for G + margin loss for D + checkpoint saving |

### Hyperparameters (starting point)

```
batch_size:        128
embedding_dim:     100
N_S:               64       # candidate pool size per positive
g_lr:              1e-4
d_lr:              1e-3
margin:            0.5
baseline_decay:    0.99
warmup_epochs:     5        # D-only warmup before REINFORCE engages
total_epochs:      100
gradient_clip:     1.0      # critical for REINFORCE stability
```

### Phase 2 individual validation gate

Two sub-gates:

**Sub-gate 2a — components standalone** (login node):
- `discriminator.py`: forward pass produces scalar scores
- TransE pre-trained 1 epoch on random negs: positive scores > random-neg scores
- `candidate_pool.py`: returns N_S candidates per positive, all type-coherent
- Cardinality weights apply correctly per relation type
- `generator.py`: forward pass produces tensor of shape `[batch, N_S]`; softmax sums to 1

**Sub-gate 2b — REINFORCE convergence** (HPC GPU):
- `g_loss` and `d_loss` both trend downward
- `baseline_ema` is stable (non-NaN throughout)
- For 20 random positives, G's top-1 candidates look plausible (manual spot-check)
- D scores positives higher than G's sampled negatives consistently

If sub-gate 2b fails (REINFORCE doesn't converge), fall back to "untrained G + concept filter" — gives a working Idea-1 style pipeline as Plan B.

## Phase 3 — Corruption Module

**Role:** use the trained GAN to corrupt triples at inference time. Expose a clean primitive for consumers (ADKGD now, future detectors later).

**Input:** `outputs/checkpoints/FB15K-237_kgsage.pt` + `outputs/concept_pools/FB15K-237.pkl`
**Output:** negative triples (in-memory, per call)

### Files in `corruption/`

| File | Purpose |
|---|---|
| `api.py` | `KGCorrupter` class — the primitive consumers instantiate |
| `infer.py` | Stage C internals (candidate pool + concept filter + sampling + validation) |
| `adkgd_bridge.py` | ADKGD wrapper that calls `KGCorrupter.corrupt()` per training batch |

### The primitive (api.py)

```python
class KGCorrupter:
    def __init__(self, checkpoint_path, concept_pools_path, real_triples, max_retries=10):
        ...

    def corrupt(self, positive, slot=None, seed=None) -> Triple:
        # 1. select slot (random or forced)
        # 2. build candidate pool from concept pools
        # 3. score with frozen G
        # 4. apply cardinality weights
        # 5. sample from P_G
        # 6. validate (not self-loop, not in real_set, retry up to max_retries)
        # 7. fallback to uniform random if all retries fail; log it
        # 8. return
        ...

    def corrupt_batch(self, positives, seed=None) -> List[Triple]:
        ...
```

### Determinism contract

- Same `(triple, seed)` → same output
- Output never equals input positive
- Output never self-loops
- Output not in `real_triple_set`
- Fallback never silently used (always logged)

### Phase 3 individual validation gate

Three sub-gates:

**Sub-gate 3a — primitive standalone** (login node):
- `corrupt()` returns type-coherent negatives on 100 random positives
- No self-loops, no collisions with `real_triple_set`
- `<1ms` per call on GPU (target for ADKGD compatibility)
- Determinism: same `(triple, seed)` → same output

**Sub-gate 3b — bridge integration** (login node):
- `adkgd_bridge.generate(positive)` returns valid triple
- 100 calls without exception
- Logged outputs look like Category 5 anomalies on visual inspection

**Sub-gate 3c — end-to-end with ADKGD** (HPC GPU):
- `python experiments/run_experiment.py --dataset FB15K-237 --neg_source gan --gan_path .../FB15K-237_kgsage.pt`
- ADKGD completes 1 training epoch + test without errors
- Final B2 (KGSAGE-trained) Precision@K, Recall@K beat B0 (random-trained) on at least 3 of 5 K cutoffs

## Incremental in-place plan — 7 phases

Each phase is small, individually validatable, and ends with an explicit pass/fail gate.

### Phase 1.1 — Concept module foundation (1 day)

```
ADD:
   experiments/gan/concept/__init__.py
   experiments/gan/concept/adapters/__init__.py
   experiments/gan/concept/adapters/base.py
   experiments/gan/concept/adapters/freebase.py

VALIDATE:
   python -c "from experiments.gan.concept.adapters import get_adapter; \
              a = get_adapter('freebase'); print(a.family_name)"
   prints "Freebase" ✓

EXISTING FILES TOUCHED: NONE.
   B0 + B1 baselines BOTH STILL WORK.
```

### Phase 1.2 — Concept pools + cardinality + preprocess (1 day)

```
ADD:
   experiments/gan/concept/concept_pools.py
   experiments/gan/concept/cardinality.py
   experiments/gan/concept/preprocess.py

CREATE (Phase 1 output):
   data/FB15K-237/entity_types.tsv
   data/FB15K-237/entity_types_metadata.json
   experiments/gan/outputs/concept_pools/FB15K-237.pkl

VALIDATE (Phase 1 gate above):
   - entity coverage ≥99%
   - all 237 relations have non-empty pools
   - cardinality classifications plausible
   - dataset_hash matches recomputed hash

EXISTING FILES TOUCHED: NONE.
   B0 + B1 baselines BOTH STILL WORK.
```

### Phase 2.1 — Adversarial components (2 days)

```
ADD:
   experiments/gan/adversarial/__init__.py
   experiments/gan/adversarial/discriminator.py
   experiments/gan/adversarial/candidate_pool.py

VALIDATE (Phase 2a sub-gate):
   - TransE D forward pass + 1-epoch warmup → pos > random_neg
   - candidate_pool builds N_S=64 type-coherent candidates per positive
   - cardinality weights applied correctly

EXISTING FILES TOUCHED: NONE.
   B0 + B1 baselines BOTH STILL WORK.
```

### Phase 2.2 — Generator rewrite (1 day, BREAKING CHANGE)

```
ADD:
   experiments/gan/adversarial/generator.py     ← rewrite of gan/gan_model.py

VALIDATE:
   - forward pass: [batch, N_S] tensor produced
   - softmax over candidates sums to 1
   - random init produces uniform-ish scores

EXISTING FILES NOT YET TOUCHED, BUT:
   gan/gan_model.py is now LOGICALLY SUPERSEDED.
   gan/train.py and gan/corrupt_triples.py still import from it for now.
```

### Phase 2.3 — REINFORCE training (3-5 days, HIGH RISK)

```
ADD:
   experiments/gan/adversarial/train.py         ← rewrite of gan/train.py

DELETE (after successful test run):
   experiments/gan/train.py                     ← legacy Gumbel training

NEW SLURM:
   experiments/slurm/train_kgsage_fb15k.slurm     (or rename existing)

RUN:
   sbatch experiments/slurm/train_kgsage_fb15k.slurm

PRODUCE:
   experiments/gan/outputs/checkpoints/FB15K-237_kgsage.pt
   experiments/gan/outputs/logs/FB15K-237_kgsage_training.json

VALIDATE (Phase 2b sub-gate):
   - both losses trend downward
   - baseline_ema stable
   - top-1 candidates plausible on 20-spot-check
   - D scores positives > G's negatives

IF REINFORCE FAILS:
   Fall back to "untrained G + concept filter" (Plan B).
   Document the decision; still produces a meaningful contribution.
```

### Phase 3.1 — Corruption inference + primitive (2 days)

```
ADD:
   experiments/gan/corruption/__init__.py
   experiments/gan/corruption/infer.py          ← rewrite of gan/corrupt_triples.py
   experiments/gan/corruption/api.py            ← KGCorrupter class

DELETE (after Phase 3.2 passes):
   experiments/gan/corrupt_triples.py           ← legacy inference

VALIDATE (Phase 3a sub-gate):
   - corrupt() returns type-coherent negatives
   - no self-loops, no real_set collisions
   - determinism: same (triple, seed) → same output
```

### Phase 3.2 — Bridge integration (1 day)

```
MOVE + EDIT:
   experiments/gan/adkgd_bridge.py → experiments/gan/corruption/adkgd_bridge.py

UPDATE IMPORTS:
   experiments/run_experiment.py: update --gan_path default path
   (no API change — same triple-in, triple-out contract)

VALIDATE (Phase 3b sub-gate):
   - adkgd_bridge.generate() returns valid triple on 100 calls
   - logged outputs look like Category 5 anomalies
```

### Phase 3.3 — End-to-end on FB15K-237 (2-3 days HPC)

```
RUN:
   sbatch experiments/slurm/run_baseline_fb15k.slurm           # B0 (random)
   sbatch experiments/slurm/run_baseline_with_gan_fb15k.slurm  # B2 (KGSAGE)

VALIDATE (Phase 3c sub-gate, the empirical claim):
   - B2 beats B0 on at least 3 of 5 K cutoffs
   - Spot-check 50 G-generated negatives — confirm Category 5
   - Build comparison table for thesis
```

### Phase 4 (FUTURE, after FB validates) — extend to WN18RR + YAGO 4.5

```
ADD:
   experiments/gan/concept/adapters/wordnet.py
   experiments/gan/concept/adapters/yago_schemaorg.py

ACQUIRE: YAGO 4.5 custom subset (Path B, separate sub-plan)

RUN: same 7-phase pipeline per dataset (no code rewrite,
     just new adapter + retraining)
```

## Development conventions (code style)

Match the style of existing `experiments/gan/*.py`:

1. **Module docstring at top** — explain purpose, key steps, the WHY. Use numbered step lists for multi-step pipelines.
2. **Short readable functions** — one job per function. Use descriptive names like `load_checkpoint`, `build_candidate_pool`, not `lc`, `bcp`.
3. **Comments explain WHY, not WHAT** — code should explain itself; comments add context the reader can't infer.
4. **Imports at top, grouped** — stdlib → third-party → local. Use `# noqa: E402` if `sys.path` manipulation forces a delayed import.
5. **Type hints in function signatures where useful** — not mandatory everywhere, but improves readability for non-trivial APIs.
6. **Print statements use `flush=True` for live logging** — matches existing `_run()` pattern in `run_experiment.py`.
7. **Defensive defaults via `os.environ.setdefault`** — preserves cluster overrides (matches `OMP_NUM_THREADS` pattern).
8. **Errors are explicit** — return non-zero exit codes, write to `sys.stderr`, don't silently swallow.

Example skeleton matching the existing style:

```python
"""One-line summary of what this file does.

Longer explanation. Mention the inputs/outputs, the role in the pipeline,
and any non-obvious design decisions.

The N-step process:

  STEP 1: What it does.
          (Why this is needed.)
  STEP 2: Next thing.
          (Edge case / rationale.)
  ...
"""
import os
import sys

import numpy as np
import torch

from experiments.gan.concept.concept_pools import load_pools


def main_entry_point(arg1, arg2):
    """One-line summary. Args and returns described if non-obvious."""
    # WHY: brief explanation if the WHY is non-obvious.
    ...
```

## Per-phase individual validation summary

Every phase ends with a validation gate. Don't move to the next phase until the current one passes.

| Phase | Gate type | Where to run | Pass criteria |
|---|---|---|---|
| 1.1 | Import smoke test | Login | `get_adapter('freebase')` returns instance |
| 1.2 | Output inspection | Login | Coverage ≥99%, pools populated, hash matches |
| 2.1 | Component smoke | Login | D scores ordered, candidates type-coherent |
| 2.2 | Generator unit | Login | Forward pass returns expected shape |
| 2.3 | Training convergence | HPC GPU | Losses trend down, top-1 candidates plausible |
| 3.1 | Primitive contract | Login | corrupt() respects all guarantees |
| 3.2 | Bridge functional | Login | 100 calls without exception |
| 3.3 | End-to-end empirical | HPC GPU | B2 beats B0 on ≥3 K cutoffs |

## Migration map: existing → new

| Current file | New location | Action |
|---|---|---|
| `experiments/gan/gan_model.py` | `experiments/gan/adversarial/generator.py` | REWRITE (Phase 2.2) |
| `experiments/gan/train.py` | `experiments/gan/adversarial/train.py` | REWRITE (Phase 2.3) |
| `experiments/gan/corrupt_triples.py` | `experiments/gan/corruption/infer.py` | REWRITE (Phase 3.1) |
| `experiments/gan/adkgd_bridge.py` | `experiments/gan/corruption/adkgd_bridge.py` | MOVE + EDIT (Phase 3.2) |
| `experiments/gan/data.py` | `experiments/gan/data.py` | KEEP IN PLACE (shared) |
| `experiments/gan/outputs/` | `experiments/gan/outputs/` | KEEP, expand subfolders |
| `experiments/gan/playground_*.py` | (delete or archive) | OPTIONAL CLEAN-UP |
| `experiments/run_experiment.py` | (no move) | MINOR EDIT — update `--gan_path` default |

## Worked example trace (Bill Gates)

| Phase | Action | Result |
|---|---|---|
| Phase 1 | `FreebaseAdapter.extract_types()` parses `/people/person/nationality` | `head_type = people`, `tail_type = location` |
| Phase 1 | `concept_pools.build_pools()` accumulates | `headPool[nationality] = {Bill Gates, Steve Jobs, ...}`, `tailPool[nationality] = {USA, UK, Canada, ...}`, `cardinality[nationality] = "N-1"` |
| Phase 2 | REINFORCE training over ~100 epochs | G learns to prefer semantically-similar candidates (e.g., Linus Torvalds for Bill Gates) over random ones (Tuvalu, Microsoft) |
| Phase 3 | `KGCorrupter.corrupt((Bill Gates, nationality, USA))` | Returns `(Linus Torvalds, nationality, USA)` — Category 5: type-coherent, factually wrong |
| Phase 3 | ADKGD bridge calls `corrupt()` per training batch | ADKGD's BiLSTM-Attention learns to discriminate semantically-similar pairs |

## Three locked claims for the thesis

1. **Architecture claim** — the library is generic at the algorithm and code level; only the trained weights and the schema adapter are dataset-specific. Adding a KB family is a ~50-line adapter; new datasets within a covered family require zero code.
2. **Method claim** — concept-aware adversarial generation (KGSAGE, following the CGSP framework) produces type-coherent semantically-plausible anomalies (Category 5), targeting the residual error class that escapes rule-based validation methods.
3. **Empirical claim** — detectors trained with our generator's negatives outperform random-negative-trained baselines on semantically-plausible test anomalies. Example: `(Bill Gates, nationality, UK)`.

## What's parked / out of scope

| Item | Status | Reason |
|---|---|---|
| Bulk injector (Consumer 2) | Parked | Defer to later; primitive contribution doesn't require it |
| `data_anomalies/` deliverable | Parked | Depends on bulk injector |
| Generic usage-based fallback adapter | Out of scope | Locked: three KB families only |
| Cat 3 cardinality / Cat 4 logical anomalies | Out of scope | Require symbolic methods, not GAN |
| Universal/zero-shot corrupter | Out of scope | ULTRA-style research project; future work |
| LLM-based corruption | Out of scope | Different paradigm; future work |
| Other KG families (NELL, Wikidata, ConceptNet, biomed) | Out of scope | Documented as extensions in adapter README |

## Timeline (FB15K-237 only — first pass)

| Week | Phases | Deliverable |
|---|---|---|
| 1 | 1.1 + 1.2 + 2.1 | Concept pools built for FB15K-237; adversarial components standalone tested |
| 2 | 2.2 + 2.3 (start) | New Generator; REINFORCE training launched on HPC |
| 3 | 2.3 (finish) + 3.1 + 3.2 | Trained KGSAGE checkpoint; primitive working; bridge connected |
| 4 | 3.3 | B0 vs B2 comparison; decision point to extend to WN18RR + YAGO 4.5 |

After FB15K-237 validates, WN18RR and YAGO 4.5 each take ~1.5-2 weeks (just adapter + retraining; no algorithmic rewrite).

## Output format summary

```
PER DATASET, PHASE 1 PRODUCES (concept module):
  data/<DATASET>/entity_types.tsv              user-facing, NTriples-style TSV
  data/<DATASET>/entity_types_metadata.json    user-facing, provenance + stats
  experiments/gan/outputs/concept_pools/<DATASET>.pkl  internal cache, rebuildable

PER DATASET, PHASE 2 PRODUCES (adversarial module):
  experiments/gan/outputs/checkpoints/<DATASET>_kgsage.pt
  experiments/gan/outputs/logs/<DATASET>_kgsage_training.json

PHASE 3 OUTPUT (corruption module):
  In-memory negatives via KGCorrupter.corrupt(); no on-disk dataset files.
  (Bulk injector parked.)
```
