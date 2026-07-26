# KGSAGE Finalization — Implementation Plan

> ⚠️ **HISTORICAL DOCUMENT (2026-07-03).** Kept as a record of what was planned
> and why. Its file paths and module names describe the codebase *as it was then*
> — e.g. `inference.py` is now `corruption_generation.py`, and the link-predictor
> modules (`lp_scorer.py`, `frozen_complex.py`, `band_sampler.py`) were removed
> when the architecture became LP-free. For current names see
> `KGSAGE_glossary.md`. Do not use this file's paths to navigate the code.

Companion to the Implementation Brief (2026-07-03). Groundwork: 33-agent verified code review + 5-agent
plan groundwork (TRIC feasibility, fixture design, design critique, line-anchored delta map, TAXO docs).
Line references verified against `dev_gan_1` (clean tree, 2026-07-03).

> **Numbering + vocabulary.** M0–M4 below are **project milestones**, not the pipeline's
> Phase 1/2/3 and not investigation rounds. This plan predates the dual-discriminator
> architecture: where it says "D" it means the single judge that later became **D_real**, the
> plausibility discriminator. See `KGSAGE_glossary.md`.

## 0. Flagged decisions (brief §8: "pause and flag") — resolve before/while M1

| # | Decision | Recommendation |
|---|----------|----------------|
| D1 | **Plausibility space for close-but-false.** The brief's "high cosine to the true value's / neighbourhood-consensus embedding" read in E' space re-imports the circularity the review flagged: E' trains only through the generator's losses, so the encoder can warp the "plausibility band" to make its own targets easy. | Use a **frozen pretrained KGE** (DistMult, train split only, ~150-line in-package trainer) as the plausibility scorer: `plaus(cand) := s(h,r,cand)` (tail) / `s(cand,r,t)` (head). Band-sample just below the *filtered* true-value score. Relation-aware (fixes 1-N relations; plain cosine-to-true-tail is relation-agnostic). Targets become stationary → precompute once. Report the KGE's filtered MRR as a sanity gate. E' remains the generator's conditioning; add a geometry probe + optional aux grounding loss. |
| D2 | **Test-axis composition.** Brief keeps test ∈ {random, gan}. Random test anomalies are ~50% fully-random garbage + ~17% relation swaps vs GAN 100% entity-slot — columns differ by composition, not just semantics. | Add **`random_entity_slot`** (uniform entity-only corruption) as a third test source and as a train source. Cheap (~30 lines) and it is the control that isolates "semantic" from "slot policy". |
| D3 | **TRIC arm.** `tric.py` is scratch code (runs at import, `input()` prompt, in-place corruption, 2 real bugs, row-adjacency semantics). On FB15K-237/WN18RR's arbitrary row order it degenerates to ~frequency-weighted random corruption. The TAXO findings doc itself says TRIC was "dropped from active comparison". | Keep it (published external baseline; 0.5–1 day reimplementation) but characterize honestly in the paper: "corrected reimplementation; near-random hardness on unordered benchmark files". Never present as the semantic competitor — `type_constrained` and `sampler_direct` are those. |
| D4 | **Fallback narrative if `sampler_direct` ties `gan`.** A plausible outcome per the review (the current GAN distils its sampler). | Pre-register: if the GAN adds nothing over the sampler, the paper's method IS the KGE-band + pair-aware sampler with the GAN as ablation. Decide claim wording now, not after the matrix. |
| D5 | **Human audit.** Real-error arm is out of scope, so all external validity rests on the M4 hardness protocol. | Include a 100-triple FB15K-237 human audit (plausible-but-false / likely-true / nonsense) of KGSAGE negatives — the cheapest partial substitute for a real-error arm. |

## 1. Milestone M0 — Run identity + RNG discipline (DO FIRST; everything else re-runs if this is late)

| Change | File / anchor |
|---|---|
| Cell-identity model label: pass `--model ADKGD_{neg_source}x{test_source}_s{seed}` so ckpt/log filenames (built from model name at `Our_TopK%_RankingList.py:128,137-138,323,331-332`) stop colliding across cells/seeds | `experiments/run_experiment.py` (derive label; no ADKGD edit needed) |
| Thread a seeded `torch.Generator` through z and Gumbel noise (currently GLOBAL torch RNG; the passed `rng` only picks slots) | `experiments/kgsage/inference.py:108-110,189`; passthrough in `bridge.py` |
| Update `self.num_anomalies = len(anomalies)` after the genuine-corruption filter (stale on shortfall; recall denominator + max_top_k derive from it) | `dataset.py:398 vs 423`; consumers `Our_TopK%_RankingList.py:397,439,467` |
| Per-run JSON (config + metrics + timings) emitted by the orchestrator; new aggregator producing mean±std per cell | `experiments/run_experiment.py`; new `experiments/aggregate_results.py` |

**Accept:** same-seed rerun ⇒ bit-identical negatives; two cells run concurrently ⇒ distinct artifacts; aggregator produces a table from ≥2 runs.

## 2. Milestone M1 — Close-but-false targets (the core science change)

New in-package plausibility module (kgsage stays ADKGD-agnostic):
- `experiments/kgsage/plausibility.py` + `experiments/kgsage/cli/train_kge.py`: minimal DistMult trainer
  (train split only — leakage rule §6), frozen scores cached per dataset. Sanity gate: filtered MRR
  reported and above a floor (≈0.20 FB15K-237 / ≈0.35 WN18RR) before any target uses it.

`experiments/kgsage/gan/targets.py` (`ContradictionTargetSampler`):
- Rename `_pick_context_distant` → `_pick_close_but_false`. Replace L156 `weights = (1.0 - cosine)` with
  KGE band sampling: rank the relation's type pool by `s(h,r,·)`, mask **all** known-true fillers of
  `(h,r,·)` / `(·,r,t)` across train+valid+test (not just this triple's value — 1-N relations), sample the
  band `[s(true)−m2, s(true)−m1]`.
- Pair-aware sourcing (kept, as sourcing only): candidates that are true fillers for an analogous entity
  in `N1(h)` sharing the relation ("real somewhere, wrong here"); precomputed (5-step: answer maps `A_r`,
  adjacency, cap hub degree ~100, subtract `A_r[h]` over all splits, KGE-band rank). Fall back to plain
  band when empty; **report pair-aware coverage %**. Prefer low-cardinality (quasi-functional) relations;
  frequency-debias within the band (hub-tail shortcut risk).
- Slots: **head/tail only** (`--slots ht` default) — drops the 1/3 uniform-random relation-swap supervision
  that inference never uses (L81, L85-93); resolves the brief's silence on the relation slot.
- `train.py:300-308`: pass `kg['triple_set_all']` into the sampler; keep GAN training on train split
  (`--train_split train`), collision set all splits (L285) — unchanged.

New fixture `data/fixture_converging_kg/` (**dummy_kg is unsuitable**: valid/test are copies of train so
cross-split masking is untestable; 4 of 6 persons are internally inconsistent; no near-miss tier):
49 train / 1 valid / 1 test triples, 3 regional clusters with `adjacent_to`/`located_in` near-miss
structure, 6 cases — Alice→NewZealand (2-hop spouse+employer), Dave→Korea (1-hop), Frank→Germany
(region-only, no pair-aware coworker), Bob head-slot (Carol), plus valid-split (Mia) and test-split (Jack)
masking probes that the CURRENT code provably fails. Full triple list + assertions A1–A11 in the
groundwork record; implement as `experiments/kgsage/tests/test_close_but_false.py`.
Key assertions: A1 single-slot head/tail only; A2 never the true value; A3 never any real triple
(split-specific probes); A4 type-valid 100%; A5–A7 intended near-miss is modal with freq ≥0.5 (tail) /
≥0.4 (head) over 200 draws; A8 masked-row fallback still corrupts (0 nulls); A9 new-vs-old sampler
direction inversion on the same frozen E'; A10 3-seed stability; A11 same invariants at the bridge surface.

New probe `experiments/kgsage/cli/probe_context.py`: Spearman corr(E'-cosine, graph k-hop distance) +
filtered-MRR of a linear decode of E' — one number that says whether the encoder learned structure.

**Flags:** `--target_mode {near,far}` (far kept for ablation), `--band_margin`, `--plausibility_band`,
`--pair_aware_frac` (default 0.5), `--slots {ht,hrt}` (default ht), `--kge_path`.
**Accept:** fixture A1–A11 pass (3 seeds); FB15K-237 sample of (positive → target) pairs reads as
near-misses; KGE MRR gate passes.

## 3. Milestone M2 — Make the GAN earn its name

`experiments/kgsage/gan/train.py`, `models.py`:
- **D_real positive class = (anchor, related-real-triple)** — a second real triple sharing h or (h,r)
  with the anchor (NOT (anchor, anchor): degenerate identity check; NOT the sampler target as now,
  L133-141). One-sided label smoothing 0.9; optional spectral norm on D_real (pick ONE stabilizer set
  up front).
- **Anti-truth guardrails** (the new equilibrium's failure mode is generating TRUE-but-unobserved facts,
  plus the copy equilibrium — 2/3 of entity-head supervision is currently copy-the-truth with no mask):
  training-time true-value + type-pool masks on the corrupted slot's logits (mirror inference);
  **one-slot candidates** with a slot-indicator appended to the conditioning vector and CE only on the
  corrupted slot's head; KGE **truth-penalty**: penalize candidates scoring above `s(true)−margin`;
  generation-time KGE top-rank filter with reported drop rate.
- **Recon anneal with a floor** (`--recon_weight_start 10 → --recon_weight_end 1`, never 0 — recon is the
  only force opposing truth-drift, and the checkpoint is saved at the END of training).
- Straight-through Gumbel (`hard=True`) or tau annealed to ~0.3; expose `--tau` (call sites L124-126,
  155-157 currently never pass it).
- Arms: `--arm {full, adv_only, recon_only, sampler_direct}`. `sampler_direct` emits the M1 sampler's
  output with no generator (needs E' from a trained run — note the arm cannot train the encoder itself).
- Per-epoch drift diagnostics (these become paper curves): pre-filter would-have-collided rate,
  fraction of candidates KGE-ranked ≤10 / above the true value, raw-argmax copy rate, D_real accuracy
  per class (logged as the frozen token `D-acc=`).
- Checkpoint: store type pools + arm/loss config; save **best-by-validation**, not last epoch.

**Accept:** all four arms train on fixture + dummy_kg without divergence; drift curves flat; D_real
accuracy neither 0.5 (artifact) nor 1.0 (collapse); fixture A5–A7 still pass through the full GAN.

## 4. Milestone M3 — Deployment-faithful inference

`experiments/kgsage/inference.py`:
- Pool-masked decode: `-inf` outside `tail_pool[r]`/`head_pool[r]` before Gumbel-argmax (L94-110, L223 —
  currently full vocab). Pools ship in the checkpoint (M2); fallback derivation for old checkpoints.
- Bounded resample loop (default 8) before recording `used_original`; return an `is_null` flag.
  *(`used_original` is a frozen stats key of the bridge API — it counts triples the decode could not
  corrupt, so the true_filler was kept.)*
- Stats: + type-valid rate, resample count (`used_original` + slot distribution already exist).
- Keep head/tail-only slot policy (L193-204) exactly.

ADKGD side: `dataset.py:291` (`get_data`) drops/replaces null negatives (eval path L421-422 already
filters — the asymmetry is the bug); fix stale docstring L301-308. `bridge.py` passes kwargs through.
`cli/inspect_corruptions.py`: **invert the diagnostic** — primary column becomes true_filler similarity
(high = good) + band percentile + a "corroborated by the anchor's neighbourhood?" column; make it call
the shared inference decode path so reports reflect deployment. Pool-size distribution report per dataset; small pools augmented via KGE
neighbours — never fall back to the full vocabulary.

**Accept:** fixture A11 at the bridge surface; FB15K-237 type-valid rate = 100% by construction,
post-resample null rate ≈ 0 and reported.

## 5. Milestone M4 — Baselines, matrix, metrics, hardness

Sources (train axis): `random` (bit-identical, untouched), `random_entity_slot` (new control),
`type_constrained` (new; per-relation pools from `Reader.triples`), `tric` (new
`tric_negatives.py` at repo root — corrected, seeded, collision-masked reimplementation; original left
untouched as reference), `sampler_direct`, `gan`. Test axis: `random`, `random_entity_slot`, `gan-B`
(**independent second checkpoint**: new `--test_gan_path` in `Our_TopK%_RankingList.py` + Reader payload
split + guard `test_gan_path != gan_path`; the second checkpoint needs no trainer changes — existing
`--seed`/`--out`).

Metrics: global AUC + AUPRC computed once over `(all_loss, all_label)` after `Our_TopK%_RankingList.py:393-402`
(do NOT uncomment L387-391 — that was per-batch), plus existing P@K/R@K; `np.savez` scores+labels per run.
SLURM: `MAX_EPOCH` env knob replacing the literal `--max_epoch 1` (exp1:65, exp2:75, exp3:74, exp4:73);
seed arrays `--array=0-2`; `TEST_GAN_CKPT` for gan-test cells.

**Hardness protocol (two separated claims — the load-bearing evidence absent a real-error arm):**
- Claim 1, intrinsic hardness (detector-free): score every source's anomalies + real test triples with
  FROZEN scorers of a *different family than the selection KGE* (train ComplEx alongside DistMult;
  selection = DistMult, measurement = ComplEx + cross-trained ADKGD detectors). Metrics: real-vs-anomaly
  AUC/AUPRC per scorer, score-margin distributions. New `experiments/analysis/hardness.py` over the npz dumps.
- Claim 2, training utility: the full cross-source transfer matrix; **headline = off-diagonal cells**
  (gan-trained detector on non-gan test columns). Never report a diagonal cell alone.
- Co-report per source: slot distribution, type-valid rate, null rate, KGE-likely-true rate (FN proxy) —
  hardness is trivially maximized by generating true facts, so hardness and falseness are claimed jointly.

**Run budget (per dataset):** Tier 1 (core): {random, random_entity_slot, type_constrained, sampler_direct,
gan} × {random_entity_slot, gan-B} × 3 seeds = 30 runs. Tier 2: + tric row, + random test column. GAN
training: 2 seeds × 2 datasets + 1 KGE pair per dataset.

**Accept:** aggregated mean±std tables; hardness report; composition table; pre-registered success criteria
(off-diagonal wins; FN-proxy ceiling e.g. <10% of emitted negatives KGE-ranked top-1) written down BEFORE
the matrix runs.

## 6. Leakage rules (one paragraph, absolute)

GAN, selection KGE, measurement KGE, and any rule miner train on the **TRAIN split only**. Only
collision/masking **sets** use all splits (that is label hygiene, not learning). The `--gan_path` /
`--test_gan_path` checkpoints must match `--dataset` (`--gan_path` is a frozen detector-CLI flag; its
default is the promoted artifact `generator_<dataset>.pt`) (add a dataset-name + triple-set-hash guard in the
checkpoint; currently a mismatch is a bare KeyError or silent). Auxiliary-scorer leakage contaminates both
the training negatives and the hardness evidence.

## 7. Commit plan

One commit per milestone on `dev_gan_1`: `M0: run identity + RNG determinism`, `M1: close-but-false
targets + fixture`, `M2: plausibility discriminator + arms`, `M3: pool-masked inference + null filtering`,
`M4: baselines + matrix + hardness`. Verify each on `data/fixture_converging_kg/` + `dummy_kg` (CPU,
minutes) before FB15K-237/WN18RR. Env: pytorch conda env; `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
KMP_DUPLICATE_LIB_OK=TRUE` mandatory.

## 8. TAXO wording constraints (from experiments/docs/ADKGD_TAXO_findings.md — quote-verified)

The claim is **depth on TAXO #2, not breadth**: coverage "shifts from mostly TAXO #3 by accident to TAXO #2
by design"; adding a generator "does not automatically expand the detector's TAXO coverage". Random-baseline
comparisons are #3-negatives vs #2-negatives comparisons. Drop remaining "contradiction" language from specs
and docstrings — the implemented object is a type-coherent plausible-but-false near-miss (#2), not a
multi-triple contradiction (#5).
