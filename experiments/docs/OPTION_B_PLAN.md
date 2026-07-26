# Option B — LP Band Sampler (PoC control, no GAN)

> ⚠️ **HISTORICAL DOCUMENT — path not taken.** The link-predictor route described
> here (`lp_scorer.py`, `band_sampler.py`, a frozen ComplEx signal) was explored
> and then removed: the shipped architecture is **LP-free**, using the
> dual-discriminator design instead. The modules named below do not exist in the
> codebase. Kept for provenance. Current names: `KGSAGE_glossary.md`.

Project Milestone 1 of the staged path B → A → C (a milestone of the plan, not a pipeline phase).
Everything here carries forward into Option A (LP-in-the-loop GAN) and the full plan
(KGSAGE_IMPLEMENTATION_PLAN.md): the scorer, the band sampler (= the future `sampler_direct`
ablation arm), the masks, and the hygiene fixes.

## 0. Objective and pre-registered success criteria

**PoC question:** do hard, plausible-but-false (close-but-false) negatives change ADKGD's ability to
detect neighbourhood-inconsistent anomalies, relative to random corruption?

**Matrix (24 runs):** `neg_source ∈ {random, lp_band}` × `test_anomaly_source ∈ {random, lp_band}`
× seeds {0,1,2} × {FB15K-237, WN18RR}. *(`neg_source` / `test_anomaly_source` are frozen detector-CLI
flag names shared by `run_experiment.py`, `exp_cell.slurm` and `Our_TopK%_RankingList.py`.)*

Pre-registered read-out (adjust numbers before first cluster run, then freeze):
1. **Loader gate:** our standalone scorer reproduces LibKGE's published filtered MRR on our own
   valid split (±0.01): ComplEx 0.348 FB15K-237 / 0.475 WN18RR. This single number proves the
   checkpoint parse, re/im layout, and string-remap are all correct.
2. **The gap exists:** (random-train, lp_band-test) drops materially vs (random-train, random-test)
   — evidence that random-corruption training does not prepare the detector for hard anomalies.
3. **The recovery:** (lp_band-train, lp_band-test) recovers ≥ half of that gap.
4. **No regression:** (lp_band-train, random-test) within seed-noise of (random, random).
5. **FN control:** fraction of candidates dropped for ranking above the true value < 15%, reported
   per dataset; emitted negatives' LP-top-1 rate ≈ 0 by construction.

Disclosure to carry into any write-up: in this PoC the hard test anomalies come from the same scorer
family that generates training negatives (different seed, but same ComplEx). That self-grading is
acceptable for a PoC and is exactly what the full plan's off-family hardness protocol later removes.

## 1. Which link predictor — decision

**Primary: ComplEx, LibKGE "You CAN Teach an Old Dog New Tricks!" (Ruffinelli et al., ICLR 2020)
pretrained checkpoints. Zero training.**

| | URL |
|---|---|
| FB15K-237 ckpt | `http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-complex.pt` |
| WN18RR ckpt | `http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/wnrr-complex.pt` |
| ID maps | LibKGE preprocessed dataset archives (`kge-datasets/fb15k-237.tar.gz`, `wnrr.tar.gz`) → `entity_ids.del`, `relation_ids.del` |

Why ComplEx: (a) handles antisymmetric relations — mandatory for WN18RR, where ~57% of triples are
hierarchy relations and symmetric DistMult would conflate parent/child direction; (b) near-SOTA for
shallow models with fully citable published numbers; (c) scoring is one vectorised complex
tri-linear product (~20 lines from raw tensors), no framework import; (d) trained on the standard
train split only → our leakage rule holds by construction. Why not: PyKEEN (installable on 3.13 —
`requires_python >=3.10`, no cap — but last released Dec 2024/untested on 3.13, ships NO pretrained
FB15K-237/WN18RR checkpoints so it would reintroduce the training we're avoiding, and drags
optuna/pandas/sklearn/click onto an env that needed manual NumPy-2.x/Py3.13 patches to run at all;
if the self-train fallback is ever chosen, PyKEEN in an ISOLATED 3.11 env exporting raw .npz tensors
is a legitimate variant), RotatE (equal quality, more scoring code — reserve as the *off-family
measurement* scorer for the full plan so selection/measurement stay separated), GNN LPs (framework-
bound, overkill for ranking a type pool).

**Fallback (decision gate, see risks):** if the checkpoint unpickle shim gets ugly, self-train
ComplEx with a compact in-repo trainer (~150 lines, 30–60 GPU-min per dataset). Same API, same gate.

## 2. Components

### B0 — Hygiene prerequisites (~25 lines)
- `dataset.py` `inject_anomaly`: `self.num_anomalies = len(anomalies)` after the genuine-corruption
  filter (stale-denominator fix; L398 vs L423).
- `run_experiment.py`: derive `--model ADKGD_{neg}x{test}_s{seed}` so ckpt/log filenames stop
  colliding across cells/seeds.

### B1 — Scorer module: `experiments/kgsage/lp_scorer.py` (~120–160 lines)
- `load_complex(ckpt_path, entity_ids_path, relation_ids_path, our_ent2id, our_rel2id)`:
  `torch.load` with a stub-`kge`-module custom Unpickler (LibKGE checkpoints pickle Config objects;
  a `find_class` returning dummy types extracts the tensors, ~30 lines). Pull entity/relation
  embedding matrices; determine re/im layout at runtime; **remap rows by strings** from LibKGE's
  `.del` maps onto our vocab (both sides use standard splits — same MIDs/synset offsets).
- `score_tails(h, r, cand_ids) / score_heads(cand_ids, r, t)`: batched ComplEx score.
- `filtered_mrr(valid_triples, all_true)`: the loader gate. Run once, assert ≈ published, log it.
- Entities in our vocab but absent from LibKGE's (should be none on standard splits — assert) get
  score −inf and are excluded from bands.

### B2 — Band sampler: `experiments/kgsage/band_sampler.py` (~150–180 lines)
Stays inside `kgsage/` (ADKGD-agnostic). Precompute once per dataset (all torch, chunked per
relation — WN18RR's two giant-pool relations demand it):
1. Type pools from observed train slot fillers (pattern of `targets.py:67-76`).
2. All-splits answer maps `A_r[h]` / inverse for masking every known-true filler (1-N safety).
3. Per positive triple: slot 50/50 head/tail (seeded numpy rng) → pool → mask `A_r` + self-loop →
   score with LP → **drop candidates scoring above s(true)** (count = FN proxy) → take the top
   `band_k` (default 10) *by rank, not absolute margin* (scores aren't calibrated across relations)
   → softmax-sample at temperature `band_temp` (default 0.5).
4. Fallbacks, each counted and reported: band empty → widen to all-below-true → still empty →
   type-valid random draw. No nulls ever emitted (contract: 1 genuine negative per positive).

### B3 — ADKGD wiring (~100 lines)
- `experiments/kgsage_bridge/bridge.py`: add `load_lp()` / `generate_band()` (mirror of
  `load_gan`/`generate`; string round-trip identical).
- `dataset.py`: `neg_source='lp_band'` branch in `get_data` (mirror `_gan_negatives`, payload cached
  once); `test_anomaly_source='lp_band'` branch in `inject_anomaly` reusing the oversample-then-
  filter pattern, **with a different seed offset** than the train-negative draw.
- `Our_TopK%_RankingList.py` + `run_experiment.py`: `--lp_path` argument, extended choices.

### B4 — Diagnostics: `experiments/kgsage/cli/inspect_band.py` (~80 lines)
For N sampled positives print: positive → negative, s(neg), rank-of-neg in pool, s(true), gap, pool
size, masked-count, fallback flags; WN18RR rows resolved to synset names (reuse the `--wordnet`
logic from `inspect_corruptions.py`), FB15K-237 via optional `--labels mid2name.tsv`. Plus the
aggregate table: FN-dropped %, band-coverage %, fallback %, slot distribution.

### B5 — Metrics + runs (~90 lines)
- Global ROC-AUC + AUPRC over `(all_loss, all_label)` after `Our_TopK%_RankingList.py:393-402`
  (do NOT uncomment the old per-batch block), `np.savez` scores+labels per run.
- `run_experiment.py`: parse/print the two new rows; emit per-run JSON (config + metrics).
- SLURM: `MAX_EPOCH` env knob in exp templates; a 12-cell array per dataset. Local smoke on
  `dummy_kg` first (CPU, minutes).

## 3. Sequencing (2–3 engineering days + cluster time)

| Day | Work | Gate |
|---|---|---|
| 1 am | Download ckpts + `.del` maps; B1 loader + shim | **MRR gate** on both datasets |
| 1 pm | B2 band sampler; sanity on dummy_kg + 50 eyeballed WN18RR samples via B4 | negatives read as near-misses; FN% + coverage printed |
| 2 | B0 + B3 wiring + B5 metrics; full local dummy_kg run of all 4 cells | RESULTS table renders; seeds reproducible |
| 2–3 | 24 cluster runs; readout against pre-registered criteria | PoC verdict |

## 4. Risks and fallbacks

| Risk | Likelihood | Fallback |
|---|---|---|
| LibKGE checkpoint unpickle needs `kge` classes | medium | stub-module Unpickler (~30 lines); if ugly → self-train ComplEx (~150 lines, 30–60 GPU-min) |
| Re/im layout ambiguity in ComplEx tensors | low | MRR gate catches it deterministically; try both layouts |
| WN18RR band emptiness on 1-N relations after all-splits masking | medium | widen-then-random fallback, counted; huge-pool relations chunked |
| Vocab mismatch LibKGE vs our splits | low | assert full string coverage at load; standard splits on both sides |
| (random→lp_band-test) gap doesn't appear | possible | that is itself a PoC result: hard negatives from a shallow LP don't challenge ADKGD → motivates the GAN/neighbourhood route (Option A) rather than killing it |
| Same-family test circularity | certain | different seed draw + explicit disclosure; removed later by full plan's off-family protocol |

## 5. What Option B deliberately does not do

No GAN (Option A), no RGCN conditioning used, no pair-aware sourcing, no fixture battery, no TRIC /
type_constrained / random_entity_slot arms, no hardness.py, no real-error arm. Upgrade path: B2
becomes the `sampler_direct` arm of Option A/C verbatim; B1 becomes the truth-penalty and FN-proxy
scorer; B0/B5 are the M0/M4 fixes the full plan needs anyway.
