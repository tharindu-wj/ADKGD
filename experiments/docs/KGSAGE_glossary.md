# KGSAGE — Canonical Glossary

> **Purpose.** One name per concept, across all of `experiments/`. This file is
> the specification the codebase follows: if a term is not in the *Canonical*
> column, it should not appear in new code, comments, docs or figures.
>
> **Two categories.** Most names are *free* and are unified. A minority are
> **FROZEN** — renaming them would break a saved checkpoint, a `state_dict`, a
> cross-script file format, or an API that repo-root `dataset.py` imports.
> Frozen names stay, and each carries a one-line decoder comment where it
> appears. 100% uniformity is therefore impossible by construction; 100%
> *decodability* is the standard we hold.

---

## 1. Why some names cannot change

| Contract | Examples | Breaks if renamed |
|---|---|---|
| Checkpoint payload keys | `dreal_state`, `dmatch_state`, `generator_state`, `context_embeddings`, `sketches`, `pool_masks`, `arch`, `alpha_target`, `alpha_final`, `cand_k`, `dim`, `tau` | Every saved `.pt` — including the archived locked artifacts — becomes unloadable |
| `nn.Module` attribute names | `rel_embedding`, `net`, `q_proj`, `k_proj`, `v_proj`, `score`, `relation_embedding`, `sketch_proj`, `trunk`, `q_head`, `q_tail`, `cand_tower`, `entity_embeddings`, `rgcn_layers` | They *are* the `state_dict` keys of the locked artifacts |
| Architecture string | `"candidate_v2"` | Compared as a literal in `corruption_generation.py` and `knockout_eval.py` |
| Bridge public API | `load_gan`, `generate`, `render_stats`, and the stats keys `used_original`, `null_indices` | Imported by repo-root `dataset.py` (outside `experiments/`) |
| Detector CLI contract | `--neg_source`, `--test_anomaly_source`, values `random`\|`gan`, `--gan_path`, env `NEG_SOURCE`/`TEST_SOURCE`/`GAN_CKPT` | `run_experiment.py` ↔ `exp_cell.slurm` ↔ `Our_TopK%_RankingList.py` |
| CSV column format | `orig_head/orig_relation/orig_tail`, `corr_head/corr_relation/corr_tail` | Written by `gen_corruptions_csv.py`, read by `ego_from_csv.py`, `format_for_llm.py`, `gen_neighbourhood_context.py` |
| Dataset literals | registry keys `fb15k237`/`wn18rr`/`yago45`/`fb15k_mini`; directories `FB15K-237`/`WN18RR`/`YAGO4.5`/`FB15K-mini` | Registry lookups, on-disk paths, SLURM `case` labels |
| Runtime log tokens | `corr-pick=`, `alpha=`, `D-acc=`, `g_match=`, `dm-online=`, `distinct=`, `LP_loss=` | Comparability with every training log already collected on HPC/Colab |

---

## 2. Components

> **One name per component — no short symbols.** Prose, comments and docstrings
> use the full words: **the plausibility discriminator** and **the neighbourhood
> discriminator**. The abbreviations `D_real` / `D_match` are **retired from all
> writing**; they survive only inside the frozen tokens listed below, where each
> carries a decoder comment. Writing `D_real` in a comment re-introduces exactly
> the two-names-for-one-thing problem this glossary exists to remove.

| Concept | **Canonical** identifier | Says, in words | FROZEN aliases (keep + decoder comment) |
|---|---|---|---|
| Judge: "could this triple be real?" | `PlausibilityDiscriminator`, `plausibility_discriminator`, `plausibility_*` | *the plausibility discriminator* | `dreal_state`, `dreal-pre`, `D-acc=` |
| Judge: "does the filler fit this anchor's neighbourhood?" | `NeighbourhoodDiscriminator`, `neighbourhood_discriminator`, `neighbourhood_*` | *the neighbourhood discriminator* | `dmatch_state`, `dmatch`, `dm-online=`, `g_match=` |
| Proposes the corruption | `CandidateScoringGenerator`, `generator` | *the generator* | `generator_state` |
| Builds the context table | `NeighbourhoodContextEncoder`, `context_encoder` | *the context encoder* | — |
| Throwaway warm-up decoder | `distmult_decoder` | *the DistMult decoder* | — |
| Per-triple candidate sets | `CandidateSampler`, `candidate_sampler` | *the candidate sampler* | `cand_k`, `cand_tower` |

Write the generator's objective in words too, not symbols:

> maximise **plausibility**, minus α times the hinged **neighbourhood fit**.

**Retired** (do not use): `D_real`, `D_match`, `G` as a standalone symbol,
"realism discriminator", "consistency discriminator", "real-fact discriminator",
"Plausibility-anchored discriminator", "critic", "matcher"/"matching-aware D",
bare "Discriminator", `KGSAGEEncoder`.

---

## 3. Data objects

| Concept | **Canonical** | FROZEN aliases |
|---|---|---|
| Frozen per-entity context table | `context_table` (variables, parameters) · **E'** in prose (ASCII apostrophe) | `context_embeddings` (payload key) |
| RGCN layer-0 **input** features | `entity_embeddings` — **input only** | (is the `nn.Module` attribute) |
| Bloom membership sketches | `membership_sketches` · "membership sketch" in prose | `sketches`, `sketch_proj` |
| Type pools | `pool_masks` (tensor) · `type_pool` (prose) | `pool_masks` |

> ⚠️ **Never call E' "entity embeddings".** In code `entity_embeddings` is the
> encoder's **input**; E' is its **output**. Docs previously inverted this.

---

## 4. One corruption — the per-triple vocabulary

| Concept | **Canonical** | Notes |
|---|---|---|
| Entity that **keeps** its slot | `anchor` | — |
| Original value of the corrupted slot | `true_filler` | was: `clean`, `clean_value`, `clean_index` |
| One entity in the sampled set | `candidate` | set size = `NUM_CANDIDATES` |
| The candidate the generator selected | `picked_candidate` / `picked_*` | log token `distinct=` counts these |
| The other entity of the triple (self-loop ban) | `other_entity` | was: `self_row` |
| The emitted false triple | **corruption** | inside `kgsage/` always "corruption" |
| Same object, detector side | **negative** (training) / **anomaly** (evaluation) | legitimate seam — ADKGD's vocabulary, not drift |

---

## 5. The oracle relation, the controller, and the pressure

The relation *"the anchor's neighbourhood vouches for candidate x"* had two
competing word families (*support* vs *corroborate*). **Canonical: corroborate.**

| Concept | **Canonical** | FROZEN aliases |
|---|---|---|
| The oracle relation | `corroborated` / `corroboration` | `corr-pick=` (log), `support_max` (public kwarg of `generate_negatives`) |
| Fraction of picks the graph corroborates | `corroborated_fraction` | `corr-pick=`, `alpha_target` (payload) |
| PI set-point for that fraction | `CORROBORATION_TARGET` | `alpha_target` (payload key stores it) |
| Weight on the neighbourhood penalty | `alpha` | `alpha=`, `alpha_final`, `alpha_target` |
| What the generator is pushed toward | **contradiction** (`CONTRADICTION_MARGIN`) | — |
| Decode-time mask of corroborated candidates | `corroboration_mask` | `support_max` kwarg |

**Retired**: "alienation"/"alien set" (use *contradiction*), "support-mass",
"penalty weight" as a standalone noun, λ for α. The controller is a **PI**
controller (proportional + integral) — never write "PID".

> ⚠️ **The `corr` prefix is overloaded and must stay that way.** In the eval
> CSVs `corr_*` = **corrupted** (frozen column names). In the trainer log
> `corr-pick` = **corroborated**. Both are frozen; every occurrence carries a
> decoder comment. Do not introduce any *new* `corr_` name in either sense.

---

## 6. Pipeline phases

Aligned with the paper. Use these names, not bare numbers.

| Canonical | What happens |
|---|---|
| **Phase 1 — Neighbourhood Context Encoding** | RGCN + throwaway DistMult decoder → E' frozen; membership sketches built |
| **Phase 2 — Adversarial Generator Training** | 2a discriminator pretraining · 2b the dual-discriminator game |
| **Phase 3 — Corruption Generation** | trained checkpoint → one corruption per input triple |

Inside Phase 2, the first epochs run with α = 0: call this the
**plausibility-only phase** (`EPOCHS_BEFORE_CONTRADICTION`) — **never**
"warm-up", which is reserved for the Phase-1 RGCN warm-up.

> The constant is deliberately *not* named `PLAUSIBILITY_ONLY_EPOCHS`: that read
> as a sibling of `PLAUSIBILITY_PRETRAIN_EPOCHS`, but the two are different in
> kind — one counts **pretraining** epochs for the plausibility discriminator
> *before* the game, the other counts **game** epochs run at α = 0.

**Retired**: "Phase 0b", "Phase B", "Stage 1/2", "B1a", and using "Phase N" for
investigation rounds or thesis milestones — those are **"Investigation Round N"**
and **"Project Milestone N"** respectively.

---

## 7. Architecture and artifact naming

| Concept | **Canonical** | FROZEN |
|---|---|---|
| The architecture | "dual-discriminator architecture" | `candidate_v2` (arch string) |
| A training run's checkpoint | `run_<tag>_s<seed>.pt` | — |
| A promoted/reported artifact | `generator_<dataset>.pt` | default of `--gan_path` |

**Retired in prose**: "KGSAGE-2", "kgsage2", bare "v1"/"v2". Historical run tags
(`kgsage2b`, `kgsage2c`, `v2_fb_dyn5`) are fine when naming a specific past run.

---

## 8. Evaluation vocabulary

| Concept | **Canonical** |
|---|---|
| Anchor-dependence metric | **knockout J@10** (`knockout_j10`) |
| Anchor-invariance companion | **cross-head J@10** (`xhead_j10`) — not "x-head" |
| Generator ignores the anchor | **collapse** / "universal-alien collapse" |
| Desired property | **anchor-specific** (not "anchor-aware"/"anchor-conditional") |
| The true-triple evaluation arm | **control** (`--which control` in both LLM CLIs) |

---

## 9. Package layout note

`kgsage/gan/` holds the whole adversarial training stack — encoder, sketches,
candidate sampler and both discriminators — not only the GAN. The folder name is
kept because `python -m kgsage.gan.train` is a documented entry point used by the
SLURM launchers, `commands.md` and the notebook. Read `gan/` as
*"the adversarial training stack"*.
