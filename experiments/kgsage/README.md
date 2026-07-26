# kgsage — the standalone corruption-generator package

Self-contained (imports nothing outside `kgsage.*`). Downstream-detector
integration lives exclusively in `experiments/kgsage_bridge/`.

Naming follows `experiments/docs/KGSAGE_glossary.md`. Inside this package a
generated false triple is always a **corruption**; only the detector side calls
it a *negative* (training) or an *anomaly* (evaluation).

Read `gan/` as *the adversarial training stack* — it holds the encoder,
sketches, candidate sampler and both discriminators, not only the GAN. The
folder name is kept because `python -m kgsage.gan.train` is a documented entry
point.

## The generator (dual-discriminator architecture)

`gan/train.py` trains an LP-free adversarial generator. The checkpoint records
this architecture as the string `candidate_v2` — a frozen literal, compared
directly by `corruption_generation.py` and `cli/knockout_eval.py`.

1. **Phase 1 — Neighbourhood Context Encoding.**
   `gan/neighbourhood_context_encoder.py` (RGCN) is warmed up against a
   throwaway DistMult decoder; the decoder is then discarded and the context
   table **E' is frozen**. `gan/membership_sketch.py` caches a Bloom membership
   sketch of each entity's 1–2 hop neighbour set. E' is the encoder's *output* —
   never call it "entity embeddings", which is the name of its layer-0 *input*.
2. **Phase 2 — Adversarial Generator Training.** After the discriminators are
   pretrained (2a), the dual-discriminator game runs (2b).
   `gan/generator.py` (`CandidateScoringGenerator`) scores a per-triple
   candidate set (`gan/candidate_sampler.py`, logQ-corrected) and selects one
   candidate via straight-through Gumbel-Softmax. Two discriminators judge the
   picked candidate: the **plausibility discriminator**
   (`gan/plausibility_discriminator.py`, `PlausibilityDiscriminator`) — "could
   this triple be real?", spectral-normed, wrong-anchor class — and the
   **neighbourhood discriminator** (`gan/neighbourhood_discriminator.py`,
   `NeighbourhoodDiscriminator`) — "does the filler fit this anchor's
   neighbourhood?", cross-attention. The generator raises plausibility under a
   hinge contradiction penalty whose weight `alpha` a **PI** controller holds at
   `CORROBORATION_TARGET` — the target fraction of picks the training graph
   corroborates. The first epochs run at α = 0: that is the
   **plausibility-only phase** (`EPOCHS_BEFORE_CONTRADICTION`), distinct from the
   Phase-1 RGCN warm-up.
3. **Snapshots + selection.** A checkpoint is saved every adversarial epoch;
   `cli/knockout_eval.py` picks the most **anchor-specific** snapshot — the one
   whose ranking depends most on the anchor's neighbourhood (lowest mean
   knockout J@10).

No link predictor anywhere — falseness/type-validity are guaranteed by the
decode masks, not by training.

## Phase 3 — Corruption Generation (the public API)

`corruption_generation.py` (PyG-free; conditions on the checkpoint's cached
context table E' + membership sketches): one corruption per input triple;
head/tail slot; the pick is decoded by scoring the full type pool and masking
the `true_filler` + every known-true filler (all splits) + the `other_entity`
of the triple (self-loop ban); seeded `torch.Generator` (bit-reproducible per
seed); bounded resample, then a flagged null. `stats["null_indices"]` is a
frozen stats key holding those failed rows — callers must never train on nulls
(the bridge/Reader handles this). Only `candidate_v2` payloads load;
pre-`candidate_v2` checkpoints remain valid only as `--init_context_from` E'
donors for the trainer.

## CLIs (repo root, `PYTHONPATH=experiments`)

```
python -m kgsage.gan.train                              # Phases 1-2 (snapshots)
python experiments/kgsage/cli/knockout_eval.py          # snapshot SELECTION (knockout J@10)
python experiments/kgsage/cli/gen_corruptions_csv.py    # eval CSV (7.3 + 7.4)
python experiments/kgsage/cli/ego_from_csv.py           # ego graphs from the CSV (7.4)
python experiments/kgsage/cli/gen_neighbourhood_context.py  # neighbourhood-context blocks (7.4 semantic)
python experiments/kgsage/cli/format_for_llm.py         # paste-ready triple blocks (7.3)
python experiments/kgsage/smoke_test.py                 # package + bridge smoke
```

The eval CSVs use the frozen column names `orig_head/orig_relation/orig_tail`
and `corr_head/corr_relation/corr_tail`, where `corr_` means **corrupted**.
(Beware the overloaded prefix: in the trainer log `corr-pick=` means
**corroborated**. Both are frozen; do not add new `corr_` names in either
sense.)

SLURM launchers in `slurm/` (see that folder's README). Outputs land under
`outputs/` — gitignored.

Datasets registered in `data/datasets.py`: `fb15k237`, `wn18rr`, `yago45`,
`fb15k_mini`, `dummy_kg` — these literals are frozen (registry lookups,
on-disk paths and SLURM `case` labels all key off them).
