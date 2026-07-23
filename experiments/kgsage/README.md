# kgsage — the standalone anomaly-generator package

Self-contained (imports nothing outside `kgsage.*`). Downstream-detector
integration lives exclusively in `experiments/kgsage_bridge/`.

## The generator (dual-discriminator, `candidate_v2`)

`gan/train.py` trains an LP-free adversarial generator:

1. **Phase 1 — context.** `gan/encoder.py` (RGCN) is warmed up with a DistMult
   decoder, then the decoder is discarded and the context table **E' is frozen**;
   `gan/sketch.py` caches a Bloom membership sketch of each entity's 1–2 hop
   neighbour set.
2. **Phase 2 — the game.** `gan/generator.py` (`CandidateScoringGenerator`) scores
   a per-triple candidate set (`gan/candidates.py`, logQ-corrected) and selects
   one via straight-through Gumbel-Softmax. Two discriminators judge the pick:
   `gan/d_real.py` (realism, spectral-normed, wrong-anchor class) and
   `gan/d_match.py` (neighbourhood consistency, cross-attention). The generator
   raises realism under a hinge penalty whose weight a PI controller holds at a
   target corroborated-selection rate.
3. **Snapshots + selection.** A checkpoint is saved every adversarial epoch;
   `cli/knockout_eval.py` picks the snapshot whose ranking depends most on the
   anchor's neighbourhood (lowest mean knockout J@10).

No link predictor anywhere — falseness/type-validity are guaranteed by the
decode masks, not by training.

## Generation API

`inference.py` (PyG-free; conditions on the checkpoint's cached E' + sketches):
one negative per input triple; head/tail slot; the pick is decoded by scoring
the full type pool and masking the true value + every known-true filler
(all splits) + self-loop; seeded `torch.Generator` (bit-reproducible per seed);
bounded resample, then a flagged null (`stats["null_indices"]`) — callers must
never train on nulls (the bridge/Reader handles this). Only `candidate_v2`
checkpoints load; legacy v1 checkpoints remain valid only as
`--init_context_from` E' donors for the trainer.

## CLIs (repo root, `PYTHONPATH=experiments`)

```
python -m kgsage.gan.train                          # the trainer (snapshots)
python experiments/kgsage/cli/knockout_eval.py      # snapshot SELECTION (J@10)
python experiments/kgsage/cli/gen_corruptions_csv.py # eval CSV (7.3 + 7.4)
python experiments/kgsage/cli/ego_from_csv.py       # ego graphs from the CSV (7.4)
python experiments/kgsage/smoke_test.py             # package + bridge smoke
```

SLURM launchers in `slurm/` (see that folder's README). Outputs land under
`outputs/` — gitignored.

Datasets registered in `data/datasets.py`: `fb15k237`, `wn18rr`,
`fb15k_mini`, `dummy_kg` — only directories that exist in this checkout.
