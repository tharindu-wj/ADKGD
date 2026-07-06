# kgsage — the standalone anomaly-generator package

Self-contained (imports nothing outside `kgsage.*`). Downstream-detector
integration lives exclusively in `experiments/kgsage_bridge/`.

## Two negative sources, one frozen scorer

- **Frozen LP (Option B):** `lp_scorer.py` loads the published LibKGE
  ICLR-2020 ComplEx checkpoints as raw tensors (no libkge/pykeen install) and
  must reproduce the published filtered MRR (0.348 FB15K-237 / 0.475 WN18RR)
  via its gate before anything trusts the scores. `band_sampler.py` turns it
  into the close-but-false `lp_band` source: type-valid (train pools), false
  by construction (all-splits known-true masks + self-loop bans), plausible
  (top-k by rank below s(true)); counted fallbacks, never a null.
- **KGSAGE GAN:** `gan/train.py` — RGCN LP-warmup → **freeze E'**
  → generator warm-start toward band-teacher draws → adversarial phase where
  D = per-relation z-scored frozen ComplEx **+ trainable contextual residual**
  (`gan/residual_d.py`, candidate-only input, β·tanh-bounded) and G samples
  one entity slot via masked straight-through Gumbel under a frozen fence
  below s(true). Falseness is structural (`gan/masks.py`), plausibility is
  frozen (`gan/complex_d.py`); only G and f_θ ever train.

## Generation API

`inference.py` (PyG-free; conditions on the checkpoint's cached E'): one
negative per input triple; head/tail slot by corruptibility (relation slot is
never chosen); decode masked by the true value + every known-true filler +
self-loop (+ the type pool when present); seeded `torch.Generator`
(bit-reproducible per seed); bounded resample, then a flagged null
(`stats["null_indices"]`) — callers must never train on nulls (the
bridge/Reader handles this).

## CLIs (repo root, `PYTHONPATH=experiments`)

```
python -m kgsage.cli.fetch_lp             # download LP ckpts + MRR gate
python -m kgsage.gan.train               # the trainer
python -m kgsage.cli.inspect_band         # lp_band diagnostics (gap/rank/FN)
python -m kgsage.cli.inspect_gan_lp       # GAN diagnostics via deployed decode
python  experiments/kgsage/smoke_test.py  # package + bridge smoke
```

SLURM launchers in `slurm/` (see that folder's README). Outputs land under
`outputs/` — gitignored except the committed MRR gate reports.

Datasets registered in `data/datasets.py`: `fb15k237`, `wn18rr`,
`fb15k_mini`, `dummy_kg` — only directories that exist in this checkout.
