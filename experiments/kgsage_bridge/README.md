# KGSAGE <-> ADKGD bridge

The **only ADKGD-aware module** in `experiments/`: `dataset.py` (repo root)
imports this six-function API; `kgsage/` itself never imports ADKGD code.

## Contract (`bridge.py`)

GAN source — used for `--neg_source gan` / `--test_anomaly_source gan`:

```
load_gan(ckpt_path)                  -> payload (generator + vocab + reals + masks)
generate(triples, payload=…, …)     -> (negatives, stats)   # ADKGD-id in/out
render_stats(stats)                  -> log line
```

lp_band source — used for `--neg_source lp_band` / `--test_anomaly_source lp_band`
(Option B, frozen-LP band sampler; no checkpoint, no nulls):

```
load_lp(lp_ckpt, lp_ids_dir, adkgd_data_dir, band_k=…, band_temp=…) -> payload
generate_band(triples, payload=…, …) -> (negatives, stats)
render_band_stats(stats)             -> log line
```

Key semantics the Reader relies on:

- **String round-trip**: ADKGD ids → strings → generator ids and back, so the
  two integer vocabularies never need to agree.
- **1:1, order-aligned** positives:negatives (ADKGD pairs index `i` with
  `i+n`).
- **Null handling**: `generate`'s `stats["null_indices"]` flags rows whose
  emitted "negative" is the original triple; the Reader replaces those in the
  training role and filters them in the eval role. `generate_band` never
  emits nulls (counted fallback ladder instead).
- `load_lp`'s masks/pools are built from **ADKGD's own data dir** (the graph
  being trained on), while the scorer's id maps come from the LibKGE archive
  dir — two different directories by design.
