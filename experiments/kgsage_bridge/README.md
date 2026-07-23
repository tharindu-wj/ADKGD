# KGSAGE <-> ADKGD bridge

The **only ADKGD-aware module** in `experiments/`: `dataset.py` (repo root)
imports this three-function API; `kgsage/` itself never imports ADKGD code.

## Contract (`bridge.py`)

GAN source — used for `--neg_source gan` / `--test_anomaly_source gan`:

```
load_gan(ckpt_path)                  -> payload (generator + vocab + reals + masks)
generate(triples, payload=…, …)     -> (negatives, stats)   # ADKGD-id in/out
render_stats(stats)                  -> log line
```

Key semantics the Reader relies on:

- **String round-trip**: ADKGD ids → strings → generator ids and back, so the
  two integer vocabularies never need to agree.
- **1:1, order-aligned** positives:negatives (ADKGD pairs index `i` with
  `i+n`).
- **Null handling**: `generate`'s `stats["null_indices"]` flags rows whose
  emitted "negative" is the original triple; the Reader replaces those in the
  training role and filters them in the eval role.
