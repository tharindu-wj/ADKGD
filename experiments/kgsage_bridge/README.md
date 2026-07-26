# KGSAGE <-> ADKGD bridge

The **only ADKGD-aware module** in `experiments/`: `dataset.py` (repo root)
imports this three-function API; `kgsage/` itself never imports ADKGD code.

## The vocabulary seam

This folder is where one object changes its name, on purpose:

| Side | Word | Why |
|---|---|---|
| Inside `kgsage/` | **corruption** | the package's own term for the false triple it emits |
| ADKGD, training | **negative** | what the detector is trained against |
| ADKGD, evaluation | **anomaly** | what the detector is scored on |

Same triple, three words, each correct on its own side of `bridge.py`.
Translating between them is this module's job — so the `gan`/`negative`
wording in the API below is deliberate, not leftover drift. See
`experiments/docs/KGSAGE_glossary.md` §4.

## Contract (`bridge.py`)

All three function names below are **frozen** — repo-root `dataset.py`
imports them by name, and they pair with the detector's frozen flags
`--neg_source gan` / `--test_anomaly_source gan` / `--gan_path`:

```
load_gan(ckpt_path)                  -> payload (generator + context table E'
                                        + membership sketches + vocab + reals
                                        + type pools)
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
  training role and filters them in the eval role. `null_indices` and
  `used_original` are frozen stats keys.
