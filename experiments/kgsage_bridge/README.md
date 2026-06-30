# KGSAGE <-> ADKGD bridge

The **integration layer** between the KGSAGE generation package
(`experiments/kgsage/`) and ADKGD's anomaly-detection training pipeline
(`dataset.py`, `model.py`, etc. at the repo root).

This is the **single source** of GAN-generated negatives for ADKGD — the
KGSAGE conditional GAN routes through here.

## Why this folder exists

KGSAGE is a self-contained generation library. ADKGD's training pipeline
needs to *call* it at every training batch to obtain plausible-but-wrong
triples, but the library doesn't know anything about ADKGD's specific
ID convention, batch shape, or logging format.

This bridge is the **only** Python file that knows about both worlds.
ADKGD's `dataset.py` imports the three-function API from here; the KGSAGE
package knows nothing about ADKGD.

```
ADKGD side (dataset.py)                        KGSAGE side (kgsage/)
  Reader._gan_negatives                        kgsage.inference.generate_negatives
      |                                              ^
      | uses ADKGD vocab IDs                         | uses KGSAGE vocab IDs
      v                                              |
  experiments/kgsage_bridge/bridge.py    -----------+
      - load_gan(ckpt_path)               -> kgsage.inference.load_checkpoint
      - generate(triples, ...)            -> translates IDs, calls KGSAGE,
                                             returns ADKGD-typed negatives
      - render_stats(stats)               -> kgsage.inference.render_stats
```

## Current status

**ACTIVE** — used by every ADKGD run with `--neg_source gan`.

The bridge wraps the KGSAGE conditional GAN at `kgsage.gan`. For each positive
`(h, r, t)` it returns one single-slot-corruption negative. `dataset.py` only
ever calls the three functions below, so the model can evolve without touching it.

## Contract

```python
def load_gan(ckpt_path, device=None):
    """Load a KGSAGE GAN checkpoint and return a payload dict.

    Returns:
      dict with keys:
        generator        - the trained KGSAGEGenerator (torch.nn.Module)
        device           - torch.device the model is on
        ent2id, rel2id   - GAN's string -> int vocab maps
        id2ent, id2rel   - inverse maps
        real_triple_set  - set of (h, r, t) tuples (for collision filtering)
        n_ent, n_rel     - vocabulary sizes
        z_dim            - noise dimension
    """

def generate(adkgd_triples, *,
             payload,
             adkgd_id2ent, adkgd_id2rel,
             adkgd_ent2id, adkgd_rel2id,
             rng=None):
    """Generate one single-slot-corruption negative per positive (h, r, t),
    in ADKGD's vocabulary.

    Returns:
      (negatives, stats) where:
        negatives is list of (h, r, t) tuples in ADKGD's IDs
        stats is a dict with keys 'processed', 'retries', 'uniform_fallbacks',
                                  'slot_h', 'slot_r', 'slot_t'
    """

def render_stats(stats):
    """Format a stats dict into a single human-readable log line.

    Returns:
      A string like 'processed=100,000 retries=50,000 uniform_fallbacks=0 ...'
    """
```

## File layout

```
experiments/kgsage_bridge/
├── README.md      <- this file
├── __init__.py    <- re-exports the three bridge functions
└── bridge.py      <- the actual implementation (wraps kgsage.inference.*)
```

## See also

- `experiments/kgsage/` — the standalone KGSAGE package
- `experiments/kgsage/gan/` — the KGSAGEGenerator + KGSAGEDiscriminator
- `experiments/kgsage/inference.py` — `load_checkpoint`, `generate_negatives`, `render_stats`
- `dataset.py` (repo root) — `Reader._gan_negatives` is the caller
