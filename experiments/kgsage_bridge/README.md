# KGSAGE ↔ ADKGD bridge

This folder is the **integration layer** between the standalone KGSAGE package
(in `experiments/kgsage/`) and ADKGD's anomaly detection training pipeline
(`dataset.py`, `model.py`, etc. at the repo root).

## Why this folder exists

KGSAGE is designed to ship as a standalone package — `pip install kgsage`
someday. A standalone package cannot import from ADKGD or know about ADKGD's
internal types. But ADKGD's training pipeline needs to *call* KGSAGE to get
contradiction negatives during training.

The bridge resolves this: it is the **only** Python file that knows about
both worlds. ADKGD's `dataset.py` imports the three-function API from here;
KGSAGE knows nothing about ADKGD.

## Architecture (in one diagram)

```
ADKGD side (dataset.py)                       KGSAGE side (kgsage/)
  Reader._gan_negatives                       KGSAGE.generate_contradictions
      |                                            ^
      | uses ADKGD vocab IDs                       | uses KGSAGE vocab IDs
      v                                            |
  experiments/kgsage_bridge/bridge.py  ────────────┘
      - load_gan(ckpt_path)        → loads KGSAGE checkpoint
      - generate(model, batch)     → translates IDs, calls KGSAGE, returns
                                     ADKGD-typed negatives
      - render_stats(stats)        → formats KGSAGE stats for ADKGD's log
```

## Current status

**STUB.** The three functions are placeholders that raise
`NotImplementedError` with a pointer to the thesis plan. They will be
implemented in **Phase 4** of the thesis pipeline, once:

- Phase 1 (KGSAGE encoder) is trained and validated ✅ (in progress)
- Phase 2 (KGSAGE Generator + Discriminator) is implemented
- Phase 3 (generation quality evaluation) is complete

Until then, ADKGD continues to use:
- `--neg_source random` (paper baseline) — works
- `--neg_source gan` → `experiments/gan/adkgd_bridge.py` (simple GAN) — works
- `--neg_source kgsage` → raises NotImplementedError pointing here

## Contract (when implemented)

The bridge implements the **same three-function contract** as
`experiments/gan/adkgd_bridge.py` for the simple GAN, so swapping between
the two requires only the `--neg_source` flag:

```python
def load_gan(ckpt_path, device=None, **kwargs):
    """Load a KGSAGE checkpoint and return (model, ent2id, rel2id)."""

def generate(model, batch, adkgd_ent2id, adkgd_rel2id, n_per_anchor=1):
    """Generate role-swap partner triples for a batch of anchors.
    Returns a tensor of shape (B*n_per_anchor, 3) in ADKGD's vocab."""

def render_stats(stats):
    """Format a stats dict from generate() for ADKGD's training log."""
```

## File layout

```
experiments/kgsage_bridge/
├── README.md      ← this file
├── __init__.py    ← re-exports the three bridge functions
└── bridge.py      ← the implementation (currently stub)
```

## Future SLURM script

`experiments/slurm/run_adkgd_with_kgsage_fb15k237.slurm` will invoke ADKGD's
training with `--neg_source kgsage` and the appropriate KGSAGE checkpoint
path. It lives in `experiments/slurm/` (with the ADKGD-side launchers), not
in `kgsage/slurm/`, because it's running ADKGD — not KGSAGE — even though
KGSAGE supplies the negatives.

## See also

- `experiments/docs/THESIS_PLAN_pairgan_contradictions.md` — full thesis plan
- `experiments/gan/adkgd_bridge.py` — the same contract for the simple GAN
- `experiments/kgsage/` — the standalone KGSAGE package
