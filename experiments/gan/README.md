# Simple GAN for KG triple corruption

A teaching-oriented implementation of a conditional GAN that learns to
produce "plausible-but-wrong" knowledge-graph triples. ADKGD uses these
as training-time negatives.

This version is intentionally simple: plain MLPs, no conv encoders, no
attention, no spectral norm, no exotic losses. Goal is **readability**, not
state-of-the-art performance.

## Files

| File | Purpose | Length |
|---|---|---|
| [data.py](data.py) | Load `train/valid/test.txt` → integer triples + vocab | ~70 lines |
| [model.py](model.py) | `Generator` + `Discriminator` + Gumbel-Softmax helper | ~110 lines |
| [train.py](train.py) | Training loop + CLI (`python train.py --data ...`) | ~190 lines |
| [generate.py](generate.py) | In-process negative generation (loaded by ADKGD at training time) | ~190 lines |
| [adkgd_bridge.py](adkgd_bridge.py) | Thin re-export so ADKGD's `dataset.py` has a stable import | ~35 lines |
| `outputs/checkpoints/` | Trained `.pt` files (gitignored) | — |

## The idea (one paragraph)

A real triple looks like `(Alice, born_in, Australia)`. We want the generator
to produce something like `(Carol, born_in, Australia)` — same shape, but
not actually in the graph. We feed the real triple plus random noise into
the generator; it outputs three probability distributions (over entities for
the head, over relations for the relation, over entities for the tail). The
discriminator looks at `(real, candidate)` pairs and tries to spot which
candidates are made-up. Through alternating training, the generator gets
better at fooling the discriminator, which means better at producing
plausible-looking fake triples.

## Architecture

**Generator**

```
  (h, r, t)        ─→  embed     ─→ ┐
                                    ├─→ MLP ─→ hidden ─→ head_out ─→ logits over entities
  noise z                          ─┘                 ├─→ rel_out  ─→ logits over relations
                                                       └─→ tail_out ─→ logits over entities
```

**Discriminator**

```
  real (h, r, t)         ─→ embed ─→ ┐
                                     ├─→ concat ─→ MLP ─→ score (high = looks real)
  candidate (h', r', t') ─→ embed ─→ ┘
```

That's it. Both are 3-layer MLPs.

## Training losses

```
  L_D       = BCE(D(real, target),    label=1)
            + BCE(D(real, generated), label=0)

  L_G_adv   = BCE(D(real, generated), label=1)        # G wants D to think it's real
  L_G_recon = CE(G_logits, target)                    # G should predict the target distribution
  L_G       = L_G_adv + 10 * L_G_recon
```

The reconstruction loss is the same idea as Pix2Pix: a supervised signal
that anchors the generator's output to the actual target distribution. The
adversarial loss adds the "make it look real" pressure on top.

## Train one

Local (dummy KG, CPU, ~minutes):

```bash
python experiments/gan/train.py \
    --data data/dummy_kg \
    --epochs 30 \
    --device cpu \
    --out experiments/gan/outputs/checkpoints/dummy.pt
```

HPC (FB15K-237 on V100):

```bash
sbatch experiments/slurm/train_gan_fb15k.slurm
```

Notable hyperparameters (with sane defaults in `train.py`):

| Flag | Default | What it does |
|---|---|---|
| `--epochs` | 50 | Full passes over the data |
| `--batch_size` | 64 | Pairs per step |
| `--lr` | 1e-4 | Generator learning rate (D uses 0.25× of this) |
| `--dim` | 64 | Embedding dimension |
| `--z_dim` | 16 | Noise vector dimension |
| `--recon_weight` | 10.0 | How strongly to anchor outputs to the target |

## Use it from ADKGD

After training, just point ADKGD at the checkpoint:

```bash
python experiments/run_experiment.py \
    --dataset dummy_kg --anomaly_ratio 0.15 --max_epoch 1 \
    --neg_source gan \
    --gan_path experiments/gan/outputs/checkpoints/dummy.pt
```

ADKGD's `Reader._gan_negatives` calls `adkgd_bridge.generate(...)`, which
calls `generate.generate_negatives(...)`, which runs the loaded Generator
in `torch.no_grad()` mode per training batch.

## What's deliberately omitted (and why)

The original kggan we replaced had:

- 1×1 `Conv2d` encoder + multi-head self-attention in the generator
- Spectral norm on the discriminator
- TRIC (Type-aware Random Item Corruption) for training-pair generation
- Mode-seeking diversity loss + Gumbel temperature anneal + instance noise schedule

All of those make the GAN train better, but each one adds machinery a reader
has to absorb before getting to the actual GAN concept. This version cuts
them so the file you read maps directly to the diagram above.

If you want to **add** any of those back, the simplest place to start is
the discriminator (drop `nn.utils.spectral_norm` around each `Linear`).
After that, the conv encoder + attention upgrade is mostly self-contained
in `model.py`'s `Generator.__init__`.

## Reference papers

- Conditional GAN: Mirza & Osindero (2014). [arXiv:1411.1784](https://arxiv.org/abs/1411.1784)
- Gumbel-Softmax: Jang, Gu, Poole (2017). [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)
- Pix2Pix (the recon-loss recipe): Isola et al. (2017). [arXiv:1611.07004](https://arxiv.org/abs/1611.07004)

See [`experiments/docs/`](../docs/) for the full architectural background of
the original kggan (the version this simplifies from) and the ADKGD paper.
