# KGSAGE checkpoint drop

Trained generator checkpoints land here. **Nothing in this repo produces them** —
KGSAGE is a separate package with its own repository, and this directory is the
hand-off point between the two.

The `.pt` files are gitignored; only this README is tracked.

## Producing a checkpoint (KGSAGE repo)

```bash
cd /path/to/kgsage
python -m kgsage.gan.train \
    --data data/WN18RR \
    --out  outputs/checkpoints/run_wn18rr_s0.pt \
    --epochs 8 --seed 0 --device cuda
```

Anchor-specificity peaks a few epochs after the alpha ramp and then erodes, so
train with `--snapshot_every 1` and pick the snapshot with the lowest mean
knockout J@10 rather than taking the final epoch:

```bash
python -m kgsage.cli.knockout_eval --ckpt outputs/checkpoints/run_wn18rr_s0.epNN.pt \
                                   --data data/WN18RR
```

## Consuming it (this repo)

Copy the chosen checkpoint in, then point the detector at it:

```bash
cp /path/to/kgsage/outputs/checkpoints/run_wn18rr_s0.ep06.pt artifacts/kgsage/

python "Our_TopK%_RankingList.py" \
    --dataset WN18RR \
    --neg_source gan --test_anomaly_source gan \
    --gan_path artifacts/kgsage/run_wn18rr_s0.ep06.pt
```

`KGSAGE_CKPT` overrides `--gan_path`, which is how the SLURM launchers point at
a checkpoint on scratch without editing the job script.

Running the detector's own baseline needs none of this, and does not need KGSAGE
installed at all:

```bash
python "Our_TopK%_RankingList.py" --dataset WN18RR --neg_source random
```

## Installing KGSAGE

Required only for `--neg_source gan` or `--test_anomaly_source gan`:

```bash
pip install -e /path/to/kgsage
```
