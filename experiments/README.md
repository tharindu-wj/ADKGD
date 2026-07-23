# KGSAGE experiments — operator guide

Everything under `experiments/` implements the KGSAGE pipeline: train a learned,
neighbourhood-aware generator of hard KG anomalies, and evaluate whether
training the ADKGD detector on them beats rule-based random corruption.

## The experiment matrix

One run = one cell of `(--neg_source × --test_anomaly_source)`, two sources:

| source | what it is |
|---|---|
| `random`  | ADKGD's original corruption (baseline; bit-identical to the paper) |
| `gan`     | a trained KGSAGE checkpoint (`kgsage.gan.train`: dual-discriminator `candidate_v2` generator; use the locked `generator_<dataset>.pt` artifacts) |

The 2×2 matrix (rule-based = `random`, KGSAGE = `gan`): `random×random`
(baseline) · `random×gan` (**the gap** — are KGSAGE anomalies harder?) ·
`gan×gan` (**proposed** — recovery by training on them) · `gan×random`
(**cross-check** — can it still catch the easy ones?).

## Quickstart (local, CPU, FB15K-mini smoke set)

```bash
# one matrix cell end-to-end (~2 min)
python experiments/run_experiment.py --dataset FB15K-mini \
    --neg_source gan --test_anomaly_source gan \
    --gan_path experiments/kgsage/outputs/checkpoints/generator_fb15k237.pt

# aggregate every cell run so far (mean±std over seeds)
python experiments/aggregate_results.py --dataset FB15K-mini
```

Windows/CPU note: set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
KMP_DUPLICATE_LIB_OK=TRUE`; the entry point additionally auto-disables the
torch-2.8 oneDNN LSTM kernel on Windows CPU (`ADKGD_DISABLE_MKLDNN` to
override) — both are no-ops on the GPU cluster.

## HPC workflow (DeepThought — details in RUNNING_ON_DEEPTHOUGHT.md)

1. Train the generator: `DATASET=fb15k237 SEED=0 sbatch experiments/kgsage/slurm/train.slurm` (per-epoch snapshots).
2. Select the snapshot: `python experiments/kgsage/cli/knockout_eval.py --ckpt <each .epNN.pt> --data ...` (lowest mean knockout J@10 wins); promote to `generator_<dataset>.pt`.
3. Run cells: `NEG_SOURCE=... TEST_SOURCE=... SEED=... DATASET=... [GAN_CKPT=...] sbatch experiments/slurm/exp_cell.slurm`.
4. `python experiments/aggregate_results.py --dataset <ds>`.

## What a run produces

- a RESULTS table (P@K / R@K at 5 cutoffs) + `AUC: … AUPRC: …` printed by
  `run_experiment.py`;
- artifacts under `checkpoints/<dataset>/`, all named by the **cell-identity
  label** `ADKGD_<neg>x<test>_s<seed>`: the model `.ckpt`, the metrics log
  (`…_Neighbors39__log.txt`), a raw score dump (`…_scores.npz`) and a
  machine-readable `…_run.json` for the aggregator.

Generation diagnostics printed per run: `[GAN] processed=… used_original=…
type_valid=… resampled=… slot_distribution: …` (nulls are flagged and replaced
in the training role) plus a capped `(positive -> negative)` pair preview
(`GAN_PAIR_PREVIEW` to raise).

## Layout

| path | role |
|---|---|
| `run_experiment.py` | one matrix cell (train + test subprocess, metrics, run JSON) |
| `aggregate_results.py` | mean±std tables per cell over seeds |
| `kgsage/` | the standalone generator package (ADKGD-agnostic) — see its README |
| `kgsage_bridge/` | the ONLY ADKGD-aware glue (three-function API) — see its README |
| `slurm/exp_cell.slurm` | the single generic cell launcher |
| `docs/` | plans of record + research library (gitignored) |

Smoke tests: `python experiments/kgsage/smoke_test.py` (package + bridge);
`data/FB15K-mini` (first 2000/300/300 lines of FB15K-237) is the local
ADKGD-cycle smoke dataset.
