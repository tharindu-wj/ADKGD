# KGSAGE experiments — operator guide

Everything under `experiments/` implements the KGSAGE pipeline: train a learned,
neighbourhood-aware generator of hard KG **corruptions**, and evaluate whether
training the ADKGD detector on them beats rule-based random corruption.

## Terminology

One name per concept, for all of `experiments/`: **[docs/KGSAGE_glossary.md](docs/KGSAGE_glossary.md)**
— the spec this codebase's names follow. Read it before naming anything new.
(Heads-up: `docs/` is gitignored, so the glossary is a working-tree file; only a
few plans of record in there are force-added to git.) The two seams worth
knowing up front:

- A generated false triple is a **corruption** inside `kgsage/`; the detector
  calls the same object a **negative** (training) or an **anomaly**
  (evaluation). That is ADKGD's vocabulary, not drift.
- The pipeline has three phases — **Phase 1** Neighbourhood Context Encoding,
  **Phase 2** Adversarial Generator Training, **Phase 3** Corruption
  Generation.

## The experiment matrix

One run = one cell of `(--neg_source × --test_anomaly_source)`, two sources.
(`--neg_source` / `--test_anomaly_source` and their `random`|`gan` values are a
frozen CLI contract shared by `run_experiment.py`, `exp_cell.slurm` and the
detector — do not rename them.)

| source | what it is |
|---|---|
| `random`  | ADKGD's original corruption (baseline; bit-identical to the paper) |
| `gan`     | a trained KGSAGE checkpoint (`kgsage.gan.train`: dual-discriminator generator, arch string `candidate_v2`; use the locked `generator_<dataset>.pt` artifacts) |

The 2×2 matrix (rule-based = `random`, KGSAGE = `gan`): `random×random`
(baseline) · `random×gan` (**the gap** — are KGSAGE anomalies harder?) ·
`gan×gan` (**proposed** — recovery by training on them) · `gan×random`
(**cross-check** — can it still catch the easy ones?).

## Quickstart (local, CPU, FB15K-mini smoke set)

```bash
# one matrix cell end-to-end (~2 min)
python experiments/run_experiment.py --dataset FB15K-mini \
    --neg_source gan --test_anomaly_source gan \
    --gan_path artifacts/kgsage/generator_fb15k237.pt

# aggregate every cell run so far (mean±std over seeds)
python experiments/aggregate_results.py --dataset FB15K-mini
```

Windows/CPU note: set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
KMP_DUPLICATE_LIB_OK=TRUE`; the entry point additionally auto-disables the
torch-2.8 oneDNN LSTM kernel on Windows CPU (`ADKGD_DISABLE_MKLDNN` to
override) — both are no-ops on the GPU cluster.

## HPC workflow (DeepThought — details in RUNNING_ON_DEEPTHOUGHT.md)

1. Train the generator and select a snapshot **in the KGSAGE repo** — it is a
   separate package and nothing here trains one. See its `README.md` and
   `slurm/README.md`.
2. Copy the chosen checkpoint into `artifacts/kgsage/` (see that folder's README).
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
type_valid=… resampled=… slot_distribution: …` plus a capped
`(positive -> negative)` pair preview (`GAN_PAIR_PREVIEW` to raise).
`used_original` is a frozen stats key — it counts the rows where every resample
failed, so the original triple was emitted unchanged (a *null* corruption: the
true_filler stayed in its slot). Those rows are listed in `null_indices` (also
frozen) and replaced in the training role — a null is a real fact, never train
on it.

## Layout

| path | role |
|---|---|
| `run_experiment.py` | one matrix cell (train + test subprocess, metrics, run JSON) |
| `aggregate_results.py` | mean±std tables per cell over seeds |
| `kgsage/` | the standalone generator package (ADKGD-agnostic) — see its README |
| `kgsage/gan/` | the whole adversarial training stack (encoder, sketches, candidate sampler, both discriminators), not only the GAN — the folder name is kept because `python -m kgsage.gan.train` is the documented entry point |
| `kgsage_bridge/` | the ONLY ADKGD-aware glue (three-function API) — see its README |
| `slurm/exp_cell.slurm` | the single generic cell launcher |
| `docs/` | the naming glossary + plans of record + research library (gitignored; a few plans are force-added) |

Smoke test: `PYTHONPATH=experiments python experiments/kgsage_bridge/smoke_test.py`
(bridge round-trip; needs `pip install -e <kgsage>` and a checkpoint in
`artifacts/kgsage/`). The KGSAGE package has its own smoke test in its repo.
`data/FB15K-mini` (first 2000/300/300 lines of FB15K-237) is the local
ADKGD-cycle smoke dataset.
