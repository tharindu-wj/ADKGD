# KGSAGE experiments — operator guide

Everything under `experiments/` implements the staged PoC pipeline
(**Option B → Option A-ii**; plans of record in `docs/OPTION_B_PLAN.md`,
`docs/OPTION_A_PLAN.md`, destination in `docs/KGSAGE_IMPLEMENTATION_PLAN.md`):
generate **close-but-false** KG anomalies and evaluate whether training the
ADKGD detector on them beats random corruption.

## The experiment matrix

One run = one cell of `(--neg_source × --test_anomaly_source)`, sources:

| source | what it is |
|---|---|
| `random`  | ADKGD's original corruption (baseline; bit-identical to the paper) |
| `lp_band` | Option B: a frozen pretrained ComplEx ranks the relation's type pool; sample the top-k band **below** s(true), masked against every known-true filler (all splits) — no GAN; doubles as the `sampler_direct` control arm |
| `gan`     | a trained KGSAGE checkpoint — primary arm = **A-ii** (`kgsage.gan.train_aii`: frozen ComplEx backbone + trainable contextual residual D); legacy arm = B1a (`kgsage.cli.train_gan`, ablation only) |

The PoC read-out (pre-registered in `docs/OPTION_B_PLAN.md` §0):
`random×random` (baseline) vs `random×lp_band` (**the gap**) vs
`lp_band×lp_band` (**the recovery**) vs `lp_band×random` (**no regression**)
vs `gan×lp_band` (**the A-ii receipt** — must beat the band-sampler control).

## Quickstart (local, CPU, FB15K-mini smoke set)

```bash
# once: fetch the frozen LP checkpoints + run the MRR loader gates
PYTHONPATH=experiments python -m kgsage.cli.fetch_lp --dataset all
#   expect: GATE PASS fb15k237 mrr≈0.3477 · wn18rr mrr≈0.4749

# one matrix cell end-to-end (~2 min)
python experiments/run_experiment.py --dataset FB15K-mini \
    --neg_source lp_band --test_anomaly_source lp_band

# aggregate every cell run so far (mean±std over seeds)
python experiments/aggregate_results.py --dataset FB15K-mini
```

Windows/CPU note: set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
KMP_DUPLICATE_LIB_OK=TRUE`; the entry point additionally auto-disables the
torch-2.8 oneDNN LSTM kernel on Windows CPU (`ADKGD_DISABLE_MKLDNN` to
override) — both are no-ops on the GPU cluster.

## HPC workflow (DeepThought — details in RUNNING_ON_DEEPTHOUGHT.md)

1. `PYTHONPATH=experiments python -m kgsage.cli.fetch_lp --dataset all` (login node, once).
2. Train A-ii generators: `DATASET=fb15k237 SEED=0 sbatch experiments/kgsage/slurm/train_aii.slurm` (× datasets × seeds).
3. Inspect each checkpoint: `python -m kgsage.cli.inspect_gan_lp --ckpt ... --n 40`.
4. Run cells: `NEG_SOURCE=... TEST_SOURCE=... SEED=... DATASET=... [GAN_CKPT=...] sbatch experiments/slurm/exp_cell.slurm`
   (exp1–4.slurm remain as the named random/gan cells; `exp_cell.slurm` covers everything incl. `lp_band`).
5. `python experiments/aggregate_results.py --dataset <ds>`.

## What a run produces

- a RESULTS table (P@K / R@K at 5 cutoffs) + `AUC: … AUPRC: …` printed by
  `run_experiment.py`;
- artifacts under `checkpoints/<dataset>/`, all named by the **cell-identity
  label** `ADKGD_<neg>x<test>_s<seed>`: the model `.ckpt`, the metrics log
  (`…_Neighbors39__log.txt`), a raw score dump (`…_scores.npz`) and a
  machine-readable `…_run.json` for the aggregator.

Generation diagnostics printed per run: `[GAN] processed=… used_original=…
type_valid=… resampled=… slot_distribution: …` (nulls are impossible for
`lp_band`; for `gan` they are flagged and replaced in the training role) and
a capped `(positive -> negative)` pair preview (`GAN_PAIR_PREVIEW` to raise).

## Layout

| path | role |
|---|---|
| `run_experiment.py` | one matrix cell (train + test subprocess, metrics, run JSON) |
| `aggregate_results.py` | mean±std tables per cell over seeds |
| `kgsage/` | the standalone generator package (ADKGD-agnostic) — see its README |
| `kgsage_bridge/` | the ONLY ADKGD-aware glue (six-function API) — see its README |
| `slurm/exp_cell.slurm` | generic cell launcher; `exp1–4.slurm` = named random/gan cells |
| `docs/` | plans of record + research library (gitignored except the three plan files) |

Smoke tests: `python experiments/kgsage/smoke_test.py` (package + bridge);
`data/FB15K-mini` (first 2000/300/300 lines of FB15K-237) is the local
ADKGD-cycle smoke dataset — full `dummy_kg` ADKGD training crashes natively
on Windows for reasons unrelated to this code.
