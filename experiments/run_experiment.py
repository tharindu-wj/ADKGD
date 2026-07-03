"""Run ONE ADKGD baseline experiment (one train + one test) and print the
five Precision@K / Recall@K values plus the total training time.

Defaults reproduce the FB15K-237 column of the paper's Table 2 (5% anomaly,
seed 0, 1 epoch). Designed to be invoked from a slurm script, but runs
identically from a normal shell.

Local CPU caveat: on Windows/CPU, set OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
KMP_DUPLICATE_LIB_OK=TRUE in the environment before running -- otherwise
PyTorch can segfault after a few batches. Not needed on the GPU cluster.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# Defensive defaults: PyTorch CPU on Windows segfaults under multi-threaded
# OpenMP/MKL. setdefault means the cluster's slurm-controlled values
# ($SLURM_CPUS_PER_TASK on a GPU node) still take precedence when set.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# Log lines look like:
#   INFO:root:[Test][FB15K-237][ADKGD] Precision 0.050000 -- 0.010000 : 0.951...
#   INFO:root:[Test][FB15K-237][ADKGD] Recall   0.050000-- 0.010000 : 0.190...
#   Epoch: 0, Duration: 700.32 seconds
PRECISION_RE = re.compile(r"Precision\s+(\d+\.\d+)\s*--\s*(\d+\.\d+)\s*:\s*(\d+\.\d+)")
RECALL_RE = re.compile(r"Recall\s+(\d+\.\d+)\s*--\s*(\d+\.\d+)\s*:\s*(\d+\.\d+)")
DURATION_RE = re.compile(r"Duration:\s+([\d.]+)")
AUC_RE = re.compile(r"AUC\s+([\d.]+)\s*--\s*AUPRC\s+([\d.]+)")


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess, stream its output, exit on non-zero return code."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    # Pass our env explicitly so the OMP/MKL/KMP defaults above reach the child
    # even if shell-level export didn't (Windows quirk). `cwd` lets the caller
    # pin the working directory (ADKGD uses bare relative paths like ./data/...).
    completed = subprocess.run(cmd, check=False, env=os.environ.copy(), cwd=cwd)
    if completed.returncode != 0:
        print(f"!! subprocess exited with code {completed.returncode}", file=sys.stderr)
        sys.exit(completed.returncode)


def _approx(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) < tol


def parse_metrics(log_path: Path, anomaly_ratio: float, ks: list[float]) -> dict[float, tuple[float | None, float | None]]:
    """Return {k: (precision, recall)} for the requested K cutoffs."""
    if not log_path.exists():
        print(f"!! log not found: {log_path}", file=sys.stderr)
        return {k: (None, None) for k in ks}

    text = log_path.read_text(encoding="utf-8", errors="replace")
    precisions: dict[float, float] = {}
    for m in PRECISION_RE.finditer(text):
        ar, k, v = float(m.group(1)), float(m.group(2)), float(m.group(3))
        if _approx(ar, anomaly_ratio):
            precisions[round(k, 6)] = v
    recalls: dict[float, float] = {}
    for m in RECALL_RE.finditer(text):
        ar, k, v = float(m.group(1)), float(m.group(2)), float(m.group(3))
        if _approx(ar, anomaly_ratio):
            recalls[round(k, 6)] = v
    return {k: (precisions.get(round(k, 6)), recalls.get(round(k, 6))) for k in ks}


def parse_total_test_min(test_time_path: Path) -> float | None:
    """Return total test minutes parsed from the test_time.txt file.

    Same format as the train file (`Duration: X seconds`), so we reuse
    DURATION_RE. Test writes exactly one entry per run, but we sum defensively
    in case that ever changes.
    """
    if not test_time_path.exists():
        print(f"!! test_time not found: {test_time_path}", file=sys.stderr)
        return None
    durations = [float(m) for m in DURATION_RE.findall(test_time_path.read_text(encoding="utf-8", errors="replace"))]
    if not durations:
        return None
    return sum(durations) / 60.0


def parse_total_train_min(epoch_times_path: Path) -> tuple[float, int] | None:
    """Return (total_minutes, n_epochs) parsed from the epoch_times.txt file."""
    if not epoch_times_path.exists():
        print(f"!! epoch_times not found: {epoch_times_path}", file=sys.stderr)
        return None
    durations = [float(m) for m in DURATION_RE.findall(epoch_times_path.read_text(encoding="utf-8", errors="replace"))]
    if not durations:
        return None
    return sum(durations) / 60.0, len(durations)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="FB15K-237", help="dataset folder name under data/")
    ap.add_argument("--anomaly_ratio", type=float, default=0.05, help="fraction of fakes injected (e.g. 0.05)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_epoch", type=int, default=1)
    ap.add_argument("--model", default="ADKGD", help="label written into output filenames")
    ap.add_argument("--script", default="Our_TopK%_RankingList.py", help="ADKGD entry-point script")
    # Phase B (GAN integration). Forwarded verbatim to both train and test subprocesses.
    ap.add_argument("--neg_source", default="random", choices=["random", "gan", "lp_band"],
                    help="source of TRAINING negatives; 'random' = baseline (default)")
    ap.add_argument("--test_anomaly_source", default="random", choices=["random", "gan", "lp_band"],
                    help="source of the INJECTED eval anomalies; 'random' = baseline (default). "
                         "The (neg_source x test_anomaly_source) pair is the experiment matrix.")
    ap.add_argument("--gan_path", default="experiments/kgsage/outputs/checkpoints/kgsage_fb15k237.pt",
                    help="path to the KGSAGE GAN .pt checkpoint (used when EITHER axis is 'gan'; missing file is a hard error)")
    ap.add_argument("--lp_path", default=None,
                    help="LibKGE ComplEx checkpoint for lp_band; when omitted, derived from --dataset "
                         "(FB15K* -> fb15k-237-complex.pt, WN18RR -> wnrr-complex.pt under experiments/kgsage/outputs/lp/)")
    ap.add_argument("--lp_ids_dir", default=None,
                    help="LibKGE archive dir for lp_band id maps; derived from --dataset when omitted")
    args = ap.parse_args()

    # B0 hygiene: encode the experiment cell in the model label so checkpoint
    # and log filenames (all derived from --model) can never collide between
    # matrix cells or seeds running concurrently. A user-supplied --model is
    # respected verbatim; the default gets the cell identity appended.
    if args.model == "ADKGD":
        args.model = f"ADKGD_{args.neg_source}x{args.test_anomaly_source}_s{args.seed}"
    print(f"[run_experiment] model label: {args.model}")

    # This file lives at experiments/run_experiment.py; the repo root (where
    # ADKGD's data/, Our_TopK%_RankingList.py, etc. live) is one level up.
    project_root = Path(__file__).resolve().parent.parent
    ckpt_dir = project_root / "checkpoints" / args.dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ADKGD's logger writes to <model>_<dataset>_<ratio>_Neighbors<N>__log.txt
    # where N = --num_neighbor (default 39). The "39" comes from ADKGD's batch
    # construction: every triple is expanded into a local subgraph of
    # [self, 39 head-neighbors, self, 39 tail-neighbors] = 80 triples that feeds
    # the dual-channel encoder. We never vary N, so the literal "39" is fixed.
    # See Our_TopK%_RankingList.py:64 (--num_neighbor default) and
    #     Our_TopK%_RankingList.py:112 (where the filename is built).
    log = ckpt_dir / f"{args.model}_{args.dataset}_{args.anomaly_ratio}_Neighbors39__log.txt"
    ept = ckpt_dir / f"{args.model}_{args.dataset}_epoch_times.txt"
    ttf = ckpt_dir / f"{args.model}_{args.dataset}_test_time.txt"

    # ADKGD only ever appends to these. Start fresh so this run's report is clean.
    log.unlink(missing_ok=True)
    ept.unlink(missing_ok=True)
    ttf.unlink(missing_ok=True)

    py = sys.executable                            # use the same interpreter we were launched with
    adkgd_script = project_root / args.script      # absolute path to ADKGD's entry script

    # Phase B flags get appended to BOTH the train and test invocations so the
    # Reader sees the same neg_source in either mode (Reader is rebuilt fresh
    # in each subprocess). --gan_path is only forwarded when we actually need
    # it -- otherwise it's misleading noise in the B0 log (and could mask a
    # real misconfiguration if the path is stale).
    gan_args = ["--neg_source", args.neg_source,
                "--test_anomaly_source", args.test_anomaly_source]
    if args.neg_source == "gan" or args.test_anomaly_source == "gan":
        gan_args += ["--gan_path", args.gan_path]
    if args.neg_source == "lp_band" or args.test_anomaly_source == "lp_band":
        # Derive the LP checkpoint + id-map dir from the dataset when not given.
        # FB15K-mini reuses the full FB15K-237 scorer (its vocab is a subset).
        lp_root = project_root / "experiments" / "kgsage" / "outputs" / "lp"
        name = args.dataset.upper()
        if args.lp_path is None or args.lp_ids_dir is None:
            if name.startswith("FB15K"):
                stem = "fb15k-237"
            elif name.startswith("WN18RR"):
                stem = "wnrr"
            else:
                print(f"!! cannot derive --lp_path for dataset {args.dataset}; "
                      "pass --lp_path/--lp_ids_dir explicitly", file=sys.stderr)
                return 2
            args.lp_path = args.lp_path or str(lp_root / f"{stem}-complex.pt")
            args.lp_ids_dir = args.lp_ids_dir or str(lp_root / stem)
        gan_args += ["--lp_path", args.lp_path, "--lp_ids_dir", args.lp_ids_dir]

    # Train -- cwd=project_root so ADKGD's "./data/..." / "./checkpoints/..." resolve correctly.
    _run([
        py, str(adkgd_script),
        "--dataset", args.dataset,
        "--model", args.model,
        "--mode", "train",
        "--anomaly_ratio", str(args.anomaly_ratio),
        "--seed", str(args.seed),
        "--max_epoch", str(args.max_epoch),
        *gan_args,
    ], cwd=project_root)

    # Test (same cwd reasoning).
    _run([
        py, str(adkgd_script),
        "--dataset", args.dataset,
        "--model", args.model,
        "--mode", "test",
        "--anomaly_ratio", str(args.anomaly_ratio),
        "--seed", str(args.seed),
        *gan_args,
    ], cwd=project_root)

    ratio = args.anomaly_ratio
    ks = [round(ratio * i / 5, 6) for i in range(1, 6)]  # paper convention: K = ratio/5, 2*ratio/5, ..., ratio

    # get the metrics and timing info from the log files, print them in a nice format
    metrics = parse_metrics(log, ratio, ks)
    auc_match = AUC_RE.search(log.read_text(encoding="utf-8", errors="replace")) if log.exists() else None
    auc, auprc = (float(auc_match.group(1)), float(auc_match.group(2))) if auc_match else (None, None)

    # The epoch_times file is only written during training, and the test_time
    # file is only written during testing. Either may be missing if its phase
    # crashed; report what we have instead of erroring out.
    timing = parse_total_train_min(ept)
    test_timing = parse_total_test_min(ttf)

    print()
    print("=" * 60)
    print(f"  RESULTS  ({args.dataset} @ anomaly_ratio={ratio}, seed={args.seed})")
    print("=" * 60)
    print()
    print(f"{'K':>6}  {'Precision@K':>12}  {'Recall@K':>10}")
    print(f"{'-' * 6}  {'-' * 12}  {'-' * 10}")
    for k in ks:
        p, r = metrics[k]
        p_str = f"{p:.4f}" if p is not None else "  --  "
        r_str = f"{r:.4f}" if r is not None else "  --  "
        print(f"{k * 100:>5.0f}%  {p_str:>12}  {r_str:>10}")
    print()
    if auc is not None:
        print(f"AUC:   {auc:.4f}    AUPRC: {auprc:.4f}")
    else:
        print("AUC:   -- (no AUC line found in the log)")
    print()

    # Machine-readable per-run record for aggregate_results.py (mean±std over
    # seeds per matrix cell). One JSON per run, named by the cell-identity label.
    import json
    run_record = {
        "dataset": args.dataset, "model": args.model, "seed": args.seed,
        "neg_source": args.neg_source, "test_anomaly_source": args.test_anomaly_source,
        "anomaly_ratio": ratio, "max_epoch": args.max_epoch,
        "precision_at": {str(k): metrics[k][0] for k in ks},
        "recall_at": {str(k): metrics[k][1] for k in ks},
        "auc": auc, "auprc": auprc,
    }
    run_json = ckpt_dir / f"{args.model}_{args.dataset}_run.json"
    run_json.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(f"per-run record: {run_json}")

    if timing is None:
        print("Total train time: -- (no epoch_times file found)")
    else:
        total_min, n_epochs = timing
        print(f"Total train time: {total_min:.2f} minutes ({n_epochs} epoch(s))")

    if test_timing is None:
        print("Total test time:  -- (no test_time file found)")
    else:
        print(f"Total test time:  {test_timing:.2f} minutes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
