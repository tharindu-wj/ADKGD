"""Bridge between ADKGD's Reader and the KGSAGE corruption library.

ADKGD's dataset.py imports this module by name (`from adkgd_bridge import ...`)
after adding experiments/gan/ to sys.path. We expose three symbols matching
the old Gumbel-Softmax contract so ADKGD's call site doesn't change:

  - load_gan(ckpt_path)              -> payload dict (cached KGCorrupter)
  - generate(triples, payload, ...)  -> (list of ADKGD-ID negatives, stats)
  - render_stats(stats)              -> short human-readable summary line

Why this file exists at experiments/gan/ root instead of inside corruption/:
ADKGD's dataset.py already adds experiments/gan/ to sys.path and imports
"adkgd_bridge". Keeping the file at this path means dataset.py is
untouched - this file is the ONE seam between ADKGD and our codebase.
The actual corruption logic lives in experiments/gan/corruption/api.py;
this bridge is just a thin string<->int translator.

The 3-step generate() flow per ADKGD-side batch:

  STEP 1: Translate ADKGD integer IDs -> strings using ADKGD's id-maps.
          (ADKGD and KGSAGE can number the same entity differently; strings
           are the lingua franca that keeps both worlds aligned.)

  STEP 2: Call KGCorrupter.corrupt_batch() to produce string-typed
          negatives. KGSAGE handles slot choice, candidate pooling,
          concept filtering, REINFORCE-trained scoring, and validation.

  STEP 3: Translate the negatives' strings back to ADKGD integer IDs
          using ADKGD's ent2id / rel2id maps.

Stats are computed as the per-batch DELTA on KGCorrupter's cumulative
counters so each ADKGD batch sees only its own numbers.
"""
import os
import sys

# Put the corruption package on sys.path so `from corruption.api import ...`
# works regardless of who imported us.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Also put the repo root on sys.path so the `experiments.*` package imports
# inside KGCorrupter resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.gan.corruption.api import KGCorrupter  # noqa: E402

__all__ = ["load_gan", "generate", "render_stats"]


def load_gan(ckpt_path, concept_pools_path=None, device=None):
    """Load a KGSAGE checkpoint once; return a payload to reuse per batch.

    Args:
      ckpt_path:           path to <DATASET>_kgsage.pt (Phase 2 output).
      concept_pools_path:  path to <DATASET>.pkl (Phase 1 output). If None,
                           derived from ckpt_path by replacing the
                           checkpoints/ folder with concept_pools/ and
                           stripping the "_kgsage.pt" suffix.
      device:              torch device (None -> CUDA if available else CPU).

    Returns:
      A payload dict carrying the loaded KGCorrupter. The dict shape is
      intentionally opaque to ADKGD - we just hand it back unchanged on
      every generate() call.
    """
    if concept_pools_path is None:
        concept_pools_path = _derive_concept_pools_path(ckpt_path)

    corrupter = KGCorrupter(
        checkpoint_path=ckpt_path,
        concept_pools_path=concept_pools_path,
        device=device,
    )
    print(f"[KGSAGE] loaded checkpoint from {ckpt_path}", flush=True)
    print(f"[KGSAGE] loaded concept pools from {concept_pools_path}", flush=True)
    return {
        "corrupter": corrupter,
        "ckpt_path": ckpt_path,
        "concept_pools_path": concept_pools_path,
    }


def generate(adkgd_triples, *,
             payload,
             adkgd_id2ent, adkgd_id2rel,
             adkgd_ent2id, adkgd_rel2id,
             rng=None):
    """Produce one ADKGD-ID negative per input ADKGD-ID positive.

    Args:
      adkgd_triples:    iterable of (h_id, r_id, t_id) tuples in ADKGD's IDs.
      payload:          dict returned by load_gan().
      adkgd_id2ent:     ADKGD's int -> entity-string map.
      adkgd_id2rel:     ADKGD's int -> relation-string map.
      adkgd_ent2id:     reverse map.
      adkgd_rel2id:     reverse map.
      rng:              numpy Generator (we extract a single int seed from it
                        to keep KGSAGE's Python-rng path deterministic and
                        independent of the wider torch/numpy global state).

    Returns:
      (negatives, batch_stats):
        negatives:    list of (h_id, r_id, t_id) in ADKGD's IDs, row-aligned
                      with the input.
        batch_stats:  per-batch counters dict. Keys: processed, retries,
                      uniform_fallbacks, slot_head, slot_tail.
    """
    corrupter = payload["corrupter"]

    # Derive a deterministic int seed for this batch from the numpy rng.
    # If no rng is provided, KGCorrupter falls back to its persistent
    # internal session rng (also deterministic but not call-isolated).
    if rng is not None:
        seed = int(rng.integers(0, 2**31 - 1))
    else:
        seed = None

    # STEP 1: ADKGD ints -> strings.
    string_positives = []
    for h, r, t in adkgd_triples:
        string_positives.append(
            (adkgd_id2ent[h], adkgd_id2rel[r], adkgd_id2ent[t])
        )

    # Snapshot the cumulative stats so we can compute per-batch deltas.
    stats_before = corrupter.stats()

    # STEP 2: KGSAGE corruption (strings in, strings out).
    string_negatives = corrupter.corrupt_batch(string_positives, seed=seed)

    # STEP 3: strings -> ADKGD ints.
    adkgd_negatives = []
    for h, r, t in string_negatives:
        try:
            adkgd_negatives.append(
                (adkgd_ent2id[h], adkgd_rel2id[r], adkgd_ent2id[t])
            )
        except KeyError as exc:
            raise KeyError(
                f"ADKGD vocabulary missing entity/relation produced by KGSAGE: "
                f"{exc}. This usually means ADKGD's vocab differs from the "
                f"vocab the GAN was trained on. Re-train the GAN on ADKGD's "
                f"current train.txt, or check that data/<DATASET>/train.txt "
                f"hasn't drifted between the two sides."
            ) from exc

    # Per-batch stats = delta on the cumulative counters.
    stats_after = corrupter.stats()
    batch_stats = {
        k: stats_after[k] - stats_before.get(k, 0) for k in stats_after
    }
    return adkgd_negatives, batch_stats


def render_stats(stats):
    """Format batch stats for ADKGD's '[GAN] ...' log line.

    Mirrors the legacy Gumbel-Softmax format closely; the only structural
    difference is no `rel=` column - KGSAGE only corrupts head/tail slots.
    """
    total = stats.get("processed", 0)
    retries = stats.get("retries", 0)
    fb = stats.get("uniform_fallbacks", 0)
    head = stats.get("slot_head", 0)
    tail = stats.get("slot_tail", 0)
    if total == 0:
        return f"processed=0  retries=0  uniform_fallbacks=0"
    return (
        f"processed={total:,}  retries={retries}  uniform_fallbacks={fb}\n"
        f"       slot_distribution: "
        f"head={head}/{total}({head/total*100:.1f}%) "
        f"tail={tail}/{total}({tail/total*100:.1f}%)"
    )


def _derive_concept_pools_path(ckpt_path):
    """Guess the concept_pools.pkl path from the checkpoint path.

    Expected layout:
      experiments/gan/outputs/checkpoints/<DATASET>_kgsage.pt
      experiments/gan/outputs/concept_pools/<DATASET>.pkl
    """
    ckpt_dir = os.path.dirname(ckpt_path)
    name = os.path.basename(ckpt_path)
    # Strip "_kgsage.pt" suffix to get the dataset name.
    if name.endswith("_kgsage.pt"):
        dataset = name[:-len("_kgsage.pt")]
    elif name.endswith(".pt"):
        dataset = name[:-len(".pt")]
    else:
        dataset = name
    outputs_dir = os.path.dirname(ckpt_dir)
    return os.path.join(outputs_dir, "concept_pools", f"{dataset}.pkl")
