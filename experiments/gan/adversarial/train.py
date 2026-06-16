"""REINFORCE training loop for the CGSP adversarial pair.

Algorithm overview (one training step):

  STEP 1: For each positive (h,r,t) in the batch, build N_S candidate
          triples via candidate_pool.build_candidates(). The slot
          (head/tail) is chosen per-positive via Bernoulli sampling
          driven by the relation's cardinality stats.

  STEP 2: G scores all candidate triples -> P_G distribution.
          Each row sums to 1 over N_S candidates.

  STEP 3: For each positive, SAMPLE one candidate from P_G. Record
          log_prob of the sample (for REINFORCE) and the sampled
          "fake" triple itself.

  STEP 4: D scores the positives and the fakes.
          reward = -D.score(fake) - higher reward when fake confuses D.

  STEP 5: G UPDATE (REINFORCE policy gradient):
            advantage = reward - baseline_ema   (variance reduction)
            G_loss    = -(advantage * log_prob).mean()
          Gradients flow only through G.mlp; D's embeddings see
          gradients too (because G's forward uses them), but D's
          optimizer ignores them, and we zero them before D's update.

  STEP 6: D UPDATE (pairwise margin loss):
            D_loss = max(0, margin - score(pos) + score(fake)).mean()
          D's gradients update E and R.

  STEP 7: Clip gradients (CRITICAL for REINFORCE stability), step the
          two optimizers, renormalize D's entity embeddings.

  STEP 8: Update baseline_ema using the batch's mean reward.

WARMUP: For the first N warmup_epochs, only D is updated against random
negatives. This gives D a useful starting point before REINFORCE
engages - without warmup, G's reward signal is uninformative because
an untrained D scores everything similarly.

Outputs:
  outputs/checkpoints/<DATASET>_cgsp.pt        G + D weights, metadata
  outputs/logs/<DATASET>_cgsp_training.json    per-epoch loss curves
"""
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

# Make `from experiments.gan...` importable when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.gan.concept.concept_pools import load_pools  # noqa: E402
from experiments.gan.adversarial.discriminator import TransEDiscriminator  # noqa: E402
from experiments.gan.adversarial.generator import CandidateScorer  # noqa: E402
from experiments.gan.adversarial.candidate_pool import (  # noqa: E402
    build_candidates, build_candidate_triples,
)


# Locked hyperparameter defaults (from PIPELINE.md).
DEFAULT_BATCH_SIZE = 128
DEFAULT_EMBEDDING_DIM = 100
DEFAULT_HIDDEN_DIM = 256
DEFAULT_N_S = 64
DEFAULT_G_LR = 1e-4
DEFAULT_D_LR = 1e-3
DEFAULT_MARGIN = 0.5
DEFAULT_BASELINE_DECAY = 0.99
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_TOTAL_EPOCHS = 100
DEFAULT_GRADIENT_CLIP = 1.0
DEFAULT_CHECKPOINT_INTERVAL = 10


def warmup_one_epoch(D, real_triples, opt_d, n_entities,
                     batch_size, margin, gradient_clip, rng):
    """D-only training epoch using random negatives.

    Same signature as a real margin-loss epoch. Negatives are sampled
    uniformly from the full entity vocab (no concept filter) - the
    point is to get D's embeddings into a regime where it can
    distinguish pos from random, after which REINFORCE engages.
    """
    random.shuffle(real_triples)
    losses = []
    for i in range(0, len(real_triples), batch_size):
        batch = real_triples[i:i + batch_size]
        if len(batch) < 2:
            continue
        h_p = torch.tensor([h for h, r, t in batch])
        r_p = torch.tensor([r for h, r, t in batch])
        t_p = torch.tensor([t for h, r, t in batch])
        # Random negs - corrupt head or tail per-positive.
        h_n = h_p.clone()
        t_n = t_p.clone()
        for j in range(len(batch)):
            if rng.random() < 0.5:
                h_n[j] = rng.randint(0, n_entities - 1)
            else:
                t_n[j] = rng.randint(0, n_entities - 1)
        loss = D.margin_loss(h_p, r_p, t_p, h_n, r_p, t_n, margin=margin)
        opt_d.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(D.parameters(), gradient_clip)
        opt_d.step()
        D.renormalize_entities()
        losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


def reinforce_one_epoch(G, D, opt_g, opt_d, state, baseline_ema,
                        batch_size, n_s, margin, baseline_decay,
                        gradient_clip, rng):
    """One REINFORCE training epoch (G + D alternating updates).

    Returns a dict of epoch summary stats plus the updated baseline_ema.
    """
    real_triples = list(state["real_triple_set"])
    random.shuffle(real_triples)
    g_losses, d_losses, rewards = [], [], []

    for i in range(0, len(real_triples), batch_size):
        batch = real_triples[i:i + batch_size]
        if len(batch) < 2:
            continue
        B = len(batch)

        # STEP 1: build per-positive candidate triples [B, N_S, 3].
        all_cands = torch.zeros(B, n_s, 3, dtype=torch.long)
        for j, pos in enumerate(batch):
            slot, cand_ids, _ = build_candidates(pos, state, n_candidates=n_s, rng=rng)
            all_cands[j] = build_candidate_triples(pos, slot, cand_ids)

        # STEP 2: G scores -> P_G distribution over candidates.
        P_G = G.distribution_from_ids(all_cands, D.E, D.R)  # [B, N_S]

        # STEP 3: sample one fake per positive (Categorical).
        dist = torch.distributions.Categorical(probs=P_G)
        sampled_idx = dist.sample()                          # [B]
        log_prob = dist.log_prob(sampled_idx)                # [B]
        fake_triples = all_cands[torch.arange(B), sampled_idx]  # [B, 3]

        # STEP 4: D scores fakes (no grad - used only as reward).
        with torch.no_grad():
            d_fake_for_reward = D.score(
                fake_triples[:, 0], fake_triples[:, 1], fake_triples[:, 2]
            )
            reward = -d_fake_for_reward                       # [B]

        # STEP 5: G update (REINFORCE policy gradient).
        advantage = reward - baseline_ema
        g_loss = -(advantage * log_prob).mean()
        opt_g.zero_grad()
        opt_d.zero_grad()                                    # clear D's accumulated grads too
        g_loss.backward()
        nn.utils.clip_grad_norm_(G.parameters(), gradient_clip)
        opt_g.step()

        # STEP 6: D update (margin loss against G's fakes).
        h_p = torch.tensor([h for h, r, t in batch])
        r_p = torch.tensor([r for h, r, t in batch])
        t_p = torch.tensor([t for h, r, t in batch])
        opt_d.zero_grad()                                    # fresh grads for D's loss
        opt_g.zero_grad()
        d_loss = D.margin_loss(
            h_p, r_p, t_p,
            fake_triples[:, 0], fake_triples[:, 1], fake_triples[:, 2],
            margin=margin,
        )
        d_loss.backward()
        nn.utils.clip_grad_norm_(D.parameters(), gradient_clip)
        opt_d.step()
        D.renormalize_entities()

        # STEP 8: update baseline_ema with this batch's mean reward.
        batch_reward = reward.mean().item()
        baseline_ema = baseline_decay * baseline_ema + (1 - baseline_decay) * batch_reward

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        rewards.append(batch_reward)

    return {
        "g_loss": sum(g_losses) / max(len(g_losses), 1),
        "d_loss": sum(d_losses) / max(len(d_losses), 1),
        "mean_reward": sum(rewards) / max(len(rewards), 1),
        "baseline": baseline_ema,
    }


def train_cgsp(state, n_warmup_epochs, n_total_epochs, hp, log_path,
               checkpoint_path, seed=0, verbose=True,
               checkpoint_interval=DEFAULT_CHECKPOINT_INTERVAL):
    """End-to-end training loop. Returns the final state of (G, D, baseline).

    Saves checkpoint + log every `checkpoint_interval` epochs as a rolling
    overwrite. If the job is canceled mid-training, the most recent save
    is preserved (worst case: lose the last `checkpoint_interval` epochs).
    A final save always happens after the last epoch.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    n_ent = state["n_entities"]
    n_rel = state["n_relations"]
    real_triples = list(state["real_triple_set"])

    # Build models.
    D = TransEDiscriminator(n_ent, n_rel, dim=hp["embedding_dim"])
    G = CandidateScorer(embedding_dim=hp["embedding_dim"],
                        hidden_dim=hp["hidden_dim"])

    # Separate optimizers - G updates only its MLP head; D updates E and R.
    opt_g = torch.optim.Adam(G.parameters(), lr=hp["g_lr"])
    opt_d = torch.optim.Adam(D.parameters(), lr=hp["d_lr"])

    baseline_ema = 0.0  # initialised in the first batch's reward range
    epoch_records = []
    start_time = time.time()

    # ─── WARMUP: D-only on random negatives ────────────────────
    for ep in range(n_warmup_epochs):
        t0 = time.time()
        avg_d_loss = warmup_one_epoch(
            D, real_triples, opt_d, n_ent,
            hp["batch_size"], hp["margin"], hp["gradient_clip"], rng,
        )
        elapsed = time.time() - t0
        rec = {
            "epoch": ep,
            "phase": "warmup",
            "d_loss": avg_d_loss,
            "g_loss": None,
            "baseline": None,
            "mean_reward": None,
            "elapsed_seconds": elapsed,
        }
        epoch_records.append(rec)
        if verbose:
            print(f"  [warmup {ep+1}/{n_warmup_epochs}] d_loss={avg_d_loss:.4f} "
                  f"({elapsed:.1f}s)", flush=True)

    # ─── REINFORCE: G + D alternating ───────────────────────────
    for ep in range(n_warmup_epochs, n_total_epochs):
        t0 = time.time()
        stats = reinforce_one_epoch(
            G, D, opt_g, opt_d, state, baseline_ema,
            hp["batch_size"], hp["n_s"], hp["margin"],
            hp["baseline_decay"], hp["gradient_clip"], rng,
        )
        baseline_ema = stats["baseline"]
        elapsed = time.time() - t0
        rec = {
            "epoch": ep,
            "phase": "reinforce",
            "d_loss": stats["d_loss"],
            "g_loss": stats["g_loss"],
            "baseline": stats["baseline"],
            "mean_reward": stats["mean_reward"],
            "elapsed_seconds": elapsed,
        }
        epoch_records.append(rec)
        if verbose:
            print(
                f"  [reinforce {ep+1}/{n_total_epochs}] "
                f"d_loss={stats['d_loss']:.4f}  g_loss={stats['g_loss']:.4f}  "
                f"reward={stats['mean_reward']:.4f}  "
                f"baseline={stats['baseline']:.4f}  ({elapsed:.1f}s)",
                flush=True,
            )

        # Periodic checkpoint: rolling overwrite every checkpoint_interval epochs.
        # Protects against job cancellation / time limit / NaN crash later on.
        # The final epoch always saves (handled below the loop).
        epochs_done = ep + 1
        if epochs_done < n_total_epochs and epochs_done % checkpoint_interval == 0:
            _save_log(log_path, state, hp, epoch_records, time.time() - start_time)
            _save_checkpoint(checkpoint_path, state, hp, G, D, baseline_ema, epochs_done)
            if verbose:
                print(
                    f"  [checkpoint @ epoch {epochs_done}] -> {checkpoint_path}",
                    flush=True,
                )

    total_elapsed = time.time() - start_time
    if verbose:
        print(f"\n  Total training time: {total_elapsed/60:.2f} minutes", flush=True)

    # ─── Final save (always runs) ─────────────────────────────
    _save_log(log_path, state, hp, epoch_records, total_elapsed)
    _save_checkpoint(checkpoint_path, state, hp, G, D, baseline_ema, n_total_epochs)

    return G, D, baseline_ema


def _save_log(log_path, state, hp, epoch_records, total_elapsed):
    """Write per-epoch loss curves to JSON for plotting / inspection."""
    log = {
        "schema_version": 1,
        "dataset_name": state.get("dataset_name", "unknown"),
        "dataset_hash": state["dataset_hash"],
        "hyperparameters": hp,
        "epochs": epoch_records,
        "total_elapsed_seconds": total_elapsed,
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "produced_by": "kg_corrupter v0.1",
    }
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
        f.write("\n")


def _save_checkpoint(checkpoint_path, state, hp, G, D, baseline_ema, n_epochs_run):
    """Save G + D + metadata for Phase 3 corruption to load."""
    payload = {
        "schema_version": 1,
        "dataset_hash": state["dataset_hash"],
        "n_entities": state["n_entities"],
        "n_relations": state["n_relations"],
        "embedding_dim": hp["embedding_dim"],
        "hidden_dim": hp["hidden_dim"],
        "p_norm": 2,
        "generator_state": G.state_dict(),
        "discriminator_state": D.state_dict(),
        "hyperparameters": hp,
        "training_metadata": {
            "epochs_run": n_epochs_run,
            "final_baseline": baseline_ema,
        },
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "produced_by": "kg_corrupter v0.1",
    }
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def main():
    """CLI entry point: train CGSP on one dataset."""
    # Defensive defaults for Windows/CPU - cluster overrides via env take precedence.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    args = _parse_args()

    project_root = Path(args.project_root).resolve()
    cache_path = project_root / "experiments" / "gan" / "outputs" / "concept_pools" / f"{args.dataset}.pkl"
    if not cache_path.exists():
        print(f"!! concept_pools cache not found at {cache_path}", file=sys.stderr)
        print(f"   Run Phase 1 first:", file=sys.stderr)
        print(f"     python -m experiments.gan.concept.preprocess --dataset {args.dataset} --family <family>",
              file=sys.stderr)
        return 1

    print(f"Loading concept_pools cache: {cache_path}", flush=True)
    state = load_pools(str(cache_path))
    state["dataset_name"] = args.dataset

    hp = {
        "batch_size": args.batch_size,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "n_s": args.n_s,
        "g_lr": args.g_lr,
        "d_lr": args.d_lr,
        "margin": args.margin,
        "baseline_decay": args.baseline_decay,
        "gradient_clip": args.gradient_clip,
        "warmup_epochs": args.warmup_epochs,
        "total_epochs": args.total_epochs,
        "seed": args.seed,
    }
    print(f"Hyperparameters: {hp}", flush=True)

    checkpoint_path = project_root / "experiments" / "gan" / "outputs" / "checkpoints" / f"{args.dataset}_cgsp.pt"
    log_path = project_root / "experiments" / "gan" / "outputs" / "logs" / f"{args.dataset}_cgsp_training.json"

    print(f"\nTraining CGSP on {args.dataset}", flush=True)
    print(f"  {state['n_entities']:,} entities, {state['n_relations']} relations, "
          f"{len(state['real_triple_set']):,} triples", flush=True)
    print(f"  warmup: {args.warmup_epochs} epochs | reinforce: {args.total_epochs - args.warmup_epochs} epochs", flush=True)
    print(f"", flush=True)

    G, D, final_baseline = train_cgsp(
        state, args.warmup_epochs, args.total_epochs, hp,
        log_path=str(log_path), checkpoint_path=str(checkpoint_path),
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
    )

    print(f"\nDone.")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  log:        {log_path}")
    return 0


def _parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", required=True,
                    help="dataset folder name (concept_pools cache must already exist)")
    ap.add_argument("--project_root", default=str(_PROJECT_ROOT))
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--embedding_dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    ap.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM)
    ap.add_argument("--n_s", type=int, default=DEFAULT_N_S,
                    help="number of candidate triples per positive")
    ap.add_argument("--g_lr", type=float, default=DEFAULT_G_LR)
    ap.add_argument("--d_lr", type=float, default=DEFAULT_D_LR)
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--baseline_decay", type=float, default=DEFAULT_BASELINE_DECAY)
    ap.add_argument("--gradient_clip", type=float, default=DEFAULT_GRADIENT_CLIP)
    ap.add_argument("--warmup_epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    ap.add_argument("--total_epochs", type=int, default=DEFAULT_TOTAL_EPOCHS,
                    help="total epochs (includes warmup)")
    ap.add_argument("--checkpoint_interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL,
                    help="save checkpoint + log every N REINFORCE epochs (rolling overwrite). "
                         "Protects against cancellation / time-limit / NaN crash.")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
