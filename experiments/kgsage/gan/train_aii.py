"""A-ii adversarial training: frozen ComplEx backbone + trainable contextual
residual discriminator (stage A4; design: experiments/docs/OPTION_A_PLAN.md).

Phases
  0b WARMUP    encoder + throwaway DistMult decoder train on link prediction
               over the train graph, then the encoder FREEZES -> E' (the only
               job of this phase is giving f_theta / G real neighbourhood
               features; the decoder is discarded -- plausibility comes from
               the LibKGE ComplEx, which we never train).
  1  WARMSTART generator gets a few epochs of CE toward band-teacher draws
               (frozen-ComplEx top-k below s(true), masked); CE is applied
               ONLY to the corrupted slot's head -- no copy supervision, no
               relation-slot supervision. Then the CE is dropped forever.
  2  ADVERSARIAL (1:1 steps)
       D step (f_theta only): positives = real train triples; fakes = 50% G
         hard samples + 25% band draws + 25% type-valid random + replay
         buffer. L_D = BCE(D(real), 1-ls) + BCE(D(fake), 0) + l_res*f^2.
       G step: masked straight-through Gumbel on ONE slot;
         L_G = -[s_z + f_theta] + l_fence*hinge(s_f(g) - (s_f(true) - m_r))
               - l_H * entropy(masked logits).
       All falseness guarantees are structural (masks) or frozen (fence).

Legacy note: the B1a arm (kgsage.gan.train + ContradictionTargetSampler)
is untouched and remains runnable for ablation; this module shares the
checkpoint format via train.save_checkpoint(extra=...), so inference and the
ADKGD bridge consume A-ii checkpoints unchanged.

Run (repo root):
  PYTHONPATH=experiments python -m kgsage.gan.train_aii --data data/FB15K-mini \
      --lp_ckpt experiments/kgsage/outputs/lp/fb15k-237-complex.pt \
      --lp_ids  experiments/kgsage/outputs/lp/fb15k-237 \
      --out experiments/kgsage/outputs/checkpoints/kgsage_aii_mini.pt
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from kgsage.data.loaders import load_kg, build_edge_index
from kgsage.gan.encoder import KGSAGEEncoder
from kgsage.gan.models import KGSAGEGenerator, gumbel_softmax
from kgsage.gan.masks import CandidateMasks, HEAD, TAIL
from kgsage.gan.complex_d import FrozenComplEx
from kgsage.gan.residual_d import ResidualContextD
from kgsage.gan.train import save_checkpoint
from kgsage.lp_scorer import ComplExScorer


# ---------------------------------------------------------------------------
# frozen assets
# ---------------------------------------------------------------------------

def load_frozen_complex(lp_ckpt: str, lp_ids: str, kg: dict) -> FrozenComplEx:
    """Load the LibKGE scorer and realign its tables into the GAN vocab."""
    scorer = ComplExScorer.from_libkge(lp_ckpt, lp_ids)
    try:
        ent_perm = torch.tensor([scorer.ent2row[kg["id2ent"][i]]
                                 for i in range(kg["n_ent"])], dtype=torch.long)
        rel_perm = torch.tensor([scorer.rel2base[kg["id2rel"][i]]
                                 for i in range(kg["n_rel"])], dtype=torch.long)
    except KeyError as exc:
        raise ValueError(f"GAN vocab symbol missing from the LP scorer map: {exc} "
                         "-- wrong --lp_ckpt/--lp_ids for this dataset?") from exc
    return FrozenComplEx.from_lp_scorer(scorer).realigned(ent_perm, rel_perm)


def relation_zstats(frozen: FrozenComplEx, real_all: torch.Tensor,
                    n_rel: int, sample_per_rel: int = 2000,
                    seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-relation mean/std of forward s_f over (sampled) train triples --
    puts raw ComplEx scores on a comparable scale across relations so
    s_z + f_theta is a sane sum and fence margins can be sigma-relative."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.zeros(n_rel)
    sigma = torch.ones(n_rel)
    with torch.no_grad():
        for r in range(n_rel):
            rows = real_all[real_all[:, 1] == r]
            if rows.shape[0] == 0:
                continue
            if rows.shape[0] > sample_per_rel:
                rows = rows[torch.randperm(rows.shape[0], generator=g)[:sample_per_rel]]
            s = frozen.score_hard(rows[:, 0], rows[:, 1], rows[:, 2])
            mu[r] = s.mean()
            sigma[r] = s.std().clamp(min=1e-3) if rows.shape[0] > 1 else 1.0
    return mu, sigma


# ---------------------------------------------------------------------------
# band teacher (tensorised close-but-false draws in GAN-id space)
# ---------------------------------------------------------------------------

def band_teacher_draw(frozen: FrozenComplEx, masks: CandidateMasks,
                      h, r, t, slot: int, band_k: int, band_temp: float,
                      generator: torch.Generator) -> torch.Tensor:
    """[B] close-but-false slot fillers: masked, below s(true), top-k by rank,
    softmax-sampled. Mirrors kgsage.band_sampler in tensor form."""
    with torch.no_grad():
        if slot == TAIL:
            scores = frozen.score_tails_all(h, r)
            s_true = scores[torch.arange(len(h)), t]
        else:
            scores = frozen.score_heads_all(r, t)
            s_true = scores[torch.arange(len(h)), h]
        mask, _ = masks.logits_mask(h, r, t, slot)
        scores = scores + mask
        # drop candidates at/above the true value (FN control); if that empties
        # a row, fall back to the masked pool as-is
        below = scores.masked_fill(scores >= s_true.unsqueeze(1), float("-inf"))
        dead = torch.isinf(below).all(dim=1)
        below[dead] = scores[dead]
        topv, topi = below.topk(min(band_k, below.shape[1]), dim=1)
        probs = torch.softmax(
            topv.masked_fill(torch.isinf(topv), -1e30) / max(band_temp, 1e-6), dim=1)
        pick = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        return topi[torch.arange(len(h)), pick]


def assemble(h, r, t, fill, slot: int) -> torch.Tensor:
    """[B,3] triples with `fill` substituted into `slot`."""
    out = torch.stack([h, r, t], dim=1).clone()
    out[:, 0 if slot == HEAD else 2] = fill
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lp_ckpt", required=True)
    ap.add_argument("--lp_ids", required=True)
    ap.add_argument("--train_split", choices=["train", "all"], default="train")
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--warmup_batch", type=int, default=4096)
    ap.add_argument("--warmstart_epochs", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30, help="adversarial epochs")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr_g", type=float, default=1e-4)
    ap.add_argument("--lr_d", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--z_dim", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--band_k", type=int, default=10)
    ap.add_argument("--band_temp", type=float, default=0.5)
    ap.add_argument("--fence_sigma", type=float, default=0.5,
                    help="fence margin = this many per-relation sigmas below s_f(true)")
    ap.add_argument("--lambda_fence", type=float, default=1.0)
    ap.add_argument("--lambda_h", type=float, default=0.01)
    ap.add_argument("--lambda_res", type=float, default=1e-3)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--beta_residual", type=float, default=1.0)
    ap.add_argument("--num_bases", type=int, default=30)
    ap.add_argument("--encoder_layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    sample_gen = torch.Generator().manual_seed(args.seed + 1)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)

    kg = load_kg(args.data)
    if args.train_split == "all":
        train_triples = (list(kg["triples_train"]) + list(kg["triples_valid"])
                         + list(kg["triples_test"]))
        kg["edge_index"], kg["edge_type"] = build_edge_index(train_triples)
    else:
        train_triples = list(kg["triples_train"])
    kg["triple_set"] = kg["triple_set_all"]
    real_all = torch.tensor(train_triples, dtype=torch.long)
    n_ent, n_rel = kg["n_ent"], kg["n_rel"]
    print(f"KG: {n_ent:,} entities, {n_rel:,} relations, "
          f"{len(train_triples):,} '{args.train_split}' triples", flush=True)

    frozen = load_frozen_complex(args.lp_ckpt, args.lp_ids, kg)
    masks = CandidateMasks(kg, n_ent, n_rel)
    mu_r, sigma_r = relation_zstats(frozen, real_all, n_rel, seed=args.seed)
    print(f"Frozen ComplEx realigned (reciprocal={frozen.reciprocal}); "
          f"per-relation z-stats ready", flush=True)

    edge_index, edge_type = KGSAGEEncoder.to_tensors(kg["edge_index"], kg["edge_type"], device)
    encoder = KGSAGEEncoder(n_ent, n_rel, dim=args.dim, num_bases=args.num_bases,
                            num_layers=args.encoder_layers).to(device)

    # ---------------- Phase 0b: LP warmup, then freeze E' ----------------
    dec_rel = torch.nn.Parameter(torch.randn(n_rel, args.dim) * 0.1)
    warm_opt = torch.optim.Adam(list(encoder.parameters()) + [dec_rel], lr=1e-3)
    t0 = time.perf_counter()
    for epoch in range(1, args.warmup_epochs + 1):
        perm = torch.randperm(real_all.shape[0], generator=sample_gen)
        tot, nb = 0.0, 0
        for start in range(0, len(perm), args.warmup_batch):
            rows = real_all[perm[start:start + args.warmup_batch]]
            h, r, t = rows[:, 0], rows[:, 1], rows[:, 2]
            t_neg = torch.randint(0, n_ent, (len(rows),), generator=sample_gen)
            ctx = encoder(edge_index, edge_type)          # full-graph forward
            pos = (ctx[h] * dec_rel[r] * ctx[t]).sum(-1)  # DistMult decoder
            neg = (ctx[h] * dec_rel[r] * ctx[t_neg]).sum(-1)
            loss = (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                    + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
            warm_opt.zero_grad(); loss.backward(); warm_opt.step()
            tot += loss.item(); nb += 1
        print(f"  warmup {epoch:3d}/{args.warmup_epochs}  LP_loss={tot / nb:.4f}", flush=True)
    encoder.eval()
    encoder.requires_grad_(False)
    with torch.no_grad():
        context = encoder(edge_index, edge_type).detach()  # THE frozen E'
    del dec_rel, warm_opt
    print(f"Encoder frozen after warmup ({time.perf_counter() - t0:.1f}s); "
          f"E' cached [{context.shape[0]}, {context.shape[1]}]", flush=True)

    generator = KGSAGEGenerator(n_ent, n_rel, dim=args.dim, z_dim=args.z_dim).to(device)
    f_theta = ResidualContextD(n_rel, dim=args.dim, beta=args.beta_residual).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(f_theta.parameters(), lr=args.lr_d)

    def z_of(s_raw, r_ids):
        return (s_raw - mu_r[r_ids]) / sigma_r[r_ids]

    def g_sample(h, r, t, slot, hard=True, with_grad=False):
        """Generator slot sample under masks. Returns (soft/ST one-hot, logits_masked)."""
        noise = torch.randn(len(h), generator.z_dim, generator=sample_gen if not with_grad else None)
        head_logits, _, tail_logits = generator(h, r, t, noise, context)
        logits = tail_logits if slot == TAIL else head_logits
        mask, _ = masks.logits_mask(h, r, t, slot)
        sample = gumbel_softmax(logits, tau=args.tau, hard=hard, mask=mask,
                                generator=None if with_grad else sample_gen)
        return sample, logits + mask

    # ---------------- Phase 1: G warm start (band-teacher CE) ----------------
    for epoch in range(1, args.warmstart_epochs + 1):
        perm = torch.randperm(real_all.shape[0], generator=sample_gen)
        tot, nb = 0.0, 0
        for bi, start in enumerate(range(0, len(perm), args.batch_size)):
            rows = real_all[perm[start:start + args.batch_size]]
            if len(rows) < 2:
                continue
            h, r, t = rows[:, 0], rows[:, 1], rows[:, 2]
            slot = TAIL if bi % 2 == 0 else HEAD
            teach = band_teacher_draw(frozen, masks, h, r, t, slot,
                                      args.band_k, args.band_temp, sample_gen)
            noise = torch.randn(len(h), generator.z_dim)
            head_logits, _, tail_logits = generator(h, r, t, noise, context)
            logits = tail_logits if slot == TAIL else head_logits
            loss = F.cross_entropy(logits, teach)   # ONLY the corrupted slot
            opt_g.zero_grad(); loss.backward(); opt_g.step()
            tot += loss.item(); nb += 1
        print(f"  warmstart {epoch}/{args.warmstart_epochs}  CE={tot / nb:.4f}", flush=True)

    # ---------------- Phase 2: adversarial ----------------
    replay: deque = deque(maxlen=10_000)
    print("-" * 60, flush=True)
    print(f"Adversarial: {args.epochs} epochs, batch={args.batch_size}, tau={args.tau}, "
          f"fence={args.fence_sigma}sigma, l_H={args.lambda_h}", flush=True)
    start_wall = datetime.now()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(real_all.shape[0], generator=sample_gen)
        ep = {"d_loss": 0.0, "g_loss": 0.0, "fence_hit": 0, "above_true": 0,
              "copy_argmax": 0, "d_real_ok": 0, "d_fake_ok": 0, "n": 0, "nb": 0}
        picks_per_slot: dict[int, set] = {}
        for bi, start in enumerate(range(0, len(perm), args.batch_size)):
            rows = real_all[perm[start:start + args.batch_size]]
            if len(rows) < 4:
                continue
            h, r, t = rows[:, 0], rows[:, 1], rows[:, 2]
            B = len(rows)
            slot = TAIL if bi % 2 == 0 else HEAD

            # ---- D step (f_theta only) ----
            with torch.no_grad():
                g_soft, _ = g_sample(h, r, t, slot, hard=True)
                g_pick = g_soft.argmax(dim=1)
            n_g, n_band = B // 2, B // 4
            band_pick = band_teacher_draw(frozen, masks, h[:n_band], r[:n_band],
                                          t[:n_band], slot, args.band_k,
                                          args.band_temp, sample_gen)
            mask_rand, _ = masks.logits_mask(h[n_g + n_band:], r[n_g + n_band:],
                                             t[n_g + n_band:], slot)
            rand_pick = gumbel_softmax(torch.zeros_like(mask_rand), tau=1.0,
                                       hard=True, mask=mask_rand,
                                       generator=sample_gen).argmax(dim=1)
            fakes = torch.cat([
                assemble(h[:n_g], r[:n_g], t[:n_g], g_pick[:n_g], slot),
                assemble(h[:n_band], r[:n_band], t[:n_band], band_pick, slot),
                assemble(h[n_g + n_band:], r[n_g + n_band:], t[n_g + n_band:],
                         rand_pick, slot),
            ])
            for row in fakes[:max(1, B // 10)].tolist():
                replay.append(tuple(row))
            if len(replay) >= 32:
                ridx = torch.randint(0, len(replay), (B // 10,), generator=sample_gen)
                fakes = torch.cat([fakes, torch.tensor([list(replay[i]) for i in ridx])])

            def d_score(trip):
                s_raw = frozen.score_hard(trip[:, 0], trip[:, 1], trip[:, 2])
                return z_of(s_raw, trip[:, 1]) + f_theta.from_ids(
                    context, trip[:, 0], trip[:, 1], trip[:, 2])

            d_real = d_score(rows)
            d_fake = d_score(fakes)
            l_d = (F.binary_cross_entropy_with_logits(
                       d_real, torch.full_like(d_real, 1.0 - args.label_smoothing))
                   + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
                   + args.lambda_res * (f_theta.from_ids(context, rows[:, 0], rows[:, 1],
                                                         rows[:, 2]) ** 2).mean())
            opt_d.zero_grad(); l_d.backward(); opt_d.step()

            # ---- G step ----
            g_soft, logits_masked = g_sample(h, r, t, slot, hard=True, with_grad=True)
            cand_emb = g_soft @ context                       # frozen E'
            if slot == TAIL:
                s_soft = frozen.score_soft_tail(h, r, g_soft)
                f_soft = f_theta(context[h], r, cand_emb)
                s_true = frozen.score_tails_all(h, r)[torch.arange(B), t]
            else:
                s_soft = frozen.score_soft_head(g_soft, r, t)
                f_soft = f_theta(cand_emb, r, context[t])
                s_true = frozen.score_heads_all(r, t)[torch.arange(B), h]
            fence = torch.relu(s_soft - (s_true - args.fence_sigma * sigma_r[r]))
            probs = torch.softmax(logits_masked, dim=1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
            l_g = (-(z_of(s_soft, r) + f_soft)
                   + args.lambda_fence * fence
                   - args.lambda_h * entropy).mean()
            opt_g.zero_grad(); l_g.backward(); opt_g.step()

            # ---- diagnostics ----
            with torch.no_grad():
                ep["d_loss"] += l_d.item(); ep["g_loss"] += l_g.item()
                ep["fence_hit"] += int((fence > 0).sum())
                ep["above_true"] += int((s_soft.detach() > s_true).sum())
                pick = g_soft.argmax(dim=1)
                true_slot = t if slot == TAIL else h
                ep["copy_argmax"] += int((pick == true_slot).sum())
                ep["d_real_ok"] += int((torch.sigmoid(d_real) > 0.5).sum())
                ep["d_fake_ok"] += int((torch.sigmoid(d_fake) < 0.5).sum())
                ep["n"] += B; ep["n_fake"] = ep.get("n_fake", 0) + len(fakes)
                ep["nb"] += 1
                picks_per_slot.setdefault(slot, set()).update(pick.tolist())

        n, nb = max(ep["n"], 1), max(ep["nb"], 1)
        distinct = sum(len(v) for v in picks_per_slot.values())
        print(f"  epoch {epoch:3d}/{args.epochs}  D={ep['d_loss'] / nb:.4f} "
              f"G={ep['g_loss'] / nb:.4f}  fence-hit={ep['fence_hit'] / n:.1%} "
              f"above-true={ep['above_true'] / n:.1%} copy={ep['copy_argmax'] / n:.2%} "
              f"D-acc real/fake={ep['d_real_ok'] / n:.2f}/{ep['d_fake_ok'] / max(ep.get('n_fake', 1), 1):.2f} "
              f"distinct-picks={distinct}", flush=True)

    print(f"Adversarial done ({datetime.now() - start_wall}).", flush=True)

    # ---------------- checkpoint (legacy format + A-ii extras) ----------------
    extra = {
        "arm": "aii",
        "lp_ckpt": str(args.lp_ckpt),
        "pool_masks": masks.pool,          # [2, n_rel, n_ent] bool (A5 decode)
        "zstats_mu": mu_r, "zstats_sigma": sigma_r,
        "fence_sigma": args.fence_sigma,
        "tau": args.tau,
    }
    save_checkpoint(generator, encoder, kg, args.dim, args.z_dim,
                    edge_index, edge_type, args.out, extra=extra)
    print(f"Saved A-ii checkpoint to {args.out}", flush=True)


if __name__ == "__main__":
    main()
