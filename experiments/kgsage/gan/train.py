"""KGSAGE-2 trainer: LP-free dual-discriminator adversarial training.

    D_real  (trainable, spectral-normed)  "is this triple plausible?"  G: HIGH
    D_match (pretrained on real-vs-mismatched pairs, then FROZEN)
            "does the candidate fit THIS anchor's neighbourhood?"      G: LOW
    L_G = -D_real + alpha * D_match,  alpha PID-controlled on the MEASURED
    corroborated probability mass (exact graph support, not D_match's opinion).

No pretrained link predictor anywhere. Learned components (sketches, candidate
pools, neighbour lists, both discriminators) see the TRAIN SPLIT ONLY; the all-splits
falseness guarantee remains where it always was -- the decode-time masks.

The locked hyperparameters live as module-level constants below (edit there to
re-tune); only operational flags stay on the CLI.

Usage (repo root, pytorch env):
  PYTHONPATH=experiments python -m kgsage.gan.train \
      --data data/FB15K-237 --out experiments/kgsage/outputs/checkpoints/run.pt \
      --epochs 8 --snapshot_every 1 [--init_context_from <v1_ckpt> --device cpu]
"""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from kgsage.data.loaders import load_kg, build_edge_index
from kgsage.gan.encoder import KGSAGEEncoder
from kgsage.gan.generator import CandidateScoringGenerator, gumbel_softmax
from kgsage.gan.d_real import DReal
from kgsage.gan.d_match import DMatch
from kgsage.gan.sketch import build_sketches
from kgsage.gan.candidates import CandidateSampler

HEAD, TAIL = 0, 2

# ==========================================================================
# Locked configuration. Tuned during the KGSAGE-2 diagnosis and fixed for
# every run; only operational args (data/out/device/seed/epochs/snapshots/
# E'-reuse) stay on the CLI. Re-tune by editing here.
# ==========================================================================

# -- architecture (E', sketches, candidate set) --
DIM             = 64      # context-vector / embedding width
NUM_BASES       = 30      # RGCN basis decomposition
ENCODER_LAYERS  = 2       # RGCN depth -> 2-hop context
SKETCH_BITS     = 8192    # Bloom membership-sketch length
N_NBR           = 32      # neighbours D_match attends over
CAND_K          = 256     # candidates scored per triple (decode = full pool)

# -- E' warm-up (skipped when --init_context_from is given) --
WARMUP_EPOCHS   = 10
WARMUP_BATCH    = 4096

# -- curriculum: pretrain both discriminators, then alpha=0 warm-up --
DMATCH_EPOCHS         = 2
DREAL_PRETRAIN_EPOCHS = 2
ALPHA_WARMUP_EPOCHS   = 2

# -- game optimisation --
BATCH_SIZE      = 256
TAU             = 0.5     # Gumbel-Softmax temperature (train + decode)
LR_G            = 1e-4
LR_D            = 3e-4    # D_real
LR_DMATCH       = 1e-4    # D_match online hardening on G's picks (oracle labels);
                          # a frozen D_match was exploited (48% of picks corroborated)
LABEL_SMOOTHING = 0.1

# -- contradiction-pressure controller (alpha) --
ALPHA_TARGET    = 0.13    # PID set-point; MUST stay above the ~0.12-0.13 structural
                          # floor or alpha rails and G collapses to universal-aliens
ALPHA_INIT      = 1.0     # value alpha is released at when the warm-up ends
ALPHA_MAX       = 10.0    # anti-windup clamp (a smoke wound alpha to 284)
ALPHA_KP        = 2.0     # PID proportional gain
ALPHA_KI        = 0.2     # PID integral gain
MATCH_MARGIN    = 0.0     # hinge: alpha*relu(D_match - margin); constraint, not reward
DREAL_MISMATCH  = 1.0     # GAN-CLS wrong-anchor weight -> anchor-conditional D_real


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8,
                    help="adversarial game epochs (locked recipe: 8; use 2 for a "
                         "quick smoke). Anchor-awareness peaks a few pressure "
                         "epochs after the alpha ramp and then erodes, so keep "
                         "runs short and select across snapshots by knockout J@10.")
    ap.add_argument("--snapshot_every", type=int, default=0,
                    help="save a full loadable checkpoint every N game epochs once "
                         "the alpha ramp starts (0 = off). Select the reported "
                         "generator across snapshots by knockout J@10.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--init_context_from", default=None,
                    help="load frozen E' from an existing checkpoint instead of "
                         "running the RGCN warm-up (CPU path; vocabularies must match)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)

    kg = load_kg(args.data)
    train_triples = list(kg["triples_train"])
    n_ent, n_rel = kg["n_ent"], kg["n_rel"]
    real_all = torch.tensor(train_triples, dtype=torch.long)
    print(f"KG: {n_ent:,} entities, {n_rel:,} relations, "
          f"{len(train_triples):,} train triples", flush=True)

    # ---------- Phase 0: frozen E' (RGCN warm-up, or reuse an existing table) ----------
    if args.init_context_from:
        payload = torch.load(args.init_context_from, map_location=device,
                             weights_only=False)
        assert payload["n_ent"] == n_ent and payload["dim"] == DIM, \
            "context table from --init_context_from does not match this KG"
        # vocab must match too (string-keyed check on a sample)
        for k in list(kg["ent2id"])[:50]:
            assert payload["ent2id"].get(k) == kg["ent2id"][k], \
                f"vocab mismatch at {k!r}"
        context = payload["context_embeddings"].to(device).detach()
        sample_gen = torch.Generator().manual_seed(args.seed + 1)
        print(f"E' loaded from {args.init_context_from} "
              f"[{context.shape[0]}, {context.shape[1]}]", flush=True)
        _skip_warmup = True
    else:
        _skip_warmup = False
    edge_index, edge_type = KGSAGEEncoder.to_tensors(kg["edge_index"], kg["edge_type"], device)
    encoder = KGSAGEEncoder(n_ent, n_rel, dim=DIM, num_bases=NUM_BASES,
                            num_layers=ENCODER_LAYERS).to(device)
    if _skip_warmup:
        encoder = None
    if not _skip_warmup:
        dec_rel = torch.nn.Parameter(torch.randn(n_rel, DIM, device=device) * 0.1)
        warm_opt = torch.optim.Adam(list(encoder.parameters()) + [dec_rel], lr=1e-3)
        sample_gen = torch.Generator().manual_seed(args.seed + 1)
        for epoch in range(1, WARMUP_EPOCHS + 1):
            perm = torch.randperm(real_all.shape[0], generator=sample_gen)
            tot = nb = 0
            for s in range(0, len(perm), WARMUP_BATCH):
                rows = real_all[perm[s:s + WARMUP_BATCH]].to(device)
                h, r, t = rows[:, 0], rows[:, 1], rows[:, 2]
                t_neg = torch.randint(0, n_ent, (len(rows),), generator=sample_gen).to(device)
                ctx = encoder(edge_index, edge_type)
                pos = (ctx[h] * dec_rel[r] * ctx[t]).sum(-1)
                neg = (ctx[h] * dec_rel[r] * ctx[t_neg]).sum(-1)
                loss = (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                        + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
                warm_opt.zero_grad(); loss.backward(); warm_opt.step()
                tot += loss.item(); nb += 1
            print(f"  warmup {epoch}/{WARMUP_EPOCHS} LP_loss={tot/max(nb,1):.4f}", flush=True)
        encoder.eval(); encoder.requires_grad_(False)
        with torch.no_grad():
            context = encoder(edge_index, edge_type).detach()
        del dec_rel, warm_opt
        print(f"E' frozen [{context.shape[0]}, {context.shape[1]}]", flush=True)

    # ---------- Train-split structures (learned components see TRAIN only) ----------
    adj: dict[int, set] = {}
    true_tails: dict[tuple, set] = defaultdict(set)
    true_heads: dict[tuple, set] = defaultdict(set)
    by_rel: dict[int, list] = defaultdict(list)
    for h, r, t in train_triples:
        adj.setdefault(h, set()).add(t); adj.setdefault(t, set()).add(h)
        true_tails[(h, r)].add(t); true_heads[(r, t)].add(h)
        by_rel[r].append((h, r, t))

    print("building sketches (train split)...", flush=True)
    sketches = build_sketches(train_triples, n_ent, m=SKETCH_BITS,
                              seed=args.seed).float()
    cs = CandidateSampler(train_triples, n_ent, n_rel, k=CAND_K,
                          seed=args.seed)
    # type pools for the checkpoint (decode scores the FULL pool; same
    # train-split pools the v1 masks used)
    pool_masks = torch.zeros(2, n_rel, n_ent, dtype=torch.bool)
    for h_, r_, t_ in train_triples:
        pool_masks[0, r_, h_] = True
        pool_masks[1, r_, t_] = True

    def nbr_batch(anchors, exclude=None):
        """[B, n_nbr] ids + mask from TRAIN adjacency."""
        B = len(anchors)
        ids = torch.zeros(B, N_NBR, dtype=torch.long)
        msk = torch.zeros(B, N_NBR, dtype=torch.bool)
        for i, a in enumerate(anchors):
            ns = adj.get(int(a), ())
            ns = [n for n in ns if exclude is None or n != int(exclude[i])]
            if not ns:
                continue
            if len(ns) > N_NBR:
                ns = rng.sample(ns, N_NBR)
            ids[i, :len(ns)] = torch.tensor(ns)
            msk[i, :len(ns)] = True
        return ids, msk

    _sup_cache: dict[int, torch.Tensor] = {}

    def support_row_train(a: int) -> torch.Tensor:
        """Cached bool [n_ent]: entities supported by anchor a (1-hop or
        shared-neighbour), TRAIN adjacency. ~n_ent bits per cached anchor."""
        row = _sup_cache.get(a)
        if row is None:
            na = adj.get(a, set())
            row = torch.zeros(n_ent, dtype=torch.bool)
            if na:
                idx = torch.tensor(sorted(na), dtype=torch.long)
                row[idx] = True                      # 1-hop
                two = set()
                for nb1 in na:
                    two |= adj.get(nb1, set())
                if two:
                    row[torch.tensor(sorted(two), dtype=torch.long)] = True
            row[a] = False
            if len(_sup_cache) < 20000:
                _sup_cache[a] = row
        return row

    def support_flags(anchors, cand_ids):
        """Exact graph support of candidates wrt anchors (bool [B, K]) --
        the PID's measurement signal and the epoch diagnostic. Vectorised via
        the per-anchor cached row (first epoch pays the build, then O(1))."""
        out = torch.zeros_like(cand_ids, dtype=torch.bool)
        for i, a in enumerate(anchors.tolist()):
            out[i] = support_row_train(a)[cand_ids[i]]
        return out

    # ---------- Phase 1: pretrain D_match (real vs same-relation mismatch), freeze ----------
    dmatch = DMatch(dim=DIM).to(device)
    dm_opt = torch.optim.AdamW(dmatch.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    print(f"D_match pretraining ({DMATCH_EPOCHS} epochs)...", flush=True)
    for ep in range(1, DMATCH_EPOCHS + 1):
        perm = rng.sample(train_triples, len(train_triples))
        tot = nb = 0
        for s in range(0, len(perm), BATCH_SIZE):
            chunk = perm[s:s + BATCH_SIZE]
            anchors, cands, ys = [], [], []
            for h, r, t in chunk:
                anchors.append(h); cands.append(t); ys.append(1.0)
                others = by_rel[r]
                for _ in range(6):
                    _, _, t2 = others[rng.randrange(len(others))]
                    if t2 != t and t2 not in adj.get(h, set()):
                        anchors.append(h); cands.append(t2); ys.append(0.0)
                        break
            a_t = torch.tensor(anchors); c_t = torch.tensor(cands)
            nid, nmk = nbr_batch(a_t, exclude=c_t)
            logit = dmatch(context[c_t], context[nid], nmk.to(device))
            loss = bce(logit, torch.tensor(ys, device=device))
            dm_opt.zero_grad(); loss.backward(); dm_opt.step()
            tot += loss.item(); nb += 1
        print(f"  dmatch {ep}/{DMATCH_EPOCHS} bce={tot/max(nb,1):.4f}", flush=True)
    # NOT frozen: the FB dynamics smoke proved a frozen D_match gets exploited
    # (G converges onto its false negatives). It keeps training during the game
    # on the generator's own picks with ORACLE labels (exact graph support), so
    # every blind spot G finds is corrected on the next batch. The oracle only
    # supplies labels -- D_match remains a learned discriminator.
    dm_game_opt = torch.optim.AdamW(dmatch.parameters(), lr=LR_DMATCH)

    # ---------- Phase 2: the dual-discriminator game ----------
    G = CandidateScoringGenerator(dim=DIM, sketch_bits=SKETCH_BITS,
                                  n_rel=n_rel).to(device)
    dreal = DReal(dim=DIM, n_rel=n_rel).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(dreal.parameters(), lr=LR_D)

    # ---- Phase 1b: D_real pretrain (anchor-conditional plausibility must
    #      exist BEFORE G starts learning, or G falls into the universal-alien
    #      basin the alienation signal digs on epoch 1) ----
    for ep_i in range(1, DREAL_PRETRAIN_EPOCHS + 1):
        perm = torch.randperm(real_all.shape[0], generator=sample_gen)
        tot = nb = 0
        for s in range(0, len(perm), BATCH_SIZE):
            rows = real_all[perm[s:s + BATCH_SIZE]]
            if len(rows) < 4:
                continue
            h, r, t = rows[:, 0], rows[:, 1], rows[:, 2]
            h_e = context[h.to(device)]; t_e = context[t.to(device)]
            rd = r.to(device)
            rand_t = torch.randint(0, n_ent, (len(rows),), generator=sample_gen)
            d_r = dreal(h_e, rd, t_e)
            d_f = dreal(h_e, rd, context[rand_t.to(device)])
            # SAME-RELATION mismatched anchors: a rolled mixed-relation anchor
            # is usually type-incompatible, so D_real could win on type alone
            # and never learn ANCHOR-level conditioning (dyn3/4 lesson)
            mm_anchor = torch.tensor([
                by_rel[int(r[i])][rng.randrange(len(by_rel[int(r[i])]))][0]
                for i in range(len(rows))])
            d_m = dreal(context[mm_anchor.to(device)], rd, t_e)
            loss = (F.binary_cross_entropy_with_logits(
                        d_r, torch.full_like(d_r, 1 - LABEL_SMOOTHING))
                    + F.binary_cross_entropy_with_logits(d_f, torch.zeros_like(d_f))
                    + DREAL_MISMATCH * F.binary_cross_entropy_with_logits(
                        d_m, torch.zeros_like(d_m)))
            opt_d.zero_grad(); loss.backward(); opt_d.step()
            tot += loss.item(); nb += 1
        print(f"  dreal-pre {ep_i}/{DREAL_PRETRAIN_EPOCHS} "
              f"loss={tot/max(nb,1):.4f}", flush=True)

    def _save(path):
        """Full candidate_v2 payload -- every snapshot is independently
        loadable by kgsage.inference (same contract as the final save)."""
        torch.save({
            "arch": "candidate_v2",
            "generator_state": G.state_dict(),
            "dmatch_state": dmatch.state_dict(),
            "dreal_state": dreal.state_dict(),
            "context_embeddings": context.cpu(),
            "sketches": (sketches > 0).to(torch.uint8).cpu(),
            "sketch_bits": SKETCH_BITS,
            "pool_masks": pool_masks,
            "cand_k": CAND_K, "dim": DIM, "tau": TAU,
            "ent2id": kg["ent2id"], "rel2id": kg["rel2id"],
            "id2ent": kg["id2ent"], "id2rel": kg["id2rel"],
            "real_triples": list(kg["triple_set_all"]),
            "n_ent": n_ent, "n_rel": n_rel,
            "train_split": "train", "alpha_final": alpha,
            "alpha_target": ALPHA_TARGET, "seed": args.seed,
        }, path)

    alpha = ALPHA_INIT
    err_prev = 0.0
    print("-" * 60, flush=True)
    print(f"Dual-discriminator: {args.epochs} epochs, K={CAND_K}, tau={TAU}, "
          f"alpha0={alpha} target={ALPHA_TARGET}", flush=True)
    for epoch in range(1, args.epochs + 1):
        in_warmup = epoch <= ALPHA_WARMUP_EPOCHS
        if in_warmup:
            alpha = 0.0                       # curriculum: plausible FIRST
        elif alpha == 0.0:
            alpha = ALPHA_INIT           # ramp point: hand over to PID
        perm = torch.randperm(real_all.shape[0], generator=sample_gen)
        ep = defaultdict(float); picks = set(); t0 = time.perf_counter()
        for bi, s in enumerate(range(0, len(perm), BATCH_SIZE)):
            rows = real_all[perm[s:s + BATCH_SIZE]]
            if len(rows) < 4:
                continue
            h, r, t = rows[:, 0], rows[:, 1], rows[:, 2]
            B = len(rows)
            slot = TAIL if bi % 2 == 0 else HEAD
            anchor = h if slot == TAIL else t
            clean = t if slot == TAIL else h

            cand, logq = cs.sample(r.numpy(), slot)
            # ban the true value, known-true fillers (train), self-loop
            banned = torch.zeros(B, cand.shape[1], dtype=torch.bool)
            for i in range(B):
                key = ((int(h[i]), int(r[i])) if slot == TAIL
                       else (int(r[i]), int(t[i])))
                tset = (true_tails if slot == TAIL else true_heads).get(key, set())
                row = cand[i]
                banned[i] = torch.tensor(
                    [(int(x) in tset) or int(x) == int(clean[i])
                     or int(x) == int(anchor[i]) for x in row.tolist()])

            def g_logits():
                lg = G(h.to(device), r.to(device), t.to(device), context,
                       sketches[anchor].to(device), cand.to(device),
                       logq.to(device), slot)
                lg = lg.masked_fill(banned.to(device), float("-inf"))
                # degenerate-row guard (v1 masks had a lift ladder; here some
                # mini pools are singletons -> a row can be ALL-banned, and one
                # -inf row NaN-poisons every weight downstream). Such rows get
                # finite dummy logits and are EXCLUDED from losses and stats.
                valid = torch.isfinite(lg).any(dim=1)
                if not bool(valid.all()):
                    lg = torch.where(valid.unsqueeze(1), lg, torch.zeros_like(lg))
                return lg, valid

            # ---- D_real step ----
            with torch.no_grad():
                lg0, valid = g_logits()
                onehot = gumbel_softmax(lg0, tau=TAU, hard=True)
                fake_emb = torch.einsum("bk,bkd->bd", onehot,
                                        context[cand.to(device)])
            if not bool(valid.any()):
                continue
            v = valid
            keep_emb = context[anchor.to(device)]
            d_fake = dreal(keep_emb[v], r.to(device)[v], fake_emb[v])
            d_real_out = dreal(context[h.to(device)], r.to(device),
                               context[t.to(device)]) if slot == TAIL else \
                         dreal(context[t.to(device)], r.to(device),
                               context[h.to(device)])
            l_d = (F.binary_cross_entropy_with_logits(
                       d_real_out, torch.full_like(d_real_out, 1 - LABEL_SMOOTHING))
                   + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
            if DREAL_MISMATCH > 0:
                # GAN-CLS third class with SAME-RELATION wrong anchors: the
                # real filler presented with another anchor of the same
                # relation -> fake. Type/popularity are symmetric across the
                # real and mismatched classes, so D_real can only win by
                # scoring plausibility CONDITIONAL on the individual anchor.
                filler_emb = context[t.to(device)] if slot == TAIL \
                             else context[h.to(device)]
                # The wrong anchor must come from the ANCHOR slot: heads when
                # the tail is corrupted, tails when the head is corrupted --
                # otherwise head-slot batches pair a tail-type filler with a
                # head-type anchor and D_real can reject by slot type alone.
                mm_slot = 0 if slot == TAIL else 2
                mm_anchor = torch.tensor([
                    by_rel[int(r[i])][rng.randrange(len(by_rel[int(r[i])]))][mm_slot]
                    for i in range(B)])
                d_mm = dreal(context[mm_anchor.to(device)], r.to(device),
                             filler_emb)
                l_d = l_d + DREAL_MISMATCH * F.binary_cross_entropy_with_logits(
                    d_mm, torch.zeros_like(d_mm))
            opt_d.zero_grad(); l_d.backward(); opt_d.step()

            # ---- G step ----
            lg, valid = g_logits()
            v = valid
            onehot = gumbel_softmax(lg, tau=TAU, hard=True)
            fake_emb = torch.einsum("bk,bkd->bd", onehot, context[cand.to(device)])
            g_real = dreal(keep_emb[v], r.to(device)[v], fake_emb[v])
            nid, nmk = nbr_batch(anchor)
            g_match = dmatch(fake_emb[v], context[nid.to(device)][v],
                             nmk.to(device)[v])
            # HINGE: the alienation term is a constraint, not a graded reward.
            # Past the margin there is no payoff for deeper alienation, so the
            # ranking WITHIN the alien set is carried by D_real (the
            # anchor-conditional signal).
            l_g = (-g_real
                   + alpha * torch.relu(g_match - MATCH_MARGIN)).mean()
            opt_g.zero_grad(); l_g.backward(); opt_g.step()

            # ---- measurement (hard picks = what is actually emitted) ----
            with torch.no_grad():
                hard = onehot[v].argmax(dim=1)
                picked = cand[v.cpu()].gather(1, hard.cpu().unsqueeze(1)).squeeze(1)
                sup_rows = support_flags(anchor[v.cpu()], cand[v.cpu()])
                pick_sup = sup_rows.gather(1, hard.cpu().unsqueeze(1)).squeeze(1)
                corr_frac = float(pick_sup.float().mean())
                picks.update(picked.tolist())

            # ---- online D_match hardening: G's picks with ORACLE labels ----
            dm_logit = dmatch(context[picked.to(device)],
                              context[nid.to(device)][v], nmk.to(device)[v])
            l_dm = bce(dm_logit, pick_sup.float().to(device))
            dm_game_opt.zero_grad(); l_dm.backward(); dm_game_opt.step()

            # ---- PID with anti-windup (inactive during the alpha warmup) ----
            if np.isfinite(corr_frac) and not in_warmup:
                err = corr_frac - ALPHA_TARGET
                a_new = (alpha + ALPHA_KP * (err - err_prev)
                         + ALPHA_KI * err)
                alpha = float(min(max(a_new, 0.0), ALPHA_MAX))
                if a_new == alpha:            # integrate only when unsaturated
                    err_prev = err
            with torch.no_grad():
                ep["corr_mass"] += corr_frac
                ep["d_acc_real"] += float((torch.sigmoid(d_real_out) > 0.5).float().mean())
                ep["d_acc_fake"] += float((torch.sigmoid(d_fake) < 0.5).float().mean())
                ep["g_match"] += float(g_match.mean())
                ep["dm_online"] += float(l_dm)
                ep["nb"] += 1

        nb = max(int(ep["nb"]), 1)
        print(f"  epoch {epoch:3d}/{args.epochs}  corr-pick={ep['corr_mass']/nb:.3f} "
              f"alpha={alpha:.2f}  D-acc={ep['d_acc_real']/nb:.2f}/{ep['d_acc_fake']/nb:.2f} "
              f"g_match={ep['g_match']/nb:+.2f}  dm-online={ep['dm_online']/nb:.3f} "
              f"distinct={len(picks)} ({time.perf_counter()-t0:.0f}s)", flush=True)

        # Per-epoch snapshots start with the alpha ramp: warmup epochs are not
        # generator candidates, the pressure epochs around the ramp are.
        if (args.snapshot_every > 0 and epoch > ALPHA_WARMUP_EPOCHS
                and epoch % args.snapshot_every == 0):
            stem = args.out[:-3] if args.out.endswith(".pt") else args.out
            snap_path = f"{stem}.ep{epoch:02d}.pt"
            _save(snap_path)
            print(f"  snapshot -> {snap_path}", flush=True)

    # ---------- checkpoint ----------
    _save(args.out)
    print(f"Saved KGSAGE-2 checkpoint to {args.out}", flush=True)


if __name__ == "__main__":
    main()
