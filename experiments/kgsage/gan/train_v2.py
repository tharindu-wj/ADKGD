"""KGSAGE-2 trainer: LP-free dual-critic adversarial training.

    D_real  (trainable, spectral-normed)  "is this triple plausible?"  G: HIGH
    D_match (pretrained on real-vs-mismatched pairs, then FROZEN)
            "does the candidate fit THIS anchor's neighbourhood?"      G: LOW
    L_G = -D_real + alpha * D_match,  alpha PID-controlled on the MEASURED
    corroborated probability mass (exact graph support, not D_match's opinion).

No pretrained link predictor anywhere. Learned components (sketches, candidate
pools, neighbour lists, both critics) see the TRAIN SPLIT ONLY; the all-splits
falseness guarantee remains where it always was -- the decode-time masks.

Usage (repo root, pytorch env):
  PYTHONPATH=experiments python -m kgsage.gan.train_v2 \
      --data data/FB15K-mini --out experiments/kgsage/outputs/checkpoints/v2_mini.pt \
      [--warmup_epochs 2 --dmatch_epochs 2 --epochs 4 --alpha_target 0.05]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--warmup_batch", type=int, default=4096)
    ap.add_argument("--dmatch_epochs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--cand_k", type=int, default=256)
    ap.add_argument("--sketch_bits", type=int, default=8192)
    ap.add_argument("--n_nbr", type=int, default=32)
    ap.add_argument("--lr_g", type=float, default=1e-4)
    ap.add_argument("--lr_d", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    # PID on alpha: velocity-form PI controller on corroborated-mass error
    ap.add_argument("--alpha_target", type=float, default=0.13,
                    help="must sit ABOVE the structural floor (~12-13% of rows "
                         "have no uncorroborated candidate at all): the first "
                         "FB run used 0.05, alpha railed at max all run, and "
                         "the generator collapsed to a universal-alien global "
                         "ranking (knockout J ~0.9-1.0)")
    ap.add_argument("--alpha_kp", type=float, default=2.0)
    ap.add_argument("--alpha_ki", type=float, default=0.2)
    ap.add_argument("--alpha_init", type=float, default=1.0)
    ap.add_argument("--alpha_max", type=float, default=10.0,
                    help="anti-windup clamp: the FB dynamics smoke measured "
                         "alpha winding to 284 while the error signal was "
                         "flat, collapsing diversity")
    ap.add_argument("--match_margin", type=float, default=0.0,
                    help="the alienation term is a HINGE: alpha*relu(D_match - "
                         "margin). Graded -alpha*D_match kept paying for deeper "
                         "alienation, whose ranking is anchor-independent -- "
                         "dyn2-4 all collapsed to universal-alien rankings")
    ap.add_argument("--dreal_pretrain_epochs", type=int, default=2,
                    help="pretrain D_real (real vs random-fake vs wrong-anchor) "
                         "BEFORE the game. The dyn3 smoke showed an order-of-"
                         "learning failure: the alienation signal is exact from "
                         "epoch 1 while anchor-conditional plausibility develops "
                         "slowly, so G plunges into the universal-alien basin "
                         "before D_real can price it out")
    ap.add_argument("--alpha_warmup_epochs", type=int, default=2,
                    help="hold alpha=0 for the first N game epochs (MolGAN-style "
                         "curriculum: learn PLAUSIBLE first, then ramp the "
                         "contradiction pressure)")
    ap.add_argument("--dreal_mismatch", type=float, default=1.0,
                    help="GAN-CLS matching-aware weight: D_real also sees real "
                         "triples paired with the WRONG anchor, labeled fake, "
                         "making plausibility ANCHOR-CONDITIONAL. 0 = off. "
                         "Added after the universal-alien collapse: without it "
                         "a global ranking can satisfy both critics on average "
                         "without reading the anchor")
    ap.add_argument("--lr_dmatch", type=float, default=1e-4,
                    help="D_match keeps training DURING the game on the "
                         "generator's picks labeled by EXACT graph support -- "
                         "the frozen variant was exploited (48% of emitted "
                         "picks corroborated while D_match scored them alien)")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--num_bases", type=int, default=30)
    ap.add_argument("--encoder_layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--init_context_from", default=None,
                    help="load frozen E' from an existing checkpoint instead of "
                         "running the RGCN warm-up (CPU dynamics tests; the "
                         "vocabularies must match)")
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
        assert payload["n_ent"] == n_ent and payload["dim"] == args.dim, \
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
    encoder = KGSAGEEncoder(n_ent, n_rel, dim=args.dim, num_bases=args.num_bases,
                            num_layers=args.encoder_layers).to(device)
    if _skip_warmup:
        encoder = None
    if not _skip_warmup:
        dec_rel = torch.nn.Parameter(torch.randn(n_rel, args.dim, device=device) * 0.1)
        warm_opt = torch.optim.Adam(list(encoder.parameters()) + [dec_rel], lr=1e-3)
        sample_gen = torch.Generator().manual_seed(args.seed + 1)
        for epoch in range(1, args.warmup_epochs + 1):
            perm = torch.randperm(real_all.shape[0], generator=sample_gen)
            tot = nb = 0
            for s in range(0, len(perm), args.warmup_batch):
                rows = real_all[perm[s:s + args.warmup_batch]].to(device)
                h, r, t = rows[:, 0], rows[:, 1], rows[:, 2]
                t_neg = torch.randint(0, n_ent, (len(rows),), generator=sample_gen).to(device)
                ctx = encoder(edge_index, edge_type)
                pos = (ctx[h] * dec_rel[r] * ctx[t]).sum(-1)
                neg = (ctx[h] * dec_rel[r] * ctx[t_neg]).sum(-1)
                loss = (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                        + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
                warm_opt.zero_grad(); loss.backward(); warm_opt.step()
                tot += loss.item(); nb += 1
            print(f"  warmup {epoch}/{args.warmup_epochs} LP_loss={tot/max(nb,1):.4f}", flush=True)
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
    sketches = build_sketches(train_triples, n_ent, m=args.sketch_bits,
                              seed=args.seed).float()
    cs = CandidateSampler(train_triples, n_ent, n_rel, k=args.cand_k,
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
        ids = torch.zeros(B, args.n_nbr, dtype=torch.long)
        msk = torch.zeros(B, args.n_nbr, dtype=torch.bool)
        for i, a in enumerate(anchors):
            ns = adj.get(int(a), ())
            ns = [n for n in ns if exclude is None or n != int(exclude[i])]
            if not ns:
                continue
            if len(ns) > args.n_nbr:
                ns = rng.sample(ns, args.n_nbr)
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
    dmatch = DMatch(dim=args.dim).to(device)
    dm_opt = torch.optim.AdamW(dmatch.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    print(f"D_match pretraining ({args.dmatch_epochs} epochs)...", flush=True)
    for ep in range(1, args.dmatch_epochs + 1):
        perm = rng.sample(train_triples, len(train_triples))
        tot = nb = 0
        for s in range(0, len(perm), args.batch_size):
            chunk = perm[s:s + args.batch_size]
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
        print(f"  dmatch {ep}/{args.dmatch_epochs} bce={tot/max(nb,1):.4f}", flush=True)
    # NOT frozen: the FB dynamics smoke proved a frozen D_match gets exploited
    # (G converges onto its false negatives). It keeps training during the game
    # on the generator's own picks with ORACLE labels (exact graph support), so
    # every blind spot G finds is corrected on the next batch. The oracle only
    # supplies labels -- D_match remains a learned critic.
    dm_game_opt = torch.optim.AdamW(dmatch.parameters(), lr=args.lr_dmatch)

    # ---------- Phase 2: the dual-critic game ----------
    G = CandidateScoringGenerator(dim=args.dim, sketch_bits=args.sketch_bits,
                                  n_rel=n_rel).to(device)
    dreal = DReal(dim=args.dim, n_rel=n_rel).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(dreal.parameters(), lr=args.lr_d)

    # ---- Phase 1b: D_real pretrain (anchor-conditional plausibility must
    #      exist BEFORE G starts learning, or G falls into the universal-alien
    #      basin the alienation signal digs on epoch 1) ----
    for ep_i in range(1, args.dreal_pretrain_epochs + 1):
        perm = torch.randperm(real_all.shape[0], generator=sample_gen)
        tot = nb = 0
        for s in range(0, len(perm), args.batch_size):
            rows = real_all[perm[s:s + args.batch_size]]
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
                        d_r, torch.full_like(d_r, 1 - args.label_smoothing))
                    + F.binary_cross_entropy_with_logits(d_f, torch.zeros_like(d_f))
                    + args.dreal_mismatch * F.binary_cross_entropy_with_logits(
                        d_m, torch.zeros_like(d_m)))
            opt_d.zero_grad(); loss.backward(); opt_d.step()
            tot += loss.item(); nb += 1
        print(f"  dreal-pre {ep_i}/{args.dreal_pretrain_epochs} "
              f"loss={tot/max(nb,1):.4f}", flush=True)

    alpha = args.alpha_init
    err_prev = 0.0
    print("-" * 60, flush=True)
    print(f"Dual-critic: {args.epochs} epochs, K={args.cand_k}, tau={args.tau}, "
          f"alpha0={alpha} target={args.alpha_target}", flush=True)
    for epoch in range(1, args.epochs + 1):
        in_warmup = epoch <= args.alpha_warmup_epochs
        if in_warmup:
            alpha = 0.0                       # curriculum: plausible FIRST
        elif alpha == 0.0:
            alpha = args.alpha_init           # ramp point: hand over to PID
        perm = torch.randperm(real_all.shape[0], generator=sample_gen)
        ep = defaultdict(float); picks = set(); t0 = time.perf_counter()
        for bi, s in enumerate(range(0, len(perm), args.batch_size)):
            rows = real_all[perm[s:s + args.batch_size]]
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
                onehot = gumbel_softmax(lg0, tau=args.tau, hard=True)
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
                       d_real_out, torch.full_like(d_real_out, 1 - args.label_smoothing))
                   + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
            if args.dreal_mismatch > 0:
                # GAN-CLS third class with SAME-RELATION wrong anchors: the
                # real filler presented with another anchor of the same
                # relation -> fake. Type/popularity are symmetric across the
                # real and mismatched classes, so D_real can only win by
                # scoring plausibility CONDITIONAL on the individual anchor.
                filler_emb = context[t.to(device)] if slot == TAIL \
                             else context[h.to(device)]
                mm_anchor = torch.tensor([
                    by_rel[int(r[i])][rng.randrange(len(by_rel[int(r[i])]))][0]
                    for i in range(B)])
                d_mm = dreal(context[mm_anchor.to(device)], r.to(device),
                             filler_emb)
                l_d = l_d + args.dreal_mismatch * F.binary_cross_entropy_with_logits(
                    d_mm, torch.zeros_like(d_mm))
            opt_d.zero_grad(); l_d.backward(); opt_d.step()

            # ---- G step ----
            lg, valid = g_logits()
            v = valid
            onehot = gumbel_softmax(lg, tau=args.tau, hard=True)
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
                   + alpha * torch.relu(g_match - args.match_margin)).mean()
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
                err = corr_frac - args.alpha_target
                a_new = (alpha + args.alpha_kp * (err - err_prev)
                         + args.alpha_ki * err)
                alpha = float(min(max(a_new, 0.0), args.alpha_max))
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

    # ---------- checkpoint ----------
    torch.save({
        "arch": "candidate_v2",
        "generator_state": G.state_dict(),
        "dmatch_state": dmatch.state_dict(),
        "dreal_state": dreal.state_dict(),
        "context_embeddings": context.cpu(),
        "sketches": (sketches > 0).to(torch.uint8).cpu(),
        "sketch_bits": args.sketch_bits,
        "pool_masks": pool_masks,
        "cand_k": args.cand_k, "dim": args.dim, "tau": args.tau,
        "ent2id": kg["ent2id"], "rel2id": kg["rel2id"],
        "id2ent": kg["id2ent"], "id2rel": kg["id2rel"],
        "real_triples": list(kg["triple_set_all"]),
        "n_ent": n_ent, "n_rel": n_rel,
        "train_split": "train", "alpha_final": alpha,
        "alpha_target": args.alpha_target, "seed": args.seed,
    }, args.out)
    print(f"Saved KGSAGE-2 checkpoint to {args.out}", flush=True)


if __name__ == "__main__":
    main()
