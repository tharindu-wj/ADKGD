"""Train the Phase 2 PAIR-AWARE KGSAGE Generator + Discriminator.

This is the role-swap contradiction generator from the thesis plan
(docs/THESIS_PLAN_pairgan_contradictions.md, "Phase 2", lines 293-366). It is
SEPARATE from gan/train.py (the simple single-slot-corruption GAN used by
ADKGD's `--neg_source gan` baseline) so that baseline stays intact.

Run from the repo root:

  # Local WIRING smoke test on dummy_kg — no encoder checkpoint, no PyG needed.
  # Embeddings are random-initialised so the GAN plumbing can be exercised
  # without first training (and without torch_geometric, which is HPC-only).
  PYTHONPATH=experiments OMP_NUM_THREADS=1 python -m kgsage.cli.train_kgsage_gan \
      --data data/dummy_kg --epochs 5 --device cpu --min_support 1 \
      --out experiments/kgsage/outputs/checkpoints/kgsage_dummy.pt

  # FB15K-237 with the REAL Phase 1 encoder (post-RGCN embeddings); HPC/V100.
  sbatch experiments/kgsage/slurm/train_kgsage_gan_fb15k237.slurm
  #   ... which runs, in essence:
  #   python -m kgsage.cli.train_kgsage_gan --data data/FB15K-237 \
  #       --encoder_ckpt experiments/kgsage/outputs/fb15k237_encoder.pt \
  #       --epochs 50 --out experiments/kgsage/outputs/checkpoints/kgsage_fb15k237.pt

How training works (each epoch), per the plan:
  * POSITIVES are role-swap CONTRADICTION pairs (anchor (h,r,t), partner r')
    drawn from the partner-template providers in partner_templates.py — the
    module the v2-A work built precisely as "the Phase 2 discriminator's
    positive supervision". (NB: the plan's prose at line 345 says "real
    2-cycles"; we use the template providers because they are the documented,
    KG-agnostic positive source and they return the anti-symmetric pairs that
    make a role-swap a contradiction. See the module docstring of
    partner_templates.py.)
  * Discriminator step: D learns template pairs -> 1, generator pairs -> 0.
  * Generator step: adversarial (fool D) + reconstruction (cross-entropy to the
    template partner r') + optional entropy regulariser (Test 2.2 mode-collapse
    remedy, off by default).

After training it runs the plan's acceptance probes:
  * Test 2.2 — partner-relation entropy (target > 1.5 nats).
  * Test 2.3 — syntactic validity of sampled partners (target >= 95%).
"""
import argparse
import math
import os
import random
import time
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn.functional as F

from kgsage.data.loaders import load_kg
from kgsage.gan.models import (
    KGSAGEGenerator,
    KGSAGEDiscriminator,
    gumbel_softmax,
    load_encoder_embeddings,
)
from kgsage.gan.partner_templates import (
    observed_2cycle_templates,
    mine_partner_templates,
)
from kgsage.data.audit_dataset import MIN_SUPPORT


# ---------------------------------------------------------------------------
# Positives: role-swap contradiction templates -> (h, r, t, r') pairs.
# ---------------------------------------------------------------------------
def build_positive_pairs(kg, templates, rng):
    """Expand templates into concrete (h, r, t, r_partner) supervised positives.

    templates : { anchor_rel : [(partner_rel, confidence), ...] }
                from observed_2cycle_templates / mine_partner_templates.

    For every training triple (h, r, t) whose relation r has templates, we emit
    one positive (h, r, t, r') where r' is sampled from r's partner list with
    probability proportional to confidence. The role-swap (t, r', h) is implied.
    """
    by_rel = defaultdict(list)
    for (h, r, t) in kg["triples_train"]:
        by_rel[r].append((h, r, t))

    pairs = []
    for r, parts in templates.items():
        partner_rels = [rp for rp, _ in parts]
        weights = [max(c, 1e-6) for _, c in parts]
        total = sum(weights)
        probs = [w / total for w in weights]
        for (h, rr, t) in by_rel.get(r, []):
            rp = partner_rels[_weighted_index(probs, rng)]
            pairs.append((h, rr, t, rp))
    return pairs


def _weighted_index(probs, rng):
    """Sample an index in [0, len(probs)) by the given probability list."""
    u = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if u <= cum:
            return i
    return len(probs) - 1


def prepare_tensors(pairs, device):
    """Pack (h, r, t, r') pairs into anchors (N,3) and partners (N,) tensors."""
    anchors = torch.tensor([(h, r, t) for (h, r, t, _) in pairs],
                           dtype=torch.long, device=device)
    partners = torch.tensor([rp for (_, _, _, rp) in pairs],
                            dtype=torch.long, device=device)
    return anchors, partners


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------
def train_one_epoch(G, D, opt_G, opt_D, anchors, partners, batch_size, device,
                    recon_weight=1.0, entropy_weight=0.0, tau=0.5):
    """One pass over the positive pairs. Returns (avg_D_loss, avg_G_loss)."""
    n_total = anchors.size(0)
    perm = torch.randperm(n_total, device=device)
    anchors, partners = anchors[perm], partners[perm]

    rel_table = D.rel_emb.weight   # frozen (n_rel, dim) — soft partner lookup

    total_d, total_g, n_batches = 0.0, 0.0, 0
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        if end - start < 2:
            continue
        a = anchors[start:end]
        h, r, t = a[:, 0], a[:, 1], a[:, 2]
        r_pos = partners[start:end]

        # ---------------- Discriminator step ----------------
        with torch.no_grad():
            soft = gumbel_softmax(G(h, r, t), tau=tau)   # (n, n_rel)
            partner_emb_fake = soft @ rel_table          # (n, dim), detached
        opt_D.zero_grad()
        score_pos = D(h, r, t, r_pos)                    # template contradiction -> 1
        score_neg = D(h, r, t, partner_emb_fake)         # generator sample      -> 0
        loss_d = (
            F.binary_cross_entropy_with_logits(score_pos, torch.ones_like(score_pos))
            + F.binary_cross_entropy_with_logits(score_neg, torch.zeros_like(score_neg))
        )
        loss_d.backward()
        opt_D.step()

        # ---------------- Generator step ----------------
        opt_G.zero_grad()
        rel_logits = G(h, r, t)
        soft = gumbel_softmax(rel_logits, tau=tau)
        partner_emb = soft @ rel_table                   # differentiable -> grad to G
        score_fake_for_g = D(h, r, t, partner_emb)
        loss_adv = F.binary_cross_entropy_with_logits(
            score_fake_for_g, torch.ones_like(score_fake_for_g),
        )
        loss_recon = F.cross_entropy(rel_logits, r_pos)  # match the template partner

        loss_g = loss_adv + recon_weight * loss_recon
        if entropy_weight > 0.0:
            # Encourage a spread-out partner distribution (Test 2.2 remedy).
            probs = torch.softmax(rel_logits, dim=-1).mean(0)
            entropy = -(probs * (probs + 1e-12).log()).sum()
            loss_g = loss_g - entropy_weight * entropy
        loss_g.backward()
        opt_G.step()

        total_d += loss_d.item()
        total_g += loss_g.item()
        n_batches += 1

    return total_d / max(n_batches, 1), total_g / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Post-training acceptance probes (plan Tests 2.2 + 2.3).
# ---------------------------------------------------------------------------
def evaluate_generation(G, kg, real_set, n_samples, tau, rng, device):
    """Sample partners for up to n_samples anchors; return (metrics, examples).

    metrics:
      entropy_nats        : Shannon entropy of the partner-relation distribution
      validity_frac       : fraction of partners with (t,r',h) absent from graph,
                            not a self-loop  (Test 2.3 — relaxed, see note below)
      validity_diff_frac  : same but ALSO requiring r' != r (the plan's literal
                            Test 2.3 wording — reported for completeness)
      self_swap_frac      : fraction where r' == r (legitimate self-asymmetric
                            contradictions like parentOf -> reversed parentOf)

    NOTE on Test 2.3: the plan lists "partner relation differs from anchor
    relation" as a validity criterion. We do NOT gate on it, because the most
    common contradiction is self-asymmetric (r' == r), e.g. (Alice, parentOf,
    Carol) -> (Carol, parentOf, Alice). The "absent from real graph" check is
    what makes a partner a contradiction and auto-rejects symmetric relations.
    """
    triples = kg["triples_train"]
    idx = list(range(len(triples)))
    rng.shuffle(idx)
    idx = idx[:n_samples]

    G.eval()
    rel_counts = defaultdict(int)
    valid, valid_diff, self_swap, considered = 0, 0, 0, 0
    examples = []
    with torch.no_grad():
        for j in idx:
            h, r, t = triples[j]
            if h == t:
                continue   # a self-loop anchor has no meaningful role-swap
            considered += 1
            logits = G(torch.tensor([h], device=device),
                       torch.tensor([r], device=device),
                       torch.tensor([t], device=device))[0]
            u = torch.rand_like(logits).clamp_(1e-10, 1.0 - 1e-10)
            gumbel = -torch.log(-torch.log(u))
            r_prime = int((logits + tau * gumbel).argmax().item())
            rel_counts[r_prime] += 1
            partner = (t, r_prime, h)
            is_valid = partner not in real_set            # absent => contradiction
            if is_valid:
                valid += 1
                if r_prime != r:
                    valid_diff += 1
            if r_prime == r:
                self_swap += 1
            if len(examples) < 10:
                examples.append((h, r, t, r_prime, is_valid))

    total = sum(rel_counts.values())
    if total > 0:
        entropy = -sum((c / total) * math.log(c / total) for c in rel_counts.values())
    else:
        entropy = 0.0
    G.train()
    metrics = {
        "entropy_nats": entropy,
        "validity_frac": valid / max(considered, 1),
        "validity_diff_frac": valid_diff / max(considered, 1),
        "self_swap_frac": self_swap / max(considered, 1),
        "n_considered": considered,
        "distinct_partner_rels": len(rel_counts),
    }
    return metrics, examples


# ---------------------------------------------------------------------------
# Checkpoint.
# ---------------------------------------------------------------------------
def save_checkpoint(G, kg, save_path):
    """Bundle everything inference.load_kgsage_checkpoint needs into one .pt."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "kind": "kgsage_pairgan",          # tells inference which loader to use
        "generator_state": G.state_dict(),
        "ent2id": kg["ent2id"],
        "rel2id": kg["rel2id"],
        "id2ent": kg["id2ent"],
        "id2rel": kg["id2rel"],
        "real_triples": list(kg["triple_set_all"]),
        "n_ent": G.n_ent,
        "n_rel": G.n_rel,
        "dim": G.dim,
        "hidden": G.hidden,
    }, save_path)


def main():
    ap = argparse.ArgumentParser(description="Train the Phase 2 KGSAGE pair-aware GAN.")
    ap.add_argument("--data", required=True, help="Dataset directory")
    ap.add_argument("--out", required=True, help="Output checkpoint path (.pt)")
    ap.add_argument("--encoder_ckpt", default=None,
                    help="Phase 1 encoder .pt. If omitted, embeddings are "
                         "RANDOM-initialised (wiring smoke test; no PyG needed).")
    ap.add_argument("--template", choices=["rule", "observed"], default="rule",
                    help="Positive source: 'rule' = mine_partner_templates (v2-A, "
                         "KG-agnostic); 'observed' = observed_2cycle_templates (v1).")
    ap.add_argument("--min_support", type=int, default=MIN_SUPPORT,
                    help="Relation support floor for templates (use 1 for dummy_kg).")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=200,
                    help="Embedding dim for the RANDOM-init fallback only; "
                         "ignored when --encoder_ckpt is given (dim comes from it).")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--recon_weight", type=float, default=1.0)
    ap.add_argument("--entropy_weight", type=float, default=0.0,
                    help="Enable (e.g. 0.1) only if Test 2.2 reports mode collapse.")
    ap.add_argument("--tau", type=float, default=0.5, help="Gumbel-softmax temperature.")
    ap.add_argument("--eval_samples", type=int, default=1000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)

    print(f"Loading KG from {args.data} ...", flush=True)
    kg = load_kg(args.data)
    print(f"  entities = {kg['n_ent']:,}  relations = {kg['n_rel']:,}  "
          f"train triples = {len(kg['triples_train']):,}", flush=True)

    # --- Embeddings: post-RGCN from the encoder, or random-init smoke fallback ---
    if args.encoder_ckpt:
        print(f"Loading post-RGCN encoder embeddings from {args.encoder_ckpt} ...", flush=True)
        ent_emb, rel_emb, enc_ent2id, enc_rel2id = load_encoder_embeddings(
            args.encoder_ckpt, kg, device)
        if len(enc_ent2id) != kg["n_ent"] or len(enc_rel2id) != kg["n_rel"]:
            raise SystemExit(
                f"Vocab mismatch: encoder has {len(enc_ent2id)} ent / {len(enc_rel2id)} rel "
                f"but data has {kg['n_ent']} / {kg['n_rel']}. Re-run the encoder on --data.")
        dim = ent_emb.size(1)
        print(f"  loaded ent_emb {tuple(ent_emb.shape)}  rel_emb {tuple(rel_emb.shape)}", flush=True)
    else:
        dim = args.dim
        print(f"No --encoder_ckpt: RANDOM-initialising embeddings (dim={dim}). "
              f"Wiring smoke only — not a trained model.", flush=True)
        ent_emb = torch.randn(kg["n_ent"], dim, device=device) * 0.1
        rel_emb = torch.randn(kg["n_rel"], dim, device=device) * 0.1

    # --- Positives: role-swap contradiction templates ---
    provider = mine_partner_templates if args.template == "rule" else observed_2cycle_templates
    print(f"Building positives via {provider.__name__} (min_support={args.min_support}) ...", flush=True)
    templates = provider(kg, min_support=args.min_support)
    pairs = build_positive_pairs(kg, templates, rng)
    print(f"  {len(templates):,} anchor relations templated -> {len(pairs):,} positive pairs", flush=True)
    if not pairs:
        raise SystemExit(
            "No positive pairs. Lower --min_support (dummy_kg needs --min_support 1) "
            "or check that the dataset has anti-symmetric structure (Test 1.3).")
    anchors, partners = prepare_tensors(pairs, device)

    G = KGSAGEGenerator(ent_emb, rel_emb, hidden=args.hidden).to(device)
    D = KGSAGEDiscriminator(ent_emb, rel_emb, hidden=args.hidden).to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr * 0.25, betas=(0.5, 0.999))

    print(f"Training: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"recon_weight={args.recon_weight}, entropy_weight={args.entropy_weight}", flush=True)
    print("-" * 64, flush=True)
    start_perf = time.perf_counter()
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)

    for epoch in range(1, args.epochs + 1):
        d_loss, g_loss = train_one_epoch(
            G, D, opt_G, opt_D, anchors, partners, args.batch_size, device,
            recon_weight=args.recon_weight, entropy_weight=args.entropy_weight, tau=args.tau,
        )
        log_every = max(1, args.epochs // 20)
        if epoch <= 5 or epoch % log_every == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:4d}/{args.epochs}  D_loss={d_loss:.4f}  G_loss={g_loss:.4f}", flush=True)

    total_seconds = time.perf_counter() - start_perf
    print(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}  ({total_seconds:.1f}s, "
          f"{total_seconds / args.epochs:.2f}s/epoch)", flush=True)
    print("-" * 64, flush=True)

    # --- Acceptance probes (plan Tests 2.2 + 2.3) ---
    metrics, examples = evaluate_generation(
        G, kg, kg["triple_set_all"], args.eval_samples, args.tau, rng, device)
    print("Generation probes (plan Tests 2.2 / 2.3):", flush=True)
    print(f"  Test 2.2  partner-relation entropy : {metrics['entropy_nats']:.3f} nats "
          f"(target > 1.5; max log({kg['n_rel']})={math.log(kg['n_rel']):.2f})  "
          f"{'PASS' if metrics['entropy_nats'] > 1.5 else 'WARN'}", flush=True)
    print(f"  Test 2.3  syntactic validity       : {metrics['validity_frac']:.1%} "
          f"(target >= 95%)  {'PASS' if metrics['validity_frac'] >= 0.95 else 'WARN'}", flush=True)
    print(f"            distinct partner relations: {metrics['distinct_partner_rels']}", flush=True)
    print(f"            self-swap (r'==r) share   : {metrics['self_swap_frac']:.1%} "
          f"(legitimate self-asymmetric contradictions)", flush=True)
    print(f"            validity if r'!=r required: {metrics['validity_diff_frac']:.1%} "
          f"(the plan's literal 2.3 wording)", flush=True)
    print("  sample partners (anchor -> role-swap):", flush=True)
    id2ent, id2rel = kg["id2ent"], kg["id2rel"]
    for (h, r, t, rp, ok) in examples[:10]:
        print(f"    ({id2ent[h]}, {id2rel[r]}, {id2ent[t]})  ->  "
              f"({id2ent[t]}, {id2rel[rp]}, {id2ent[h]})  "
              f"{'[contradiction]' if ok else '[in-graph: rejected at inference]'}", flush=True)

    save_checkpoint(G, kg, args.out)
    print(f"\nSaved checkpoint to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
