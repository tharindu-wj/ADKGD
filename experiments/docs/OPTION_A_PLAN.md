# Option A — LP-in-the-loop GAN (Project Milestone 2, builds on OPTION_B_PLAN.md)

> ⚠️ **HISTORICAL DOCUMENT — path not taken.** The "LP-in-the-loop" design (a
> frozen link predictor supplying the plausibility signal) was superseded: the
> shipped architecture is **LP-free**, with a learned `PlausibilityDiscriminator`
> (D_real) in place of the frozen predictor. Kept for provenance. Current names:
> `KGSAGE_glossary.md`.

Feasibility workup 2026-07-03 (3-agent: equilibrium analysis, prior-art, code/compute). Verdict:
**feasible; primary design = A-ii (frozen ComplEx + trainable contextual residual head);
fallback/baseline arm = A-iv (anchored fine-tuned ComplEx, honestly framed as the KBGAN/IGAN control).**

> **Vocabulary.** This plan predates the dual-discriminator architecture: its single judge "D" is
> the design that became **D_real**, the plausibility discriminator ("could this triple be real?").
> The neighbourhood discriminator **D_match** does not exist yet at this point in the history.
> "Milestone 2" here is a project milestone, not the pipeline's Phase 2.

## 1. Variant scoreboard (1–5, 5 best: achieves-goal / stability / effort / GAN-load-bearing)

| Variant | D | Scores | Verdict |
|---|---|---|---|
| A-i | fully fine-tuned ComplEx + frozen fence | 2/1/3/2 | dominated: no fixed point (D depresses whatever G emits → cycling); ComplEx has no context mechanism, so fine-tuning cannot encode neighbourhood contradiction — only catastrophic forgetting of the MRR-0.348/0.475 calibration; the anchor that stabilises it pins D back to the frozen scorer |
| **A-ii** | **frozen ComplEx score + trainable residual head f_θ over frozen-warmup RGCN features** | **4/4/3/4** | **primary: f_θ's optimum is by construction the contextual signal ComplEx cannot explain; G's optimum = most neighbourhood-consistent filler that is still false (= Alice→NewZealand); no forgetting, no collusion; worst case degrades gracefully into Option B's band sampler** |
| A-iii | repo pair-MLP over frozen features, ComplEx fence only | 2/2/4/2 | keeps the pair-shortcut + collusion surface; dominated by A-ii |
| A-iv | no RGCN, simple G vs fine-tuned ComplEx | 2/2/5/1 | literally IGAN (AAAI'18)'s GAN-pretrain setting modulo Gumbel-vs-REINFORCE — keep as the minimal control arm, not a contribution |

## 2. A-ii exact specification

Notation: s_f = frozen LibKGE ComplEx (post MRR gate); s_z = per-relation z-scored s_f within the type
pool (raw ComplEx scores are uncalibrated across relations); E' = RGCN context table, **frozen after
warmup**; f_θ = 2-layer MLP(128) over the CANDIDATE triple only [E'[h'] | ρ_r' | E'[t']] (no anchor
input — kills the relation-match pair shortcut by construction), output β·tanh (β ≈ 1σ of within-band
spread — bounds the residual so total score can never leave the s_f band). D_real(x) = s_z(x) + f_θ(x).

- **Step 0 — encoder warmup (~50 lines):** RGCN + DistMult decoder, BCE vs uniform corruptions,
  ~10 epochs on the train graph → `encoder.eval()` + freeze. Encoder LEAVES G's optimizer
  (train.py:319-322) — mandatory, else representation collusion returns.
- **Step 1 — G warm start:** 2 epochs CE toward Option B band-sampler draws, then weight 0 forever.
  **No recon term in the adversarial phase** (the 75-86× distillation pathology is gone by deletion).
- **Step 2 — adversarial loop (1:1 steps):**
  - D_real step (θ only): positives = train triples; fakes = 50% G hard samples + 25% band-sampler
    negatives + 25% type-valid random + 10% replay buffer (damps whack-a-mole cycling; the non-G
    fakes keep f_θ grounded in general contextual realness). L_D = BCE + λ_res·mean(f_θ²), Adam 3e-4.
  - G step: logits masked to the allowed pool (type pool − all-splits known-true fillers − self;
    training-time masks are mandatory); **straight-through Gumbel** (hard forward / soft backward —
    replaces the τ=1.0 soft-mixture artifact); L_G = −[s_z_soft + f_soft] + λ_fence·max(0, s_f(g) −
    (s_f(true) − m)) − λ_H·H(masked logits), m = 0.5σ_r, λ_H = 0.01, Adam 1e-4 β(0.5, 0.999).
- **Fence, two layers, both from the untouched s_f:** in-loop soft hinge; generation-time hard
  filter (reject s_f(g) > s_f(true), counted into the FN proxy — Option B gate #5 becomes a
  per-epoch alarm). The fence bounds SCORE, not truth — the FN proxy is the honesty instrument.
- **5 per-epoch diagnostics:** truth-drift (fence-hit %, LP-top-1 rate; alarm >25%); residual health
  (f_θ-alone AUC on held-out real-vs-band pairs, want 0.55–0.85; ~0.5 = dead channel, ~1.0 =
  shortcut; |corr(f_θ, s_z)| < 0.3); hardness band (median s_f-rank of emitted negatives ≈ 2–20);
  diversity (distinct fillers/relation, entropy, z-sensitivity); game balance (`D-acc=` on the G-slice
  0.6–0.8, alarm >0.95 for 3 epochs — `D-acc=` is a frozen log token for D_real's accuracy).

Escalation ladder if the residual is dead: more warmup epochs → dim 128 → let f_θ's optimizer (never
G's) fine-tune the last RGCN layer → report honestly as fenced-ComplEx sampling.

## 3. Effort, compute, reuse (on top of Option B)

- **New/changed code:** ~+880 / ~380 modified / −185 deleted. New: `gan/complex_d.py` (~130, trainable
  + frozen twins, soft scoring), `gan/residual_d.py` (~70), `gan/masks.py` (~70), `cli/inspect_gan_lp.py`
  (~90). Reworked: `train_one_epoch` (+170/~120/−45). Reused unchanged: the generator, the encoder,
  bridge, dataset.py gan branches, run_experiment (Option A only produces a better `.pt`).
  ContradictionTargetSampler leaves the training path entirely (kept as legacy ablation).
- **Effort:** shared substrate 1.5d → A-ii 3–3.5d → A-i/A-iv ablation arm +1–1.5d. **Programme total
  ~4.5–5.5 engineering days + cluster time.**
- **Compute (V100 16G, FB15K-237):** Option A adds <10% per-epoch overhead (~+5–10 ms/batch); ~1.6–3.2
  min/epoch during warmup, **~10–20 s/epoch after the encoder freezes** (no full-graph RGCN in the
  loop). Peak memory ~2–3 GB of 16 GB. 50 epochs fit existing SLURM windows comfortably.

## 4. Build order with gates

A0 Option B gates pass (MRR gate; FN-drop <15%) → A1 `complex_d.py` (gate: soft score of an exact
one-hot == hard score to 1e-4 on 10k triples) → A2 masks (gate: 10k draws, zero pool/known-true/self
violations; WN18RR empty-pool rows handled) → A3 ST-Gumbel behind a flag (gate: flag-off regression
on dummy_kg unchanged) → A4 A-ii loop (gates: D-acc in range, residual AUC 0.55–0.85, drift alarms
quiet on FB15K-237) → A5 A-i/A-iv ablation arm (extra gate: fine-tuned MRR ≥80% of frozen) → A6
end-to-end through the untouched bridge + Option-B experiment matrix + `sampler_direct` control.

## 5. Prior art and the honest claim

Taken (cite-and-differ): pretrained-KGE-as-adversarially-fine-tuned-D = KBGAN (NAACL'18) and
literally IGAN (AAAI'18) — hence A-iv is a control arm; frozen-scorer + trainable re-ranker as an
architecture = CascadER (AKBC'22) / RADD-style retrieve-and-rerank — bounds A-ii's architectural
novelty to the *adversarial role*. No KG precedent found for: a frozen pretrained twin as truth-fence/
anchor (RLHF-style, Ziegler et al. 2019 pattern), a relativistic frozen-score+residual plausibility discriminator
on KGs, or ANY GAN/diffusion negative sampler whose consumer is a downstream anomaly DETECTOR.
NSCaching's "GANs aren't worth it" critique is answered structurally by the built-in `sampler_direct`
control; "Hard or False" (TKDE'25) confirms truth-drift as the central risk — keep the FN proxy
reported. Claimable in one sentence: *a published pretrained link predictor used simultaneously as
(the backbone of) the adversarial plausibility discriminator and as a frozen truth-fence, with straight-through
Gumbel replacing the KBGAN line's REINFORCE, manufacturing label-guaranteed plausible-but-false,
neighbourhood-conditioned anomalies for training and evaluating a KG anomaly detector.*

## 6. Top failure modes (severity → mitigation)

1. **Truth-drift amplification [critical]** — f_θ actively rewards unobserved-true facts (maximally
   real-looking contexts). Fence in-loop + hard filter + FN-proxy per-epoch alarm + human spot audit.
2. **Whack-a-mole cycling [major]** — bounded residual (β·tanh) makes worst case a graceful collapse
   onto the band sampler; replay buffer + mixed fakes damp it.
3. **Dead contextual channel [major]** — residual-AUC tripwire (0.55 threshold) + escalation ladder;
   if dead, report honestly (the claim then reduces to fenced-LP sampling).
4. **Gumbel soft/hard phantom objective [major]** — ST-Gumbel mandatory; log hard-minus-soft score gap.
5. **Mode/z collapse [major]** — entropy bonus + distinct-filler and z-sensitivity alarms.
