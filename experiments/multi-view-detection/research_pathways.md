# Multi-Viewpoint Anomaly Detection — Research Pathway Map (v2)

*Synthesized 15 Jul 2026 from `project_scope/` (proposal pptx + STEM9003 lit review) and `research_directions.docx`; revised 16 Jul 2026. Standing decisions: (1) Phase 2 is explored **freely, not anchored on KGSAGE** — alignment is a later, optional decision (§6); (2) the **evolution of agentic AI** (ML → Gen AI → LLM → agentic) is a first-class research thread (§3.S). Claims marked ⚠ need adversarial literature verification before entering the thesis.*

---

## 1. The direction as stated (baseline to extend)

- **Multi-viewpoint** = multiple contextual/behavioural perspectives *of the same entity* (not modalities, not camera views). The hard part is preserving viewpoint meaning while reasoning *across* viewpoints — anomaly status can flip when another viewpoint is considered.
- **Two failure modes of current practice**: (1) early fusion into one feature space dilutes view-specific anomaly signals; (2) per-view models detect independently with no cross-view context.
- **Proposal**: heterogeneous graph/KG representation + viewpoint-specialised agents with shared-memory coordination; evaluated on synthetically generated anomalies.
- **Four gaps claimed**: no graph-based multi-view representation; no view-specialised reasoning; no coordinated cross-view sharing at detection time; no labelled multi-view anomaly data (→ synthesis, but never with cross-view relationship patterns).

## 2. Calibration notes (fix before deep research)

1. **Terminology collision.** "Multi-view AD" names ≥3 distinct literatures: (a) camera-viewpoint vision AD — the two works cited in the lit review (Liu et al. epipolar; Mao et al. AAAI'25) are this kind, so they support the narrative only weakly; (b) **multi-view outlier detection (MVOD)** — the HOAD lineage, latent-representation and deep MVOD methods — this is the *correct* prior-art family for "same entity, multiple perspectives" and must be surveyed + baselined; (c) multi-view learning generally. Adopting the term **multi-viewpoint** consistently (as the proposal does) and explicitly contrasting with (a)/(b) is both a defence and a positioning asset.
2. **⚠ Multiplex/multilayer graph AD exists.** Gap 1 ("representing multiple aspects of one entity in a graph is unaddressed") is overstated — anomaly detection on multiplex/multilayer networks and heterogeneous-graph AD is an active area. The defensible gap is narrower: *no framework combines view-preserving graph representation with view-specialised coordinated reasoning*.
3. **⚠ Agentic AD is moving fast.** The lit review found 2 systems (maritime; NASA DSN). By mid-2026 expect more (log/SOC agents, time-series agent frameworks, LLM-agent AD surveys). The deep-research sweep must re-establish the state of the art before the gap statement is finalised.
4. **Evaluation circularity (general principle, learned the hard way in Phase 1).** If the same generative process produces the test anomalies *and* the detector is designed for exactly those anomalies, results are circular. Whatever generator Phase 2 uses, keep an off-family test arm (anomalies from a different generator family) and at least one real-error/naturally-labelled arm.

## 3. Pathway map (S + A–E, composable)

### S. The spine — evolution of agentic AI for anomaly detection (ML → Gen AI → LLM → Agentic)
A structured account of how detection capability has evolved, one rung at a time — this is the `research_directions.docx` second bucket made rigorous, and it doubles as the thesis background chapter and possibly a standalone survey/taxonomy deliverable. Frame each rung by **what new capability it added and what limitation it left**:

| Rung | Representative methods | Capability added | Limitation left |
|---|---|---|---|
| Classical ML | OC-SVM, isolation forest, LOF, statistical/clustering | formal outlier scoring on engineered features | manual features; weak on high-dim, relational, semantic data |
| Deep learning | autoencoders, deep SVDD, GNN-based GAD | learned representations; relational structure (graphs) | needs lots of data; no semantics; reconstruction can hide anomalies |
| Generative AI | GAN family (AnoGAN → f-AnoGAN → BiGAN → GANomaly), diffusion AD; synthetic anomaly generation | modelling normality; generating rare/unseen anomalies | no reasoning; unstable training; still single-view |
| LLM-based | prompting/fine-tuned detectors for logs, time series, KG errors; LLM validation of detections | semantic/contextual reasoning; zero-shot detection; explanation in language | single-pass; no persistence, tools, or collaboration; hallucination |
| Agentic AI | multi-agent systems w/ tools, shared memory, planning (maritime, NASA DSN, SOC/log agents) | specialisation + coordination + tool grounding + autonomy | error cascades; cost; almost no principled evaluation ⚠ |

Research outputs from S: (i) a capability/limitation taxonomy; (ii) the argument that **multi-viewpoint AD is precisely the problem class that forces the agentic rung** (specialisation per view + coordination across views); (iii) the criteria grid used to position C/D against prior art.

### A. The testbed — cross-view anomaly taxonomy + controllable synthetic generation
Method-agnostic (GAN, LLM-based, rule/constraint-based, or hybrid — choose on merit during deep research). The intellectual core is the **taxonomy of view-interaction anomalies**, ordered by detection difficulty:
- **A1 single-view anomaly** — detectable inside one view (sanity tier; any baseline catches it).
- **A2 cross-view contradiction** — *every view is marginally normal, the combination is impossible* (e.g., enrolment view says full-time student; employment view says 80 h/week FIFO worker; location views disagree). Detectable **only** by cross-view reasoning.
- **A3 explained-away normal** — anomalous in one view, legitimate given another (medical-leave view explains grade collapse). **False-positive traps**: no coordination ⇒ FP storm.
- **A4 view-integrity anomalies** — missing/stale/conflicting-provenance views.

**Standalone contribution: first controllable benchmark for multi-viewpoint anomaly detection** ⚠(verify none exists). De-risks the thesis — a benchmark survives even if the detector underwhelms (proposal risk #4 mitigation). Answers RQ2.

### B. The representation — multiplex KG schema for viewpoints
Entity spine + one layer per viewpoint (multiplex/heterogeneous KG with inter-layer anchor edges). Make the schema question *empirical*, not engineering: which representation — (i) fused hetero-KG, (ii) per-view subgraphs, (iii) multiplex layers with anchors — best **preserves viewpoint meaning**? Measurable: per-view detector performance, view-identifiability probes on embeddings, signal-dilution tests (does a view's anomaly signal survive fusion?). Answers RQ3; this is proposal Phase-2 made publishable.

### C. The detector — viewpoint-specialist agents with coordinated reasoning
Per-view agents wrap *tools* (per-view detectors of any rung from S: statistical checks, embedding scorers, GNNs — agents reason over tool outputs rather than raw data, controlling hallucination). Coordination via shared blackboard memory + arbiter. Detection protocol = **flag-then-explain**: a per-view flag is a hypothesis; the arbiter must either *explain it away* with evidence from other views (kills A3 FPs) or *corroborate* it; a **cross-examination round** hunts A2 contradictions among marginally-normal views. Novelty claim: not "we used agents" but *coordination recovers interaction anomalies that fusion dilutes and independent models miss*. Answers RQ4.

**Mandatory baselines**: early fusion (one model, merged graph), late fusion (score max/mean), classic MVOD (HOAD-lineage + a deep MVOD), heterogeneous GNN (RGCN/HGT) on the fused graph, single-LLM-one-prompt (no agents). One baseline per rung of S makes the evolution thread empirical, not just narrative.

### D. The science — coordination-mechanism ablation study
The falsifiable core that answers "isn't agentic AI just prompt engineering?": vary **coordination topology** — (1) independent + vote (= late fusion with LLMs), (2) shared memory/blackboard, (3) pairwise debate/cross-examination, (4) hierarchical arbiter — and measure detection lift **per anomaly class**, error-cascade rate (how often one agent's wrong claim propagates — the lit review's own risk), and token/latency cost frontier. Predicted result that would *make* the thesis: lift concentrated on A2/A3, negligible on A1. Directly addresses proposal risks #1/#2/#4. Together with C's rung-per-baseline design, D turns the S-spine into an **empirical ladder**: each rung's best method evaluated on the same A1–A4 benchmark.

### E. The trust layer — grounded, explainable verdicts (stretch)
Every agent claim must cite KG evidence (subgraph provenance); arbiter validates evidence chains; final output = anomaly verdict + evidence subgraph + natural-language rationale. Anti-cascade guardrail *and* explainability contribution; measure explanation faithfulness. In-house precedent: ETCOD's LLM-driven validation (Senaratne group) — natural lineage story.

## 4. Recommended composition (thesis narrative, freestanding)

> **Trace how each generation of AI added a detection capability (S); show multi-viewpoint anomalies are the class that defeats every rung below agentic (A on B); build the coordinated detector (C); prove coordination is the causal ingredient with rung-by-rung and topology ablations (D); ground every verdict in evidence (E).**

- Maps 1:1 onto the proposal timeline: P1 prototype→A, P2 schema→B, P3 agents→C(+D), P4 evaluation→D(+E); S runs throughout as lit-review → background chapter → positioning grid.
- Minimal viable thesis = B(iii) fixed + A(A1–A3) with any one generation method + C with topologies (1) vs (2) + circularity guard. Everything else is upside.
- Domain instantiation (decide in deep research): **one naturally multi-view domain** as primary (smart-home IoT à la SeIoT, or synthetic student-performance KG per the proposal's running example) + optionally a KG benchmark for comparability. Avoid maritime/DSN (taken by the two cited systems).

## 5. Deep-research agenda (next step)

**Method — survey-first (standing rule):** for every sweep below, first find a survey/review paper for the path and mine its taxonomy + reference list to reach the primary papers; issue broad direct searches only where surveys are thin, missing, or outdated (expected for agentic AD ⚠). This gives structured coverage and conserves research-connector limits. Anchor surveys already in hand: Chandola 2009 (AD foundations), Pang 2022 (deep AD), Qiao 2025 (deep graph AD), Chakraborty 2024 (GANs), Sabuhi 2021 (GAN-AD), Sapkota 2026 (agentic-AI taxonomy). Each sweep's write-up should name its anchor survey(s) or declare the path survey-thin.

1. **Evolution sweep (feeds S)** — anomaly detection by generation: classical ML → deep/graph → generative (GAN/diffusion) → LLM-based → agentic; for each: representative methods, capability added, limitation left, surveys available. Target output: the S-table fully cited.
2. **Agentic/LLM AD sweep 2024–2026** — log, time-series, SOC, KG agents; coordination mechanisms used; any flag-then-explain precedents ⚠; how systems were evaluated (the evaluation-void claim).
3. **MVOD sweep** — HOAD lineage → latent/deep MVOD (2010–2026): what exactly do they detect (class-A1/A2?) and on what data; extract baseline set.
4. **Multiplex/multilayer & heterogeneous graph AD sweep** — how much of gap 1 survives ⚠.
5. **Cross-view/contextual anomaly benchmarks** — does any controllable multi-view anomaly benchmark exist ⚠ (if none: pathway A is the first).
6. **Adversarially verify the 4 gap statements** against 1–5; rewrite gaps in the narrower defensible form.

## 6. KGSAGE alignment — deferred decision (do not design around it)

Phase 2 stands alone. Revisit only after the Phase-2 shape is set, choosing zero or more of these optional integration points:
- **Generator reuse**: KGSAGE's conditional-GAN machinery as *one candidate implementation* of pathway A (competes on merit with LLM/rule-based generation).
- **Eval-protocol reuse**: the standalone corruption-quality protocol (C2ST, off-family guard) transfers to scoring any Phase-2 generator.
- **Dataset continuity**: FB15k-237/CoDEx as the optional comparability arm.
- **Narrative link**: at most, thesis frames Phase 1 as "synthetic anomaly generation groundwork"; Phase 2 claims must not depend on KGSAGE novelty.
