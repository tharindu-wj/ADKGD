# Deep-Research Report — Multi-Viewpoint Anomaly Detection (Thesis Phase 2)

*Executed 16 Jul 2026 via a survey-first, connector-based multi-agent research workflow: 5 literature sweeps (82 papers collected, 20 full-paper deep-reads), per the agenda in `research_pathways.md` §5.*
*STATUS: sweeps complete and synthesized below. The formal 3-lens adversarial verification of the gap verdicts was interrupted by a session usage limit and will be appended; the verdicts below are **preliminary but evidence-dense** — every verdict is grounded in fetched full texts.*
*Connector note: the Consensus monthly quota (30 searches) was exhausted during this run (resets 1 Aug); later stages used Scholar Gateway + web search/fetch.*

---

## 0. Executive summary

| Gap claim (as in the STEM9003 lit review) | Preliminary verdict | Surviving (defensible) form |
|---|---|---|
| **g1** — no graph-based modelling for multi-viewpoint representation | **REFUTED as stated** | No method models views as *semantically distinct per-entity contexts with view-specific attribute schemas*; existing "views" are relation types over shared attributes (multiplex GAD) or feature splits (MVOD). Multi-view has no top-level place in any GAD taxonomy. |
| **g2** — no viewpoint-specialised reasoning | **OVERSTATED** | View-specialised *modelling* exists (per-view encoders, experts, per-layer streams). What does not exist: viewpoint-specialised *reasoning* — per-view semantic verdicts with explanations, by view-appropriate detectors/agents. The "fusion dilutes" diagnosis is already published (dPoE 2023) and must be cited, not claimed. |
| **g3** — no coordinated cross-view sharing during detection | **OVERSTATED (most pressured gap)** | Embedding-level cross-view sharing is MVOD's core mechanism; agentic in-detection coordination exists (MAKGED discussion, Audit-LLM debate, CORTEX cross-source explain-away). Survives only as: no coordination of viewpoint-specialised agents over *semantically distinct views of one entity on learned multi-view representations*, with explain-away adjudication. |
| **g4** — no synthetic anomaly generation with cross-view relationship patterns | **OVERSTATED — but the narrowed form is the thesis's strongest claim** | Rule-based cross-view generation (feature-swap) has existed since 2011. What does not exist: (i) *learned / relationship-aware* cross-view anomaly generation, (ii) any *released, parameterised benchmark* with an explicit view-interaction taxonomy, (iii) **any generation protocol at all for A3 (explained-away) and A4 (view-integrity) — both taxonomy cells are unclaimed**. |

**Bottom line.** None of the four gaps survives verbatim — but all four survive in narrowed forms that are *jointly* unoccupied, and the sweeps found the exact positioning anchors to cite. The combined defensible claim: **no framework exists in which viewpoint-specialised agents reason over semantically distinct views of one entity (typed/heterogeneous graph representation), coordinate via evidence exchange and explain-away adjudication during detection, and are evaluated on a controllable, taxonomy-labelled cross-view anomaly benchmark — and the A3/A4 anomaly classes have no benchmark or generation protocol anywhere.** Pathways A (benchmark) and C+D (coordinated detection + ablation science) target exactly this hole.

---

## 1. Sweep S1 — The evolution ladder (fully cited S-table)

Survey coverage: rungs 1–4 have strong anchors; **rung 5 has no peer-reviewed dedicated survey as of Jul 2026** (closest: a Feb-2026 Sensors survey [S6], 403-blocked at publisher; a time-series-scoped arXiv survey). The thesis background chapter can legitimately BE the missing synthesis.

| Rung | Anchor survey | Canonical methods | Capability added | Limitation left (motivates next rung) |
|---|---|---|---|---|
| 1. Classical ML | Chandola et al. 2009, ACM CSUR [1] | LOF (SIGMOD'00) [7]; OC-SVM (Neural Comp.'01) [8]; iForest (ICDM'08) [9] | Principled outlierness: local density, one-class boundary, isolation depth; anomalies are few/different/context-relative | Handcrafted features; degrades on high-dimensional, structured, relational data [2] |
| 2. Deep / graph | Pang et al. 2021, ACM CSUR [2]; Qiao et al. 2025, TKDE [3] | Deep SVDD (ICML'18) [10]; MemAE (ICCV'19) [11]; DOMINANT (SDM'19) [12] | End-to-end learned representations; graph structure via GNN reconstruction | AEs "generalize so well they reconstruct anomalies too" (MemAE's stated motivation) [11]; no semantics; label scarcity |
| 3. Generative | Sabuhi et al. 2021, IEEE Access [4]; Chakraborty et al. 2024 [24]; diffusion survey arXiv:2501.11430 | AnoGAN (IPMI'17) [13] → f-AnoGAN [14] → GANomaly (ACCV'18) [15]; AnoDDPM (CVPRW'22) [16]; CutPaste (CVPR'21) [17] / DRAEM (ICCV'21) [18] | Explicit normality modelling; synthetic anomaly generation for training (crude synthetic anomalies suffice for SOTA detectors [17,18]) | GAN instability/mode collapse (AnoDDPM's rationale [16]); single-data-type generation; no reasoning or explanation |
| 4. LLM-based | Ren et al. 2025, AI Magazine [5] (FM as encoder/detector/interpreter); Su et al. 2024 SLR [25]; Xu & Ding 2024 [26] | SIGLLM (DSAA'24) [19]; AnomalyGPT (AAAI'24) [20]; AAD-LLM (BigData'24) [27]; semantic AD for robotics (Elhafsi et al., Auton. Robots '23) [28] | Zero-/few-shot detection; semantic interpretation; natural-language explanation of detections | Zero-shot LLMs still ~30% below SOTA deep AD (SIGLLM's own eval [19]); single-pass; hallucination; no iteration, tools, memory [25] |
| 5. Agentic | Sapkota et al. 2026, Inf. Fusion [6] (definitional); **no dedicated AD survey** [S6] | Argos (Microsoft, arXiv:2501.14170) [21]; AD-AGENT (IJCNLP-AACL'25) [22]; CORTEX (arXiv:2510.00311) [23]; Audit-LLM (arXiv:2408.08902) [29]; MAKGED (arXiv:2501.15791) [30]; NASA DSN (arXiv:2508.21111) [31] | Planning, tool orchestration, persistent memory, multi-agent coordination, iterative self-correction | **Thesis territory:** specialisation is by pipeline stage (Argos, AD-AGENT), alert workflow (CORTEX), or structural subgraph (MAKGED) — never by contextual/behavioural *view of one entity*; no A3 explain-away across semantic views; no principled per-anomaly-class evaluation |

---

## 2. Sweep S2 — Agentic/LLM AD systems 2024–2026 (the competitive landscape)

The multi-agent AD family is **~18 months old**; dedicated survey coverage only appeared Dec 2025–Feb 2026 [S6]. The five systems that matter most for positioning:

1. **CORTEX (arXiv 2510.00311, 2025)** [23] — *the closest precedent to pathway C and the mandatory positioning anchor.* SOC alert triage: orchestrator → behavior-analysis router → **7 workflow-specialised evidence agents** (typed APIs to SIEM/identity/TI) → reasoning agent that **reconciles cross-source evidence and explains away false positives** (Benign Positive / FP-Logic / FP-Data subclasses). Quantified: actionable F1 0.78 vs 0.66 single-agent ReAct; FP rate 29.8%→14.2% on thousands of production traces; cost 3.4× latency, 5.7× tokens. This **is** flag-then-explain with A2-like (impossible travel) and A3-like (explain-away) semantics — but over *SOC tool outputs*, not learned multi-view entity representations, and views = data sources, not modelled viewpoints. Dataset released.
2. **MAKGED (arXiv 2501.15791, 2025)** [30] — four LLM agents specialised on *directional structural subgraphs* (head/tail × fwd/bwd) of the same KG triple; GCN subgraph embedding + Llama2, LoRA per agent; independent judgment → up to 3 discussion rounds → majority vote (summarizer breaks 2-2 ties, ~12% of cases). Beats CAGED/CCA/KG-BERT etc. on FB15K/WN18RR (~30% injected errors). Proof that *perspective-specialised agents + in-detection discussion beat single-model fusion on graphs* — but its "perspectives" are structural directions around one triple, **not semantic entity views**. Code released.
3. **Audit-LLM (arXiv 2408.08902, 2024)** [29] — insider-threat detection from multi-source logs; Decomposer/Tool-Builder/Executor + **Evidence-based Multi-agent Debate** (two executors exchange reasoning to consensus during detection). Cautionary ablation: CoT decomposition −25% accuracy when removed, tools −9%, **debate only −1.5%** — redundant-peer debate on the same evidence gives marginal gains; the lift must come from *complementary* (view-specialised) evidence, which is exactly the thesis hypothesis D should test.
4. **Argos (Microsoft, arXiv 2501.14170)** [21] — agents propose/repair/review explainable rule-code offline; deployed rules are cheap deterministic detectors (1.5–34× faster inference, +9.5–28.3% F1). Stage-specialised; univariate; runtime non-agentic.
5. **AD-AGENT (arXiv 2505.12594, IJCNLP-AACL'25)** [22] — multi-agent *pipeline construction* (intent→prep→model selection→codegen over PyOD/PyGOD/TSLib) with shared workspace; agents exchange engineering artifacts, never detection signals.

Also mapped: NASA DSN agentic pipeline (sequential flag-then-explain, mutually blind components; case-study eval) [31]; the maritime "agentic anomaly management" paper (arXiv 2507.15676) turned out to be a **narrative literature review with no system, no experiments, no datasets** — cite as motivation only, not prior solution [32]; AIS-LLM (single-LLM multi-task fusion — a ready-made exemplar of the g2 dilution pattern) [33]; SentinelAgent (detects anomalies *of* multi-agent systems — inverse direction) [34]; CodeAD (Argos pattern for logs) [35]; a Dec-2025 maritime benchmark whose anomalies are **synthesized by LLM agents** (Trajectory Synthesizer + Anomaly Injector; node/edge/graph-level) — the nearest thing to agentic anomaly generation, though cross-entity within one view, not cross-view [36].

**Evaluation norms found:** single-domain benchmarks (CERT, FB15K/WN18RR, TSAD suites) + proprietary traces; ablations mostly single- vs multi-agent. **No agentic system is evaluated per anomaly class or on any cross-view/MVOD-style benchmark — an open protocol slot pathway D claims.**

## 3. Sweep S3 — Multi-view outlier detection (the correct classical prior art)

**No dedicated MVOD survey exists** (confirmed via the 2023 meta-survey of 56 OD surveys [37]; closest: multi-view deep one-class exploration, TNNLS 2024 [38]; a 2019 Springer book chapter codifies the taxonomy [39]). Lineage reconstructed via RCPMOD's related work + fetched primaries:

- **HOAD (ICDM 2011)** [40] — field origin; "horizontal anomalies" = same object inconsistent across sources (**= our A2**). Joint spectral clustering over per-source similarity graphs of the same objects; score = cross-view spectral disagreement. Also the **origin of the swap-based injection protocol**.
- **Probabilistic formalisation of A2**: Iwata & Yamada (NeurIPS 2016) [41] — normal = all views generated from ONE latent vector; anomaly = needs multiple latents (Dirichlet process infers count). Hierarchical Bayesian extension IJCAI 2020 [42]. Ready-made theory footing for the taxonomy.
- **Latent/pairwise era**: DMOD (IJCAI'15) [43] — codified the **attribute / class / class-attribute** trichotomy and the swap protocol every later paper copies; MLRA (TKDD'18) [44]; CRMOD (TIP'17) [45] — consensus regularization for >2 views.
- **Deep era**: MODDIS (ICDM'19) [46] first neural MVOD; **NCMOD (AAAI'21)** [47] — per-view autoencoders + jointly-learned consensus kNN graph; canonical modern baseline, code released; SRLSP (TKDD'22) [48]; **MODGD (Inf. Fusion 2023)** [49] — *graph-based* MVOD (per-view graphs → consensus graph via denoising, structured-outlier extraction), code released; IAMOD (TKDD'24) [50] — the field's **only theory attempt** ("three occurrence mechanisms"; covers A1/A2 space, **no A3/A4**); dPoE (ACM MM'23) [51] — *"Debunking the free fusion myth"*: view-specific experts in disentangled product-of-experts — **the g2 dilution critique, already published**; **RCPMOD (ACM MM'24)** [52] — partial (missing-view) MVOD SOTA: contrastive + outlier memory bank; missing views are *imputed, never scored as anomalous* → **A4 remains open**; Tucker/meta-learning MVOD (Inf. Fusion 2025) [53] — field active, still swap-evaluated.
- **Taxonomy mapping (load-bearing for the thesis):** MVOD *class outliers* ≈ **A2** (per-view normal, inconsistent across views); *attribute outliers* ≈ **A1**; *class-attribute* = hybrid. **No MVOD category corresponds to A3** — the taxonomy treats cross-view inconsistency as always anomalous, never exculpatory. Partial-MV line (CL 2018 [54], RCPMOD) touches A4's data condition but never treats view-integrity as an anomaly class.
- **The standard protocol (HOAD'11 → 2025, unchanged):** attribute = randomise features in all views; class = **swap feature vectors of two instances in ⌊V/2⌋ views**; class-attribute = both; ratios 2–8%/type; AUC. Iwata & Yamada prove swaps leave marginals unchanged → **swap isolates A2 from A1** [41]. In 15 years **no MVOD paper has replaced the random swap with a learned generator** — swaps are relationship-blind and cannot express plausible-but-impossible combinations.

**Baseline list (code):** NCMOD (github.com/auguscl/NCMOD), MODGD, SRLSP (github.com/wy54224/SRLSP), LDSR (github.com/kailigo/mvod) [55], RCPMOD (reimpl. github.com/yankehan/MCA2), MODDIS, MLRA, DMOD, CRMOD, HOAD, CL, IAMOD, dPoE.

## 4. Sweep S4 — Multiplex / heterogeneous graph AD (g1's direct test)

No multilayer-network AD survey exists; the 2026 J. Complex Networks multilayer review omits AD entirely [56]; the Qiao TKDE survey's repo lists **only ~4 multiplex/multi-view methods among hundreds of GAD entries** [3] — the sub-area is real but thin.

- **ANOMULY (NeurIPS'22 TGL wksp)** [57] — multiplex *dynamic* edge AD: **separate GNN+GRU stream per relation layer + cross-layer attention sharing information during detection**. 9 datasets; injected anomalies incl. a layer-dependent type. The structural precedent closest to "per-view detectors with cross-view coordination" — at embedding level, nothing agentic. Code: github.com/ubc-systopia/ANOMULY.
- **UMGAD (ICDE 2025)** [58] — SOTA multiplex heterogeneous GAD: per-relation graph-masked autoencoders + augmented-view contrastive learning + learnable-weight score fusion; +12.25% AUC avg over 25+ baselines; label-free threshold selection. **Never checks whether a node's views are mutually consistent** (fusion averages errors — a marginally-normal-everywhere node scores low) → A2 handled only implicitly. Mandatory baseline + protocol source.
- **AnomMAN (Inf. Sciences 2023)** [59] — literally titled AD on "multi-view attributed networks" (relation-type subgraphs, shared attributes; attention-fused reconstruction). **Terminological collision: the thesis's term is occupied** and must be explicitly differentiated. A1-only in our taxonomy (verified by full read).
- **SIGIL / "Cluster Aware GAD" (WWW 2025)** [60] — multi-view GAD via shared soft cluster assignment + contrastive consensus; evaluated on CERT/IMDB/DBLP vs 5 MVOD + 8 GAD baselines. Strongest recent proof that **multi-view GAD exists but is fusion-only**.
- Fraud lineage: CARE-GNN (CIKM'20) [61], PC-GNN (WWW'21) [62] — multi-relation graphs of one entity, aggregated into a single embedding before one classifier (early fusion; supports the dilution argument). KG error detection as cross-view consistency: CAGED (CIKM'22) [63] — "hyper-views" are structural augmentations, single loss; CCA (AAAI'24) [64].

**Common denominator across ALL found methods:** per-view/layer signals are fused (attention/weights/shared objectives) into one score; **none outputs per-view verdicts adjudicated across views; none implements explain-away (A3); none scores explicit cross-view consistency as its detection principle** (closest: HOAD's spectral disagreement, 2011, pre-deep).

## 5. Sweep S5 — Benchmarks & controllable generation (g4's direct test)

- **ADBench (NeurIPS'22 D&B)** [65] — the closest general precedent: 4 **parameterised** synthetic anomaly types — local (GMM, α=5), global (uniform), clustered, and **dependency anomalies (Vine-Copula: marginals normal, joint structure violated)**. Dependency anomalies are **the single-view analogue of A2** → frame A2 generation as *lifting dependency anomalies to the view level*. Flat vectors; no entity-view concept.
- **BOND (NeurIPS'22)** [66] / **PyGOD (JMLR'24)** [67] — the only *released, reusable* injection APIs (`gen_contextual_outlier`, `gen_structural_outlier`); two types, single graph, no view concept. **The artifact form pathway A should ship.**
- **GADBench (NeurIPS'23)** [68] — organic labels only; argues injected anomalies are "straightforward to identify" — **the realism critique any generator must answer**; citable motivation for learned generation grounded in real cross-view relationships. Newest "GAD in the Wild" (2026) [69]: still no view axis; its missing-attribute arm brushes A4 without treating integrity as anomalous.
- **MVOD protocol** — controllable cross-view generation since 2011, but: rule-based random swaps, parameterised only by ratio, re-implemented ad-hoc per paper, **never released as a benchmark**, and relationship-blind.
- **Vision "multi-view AD" namespace collision**: Real-IAD (CVPR'24) [70], M2AD, SiM3D, MANTA — camera angles, out of scope, but they own the term "multi-view AD benchmark"; the thesis must disambiguate.
- **Negative finding (July 2026 sweep):** no GAN/diffusion/LLM method found that conditions anomaly synthesis on multiple contextual views of an entity or cross-view relationships. Nearest: agentic anomaly injection in the maritime benchmark [36] (relational but cross-entity/single-view); Double Helix Diffusion [71] (image-domain).

**Net verdict on the boldest claim ("pathway A is first"):** FALSE in strong form — the swap protocol is a 15-year-old controllable cross-view generator and MVOD's trichotomy is a rudimentary view-interaction taxonomy. **TRUE in the form that matters:** no *released, parameterised* benchmark with an explicit view-interaction taxonomy exists; **A3 and A4 have no generation protocol anywhere**; and no generator is *learned/relationship-aware*. Position A as: ADBench's dependency-anomaly concept, lifted to view level, with A3/A4 as new cells, shipped as a PyGOD-style API.

---

## 6. Gap verdicts in full (preliminary; adversarial 3-lens verification to be appended)

### g1 — "no graph-based modelling for multi-viewpoint representation" → REFUTED as stated
*Killer citations:* HOAD (2011) [40]; ANOMULY (2022) [57]; AnomMAN (2023) [59]; MODGD (2023) [49]; UMGAD (ICDE 2025) [58]; SIGIL (WWW 2025) [60].
*Surviving form:* **"Existing graph-based multi-view AD treats views as relation types over a shared attribute space or as feature-split similarity graphs; no method represents views as semantically distinct per-entity contexts with view-specific schemas (e.g., academic/employment/health), and no GAD taxonomy recognises multi-view as a modelling axis"** — supported by Qiao'25 repo census (~4/hundreds) [3] and the 2026 multilayer review's omission of AD [56].

### g2 — "no viewpoint-specialised reasoning" → OVERSTATED
*Undermining:* NCMOD per-view AEs [47]; dPoE view-specific experts + the published "fusion dilutes" critique [51]; ANOMULY per-layer streams [57]; MAKGED perspective-specialised agents [30]; CORTEX workflow-specialised agents [23].
*Surviving form:* **"View-specialisation exists only at embedding level (encoders/experts/streams) or for structural/workflow slices; no framework performs viewpoint-specialised *reasoning* — semantic per-view verdicts with explanations by view-appropriate detectors — over contextual views of one entity."** The dilution diagnosis must be cited to dPoE, not claimed as new.

### g3 — "no coordinated cross-view sharing during detection" → OVERSTATED (most pressured)
*Undermining:* consensus/pairwise regularization is MVOD's core (DMOD [43], CRMOD [45], NCMOD [47], RCPMOD [52]); ANOMULY cross-layer attention during detection [57]; CAGED cross-view consistency scoring [63]; MAKGED 3-round discussion + vote [30]; Audit-LLM EMAD debate [29]; **CORTEX cross-source reconciliation with explain-away and quantified FP reduction** [23].
*Surviving form:* **"Cross-view sharing exists as latent-space consensus (not reasoning) or as agentic coordination over tool outputs / structural slices (not semantically distinct entity views on learned multi-view representations); no system exchanges detection-level verdicts between viewpoint-specialised agents with explain-away (A3) adjudication."** Recommend merging g2+g3 into one claim (see §7).

### g4 — "no synthetic anomaly generation with cross-view relationship patterns" → OVERSTATED, strongest narrowed form
*Undermining:* HOAD swap protocol 2011→2025 [40,43,52]; Iwata & Yamada's proof swaps create exactly cross-view anomalies [41]; ADBench dependency anomalies [65]; agentic relational injection [36].
*Surviving form:* **"Cross-view anomaly generation has been frozen as random, relationship-blind feature swaps for 15 years; no learned or relationship-aware generator exists; no released parameterised benchmark carries a view-interaction taxonomy; and the A3 (explained-away normal) and A4 (view-integrity) classes have no generation protocol at all."** GADBench's realism critique [68] independently motivates learned generation.

---

## 7. Implications — what changes in the pathway map

1. **Rewrite the four gaps in the narrowed forms above** (lit-review §3 / thesis gap statement). Every narrowed form has killer citations attached; the gaps as currently written would not survive examination.
2. **Merge g2+g3 rhetorically** into the single defensible claim: *no framework where viewpoint-specialised agents reason over semantically distinct views of one entity and adjudicate via evidence exchange + explain-away during detection.* CORTEX [23] is the mandatory positioning anchor (nearest system, different substrate); MAKGED [30] the nearest graph instance (structural, not semantic views).
3. **Pathway A is confirmed and sharpened**, not diminished: (i) frame A2 generation as *ADBench dependency anomalies lifted to view level* [65]; (ii) the HOAD swap protocol becomes the **mandatory baseline generator** to beat on realism/hardness; (iii) **A3/A4 are unclaimed cells** — the strongest novelty in the whole programme; (iv) ship as a PyGOD-style injection API + released benchmark (the artifact MVOD never produced); (v) answer GADBench's realism critique with relationship-grounded generation.
4. **Pathway B must position against multiplex GAD**: UMGAD [58] and ANOMULY [57] become mandatory baselines/protocol sources; the representation claim is now precisely "semantically distinct views with view-specific schemas (heterogeneous per-view graphs anchored on an entity spine)" vs relation-multiplex layers vs feature views. AnomMAN's title [59] forces explicit terminological differentiation; so does the vision Real-IAD family [70].
5. **Pathway C novelty is confirmed but must be argued at reasoning level**: per-view *verdicts* + arbiter with explain-away, targeting A2/A3 — no found system does this over learned multi-view entity representations. Audit-LLM's ablation (debate alone: +1.5%) [29] is the cautionary datum: **the hypothesis must be that view-complementary evidence (not peer redundancy) is what makes coordination pay** — which is exactly what pathway D tests.
6. **Pathway D gets a free protocol slot**: no agentic AD system is evaluated per anomaly class or on any cross-view benchmark; adopt Audit-LLM-style component ablations + CORTEX-style cost accounting (latency/tokens) as the evaluation template.
7. **Pathway S (evolution chapter) is fully citable now** (S-table §1); rung-5 is survey-thin → the background chapter doubles as a contribution. Fetch the Sensors 26(8):2330 survey [S6] via library access (publisher 403-blocked).
8. **Baseline superset for the whole programme** (dedupe per experiment): iForest/LOF/OC-SVM [9,7,8] · Deep SVDD/MemAE [10,11] · DOMINANT [12] · NCMOD/MODGD/SRLSP/RCPMOD/IAMOD/dPoE [47,49,48,52,50,51] · ANOMULY/UMGAD/AnomMAN/Mul-GAD [57,58,59] · CARE-GNN/PC-GNN [61,62] · CAGED/CCA [63,64] · SIGLLM zero-shot [19] · single-LLM-one-prompt · CORTEX-style multi-agent [23] · MAKGED [30] · concat-views + single detector (A2-blindness control).
9. **Terminology guardrails**: "multi-view AD" is occupied three ways (vision camera-views; MVOD feature-views; multiplex relation-views). Keep **"multi-viewpoint"** and define it against all three in one table early in the thesis.

## 8. References

[1] Chandola, Banerjee, Kumar. Anomaly Detection: A Survey. ACM CSUR 41(3), 2009. doi:10.1145/1541880.1541882
[2] Pang, Shen, Cao, van den Hengel. Deep Learning for Anomaly Detection: A Review. ACM CSUR 54(2), 2021. doi:10.1145/3439950
[3] Qiao, Tong, An, King, Aggarwal, Pang. Deep Graph Anomaly Detection: A Survey and New Perspectives. IEEE TKDE 37(9), 2025. arXiv:2409.09957; repo: github.com/mala-lab/Awesome-Deep-Graph-Anomaly-Detection
[4] Sabuhi, Zhou, Bezemer, Musilek. Applications of GANs in Anomaly Detection: SLR. IEEE Access 9, 2021. arXiv:2110.12076
[5] Ren et al. Foundation Models for Anomaly Detection: Vision and Challenges. AI Magazine 46(4), 2025. doi:10.1002/aaai.70045
[6] Sapkota, Roumeliotis, Karkee. AI Agents vs. Agentic AI. Information Fusion 126:103599, 2026. arXiv:2505.10468
[7] Breunig, Kriegel, Ng, Sander. LOF. SIGMOD 2000.
[8] Schölkopf et al. Estimating the Support of a High-Dimensional Distribution. Neural Computation 13(7), 2001.
[9] Liu, Ting, Zhou. Isolation Forest. ICDM 2008.
[10] Ruff et al. Deep One-Class Classification. ICML 2018.
[11] Gong et al. MemAE. ICCV 2019.
[12] Ding, Li, Bhanushali, Liu. DOMINANT. SDM 2019.
[13] Schlegl et al. AnoGAN. IPMI 2017.
[14] Schlegl et al. f-AnoGAN. Medical Image Analysis 54, 2019.
[15] Akcay et al. GANomaly. ACCV 2018.
[16] Wyatt et al. AnoDDPM. CVPR Workshops 2022.
[17] Li et al. CutPaste. CVPR 2021.
[18] Zavrtanik et al. DRAEM. ICCV 2021. arXiv:2108.07610
[19] Alnegheimish et al. SIGLLM: LLMs as zero-shot TS anomaly detectors? IEEE DSAA 2024. arXiv:2405.14755
[20] Gu et al. AnomalyGPT. AAAI 2024.
[21] Argos: Agentic Time-Series AD with Autonomous Rule Generation. arXiv:2501.14170, 2025. github.com/microsoft/argos
[22] AD-AGENT: A Multi-agent Framework for End-to-end AD. Findings IJCNLP-AACL 2025. arXiv:2505.12594
[23] Wei et al. CORTEX: Collaborative LLM Agents for High-Stakes Alert Triage. arXiv:2510.00311, 2025.
[24] Chakraborty et al. Ten Years of GANs. MLST 5(1), 2024. arXiv:2308.16316
[25] Su et al. LLMs for Forecasting and AD: SLR. arXiv:2402.10350, 2024.
[26] Xu, Ding. LLMs for Anomaly and OOD Detection: A Survey. arXiv:2409.01980, 2024.
[27] Russell-Gilbert et al. AAD-LLM. IEEE BigData 2024.
[28] Elhafsi et al. Semantic anomaly detection with LLMs. Autonomous Robots, 2023.
[29] Song et al. Audit-LLM: Multi-Agent Collaboration for Log-based Insider Threat Detection. arXiv:2408.08902, 2024.
[30] MAKGED: Multi-Agent Framework for KG Error Detection. arXiv:2501.15791, 2025. github.com/ElevenLiy/MAKGED
[31] Automating the Deep Space Network: Adaptive AD through Agentic AI. arXiv:2508.21111, 2025.
[32] Barenji, Khoshgoftar. Agentic AI for autonomous anomaly management. arXiv:2507.15676, 2025. (narrative review — no system)
[33] AIS-LLM. arXiv:2508.07668, 2025.
[34] SentinelAgent. arXiv:2505.24201, 2025.
[35] CodeAD. arXiv:2510.22986, 2025.
[36] Spatio-Temporal Graphs Beyond Grids: Benchmark for Maritime AD. arXiv:2512.20086, 2025.
[37] Olteanu et al. Meta-survey on outlier and anomaly detection. Neurocomputing, 2023.
[38] Multiview Deep Anomaly Detection: A Systematic Exploration. IEEE TNNLS, 2024. arXiv:2104.13000
[39] Multi-view Outlier Detection. In: Linking and Mining Heterogeneous and Multi-View Data, Springer 2019.
[40] Gao, Fan, Turaga, Parthasarathy, Han. HOAD: A Spectral Framework for Detecting Inconsistency across Multi-Source Object Relationships. ICDM 2011.
[41] Iwata, Yamada. Multi-view Anomaly Detection via Robust Probabilistic Latent Variable Models. NeurIPS 2016. arXiv:1411.3413
[42] Wang, Lan. Hierarchical Bayesian Multi-View AD. IJCAI 2020.
[43] Zhao, Fu. Dual-Regularized Multi-View Outlier Detection (DMOD). IJCAI 2015.
[44] Li, Shao, Fu. MLRA. ACM TKDD, 2018.
[45] Zhao, Liu, Ding, Fu. CRMOD. IEEE TIP, 2017.
[46] Ji et al. MODDIS. ICDM 2019.
[47] Cheng, Wang, Liu. NCMOD. AAAI 2021. github.com/auguscl/NCMOD
[48] Wang et al. SRLSP. ACM TKDD, 2022. github.com/wy54224/SRLSP
[49] Hu, Wang, Zhou, Du. MODGD: Multi-view Outlier Detection via Graphs Denoising. Information Fusion, 2023/24.
[50] Lai, Wang, Chen, Zheng. IAMOD. ACM TKDD 18(4), 2024. doi:10.1145/3638354
[51] dPoE: Debunking Free Fusion Myth. ACM MM 2023. arXiv:2310.18728
[52] Wang et al. RCPMOD. ACM MM 2024. arXiv:2408.07819
[53] Low-rank Tucker decomposition MVOD via meta-learning. Information Fusion, 2025.
[54] Guo, Zhu. Partial Multi-View Outlier Detection Based on Collective Learning. AAAI 2018.
[55] Li, Li, Ding, Zhang, Fu. LDSR. AAAI 2018. github.com/kailigo/mvod
[56] Aleta, Moreno, Fortunato et al. Multilayer network science. J. Complex Networks, 2026. arXiv:2511.23371
[57] Behrouz, Seltzer. ANOMULY. NeurIPS 2022 TGL Workshop. arXiv:2211.08378. github.com/ubc-systopia/ANOMULY
[58] Li et al. UMGAD: Unsupervised Multiplex Graph Anomaly Detection. ICDE 2025. arXiv:2411.12556
[59] Chen et al. AnomMAN. Information Sciences 628, 2023. arXiv:2201.02822
[60] Zheng et al. SIGIL / Cluster Aware Graph Anomaly Detection. WWW 2025. arXiv:2409.09770
[61] Dou et al. CARE-GNN. CIKM 2020. github.com/YingtongDou/CARE-GNN
[62] Liu et al. PC-GNN. WWW 2021. github.com/PonderLY/PC-GNN
[63] Zhang et al. CAGED: Contrastive KG Error Detection. CIKM 2022. arXiv:2211.10030. github.com/Qing145/CAGED
[64] Liu et al. CCA: KG Error Detection with Contrastive Confidence Adaption. AAAI 2024.
[65] Han, Hu, Huang, Jiang, Zhao. ADBench. NeurIPS 2022 D&B. arXiv:2206.09426
[66] Liu et al. BOND. NeurIPS 2022 D&B.
[67] Liu et al. PyGOD. JMLR, 2024.
[68] Tang, Hua, Gao, Zhao, Li. GADBench. NeurIPS 2023 D&B.
[69] GAD in the Wild. arXiv:2605.07133, 2026.
[70] Real-IAD. CVPR 2024. realiad4ad.github.io/Real-IAD
[71] Double Helix Diffusion. arXiv:2509.12787, 2025.
[S6] Agentic and LLM-Based Multimodal Anomaly Detection: Architectures, Challenges, and Prospects. Sensors 26(8):2330, 2026 (Preprints.org 202602.1368). *(publisher 403-blocked — obtain via library)*

*Provenance note: entries marked "unverified" in the underlying sweep data (e.g., Marcos Alvarez 2013 CIKM; MuvAD venue) are recorded in the workflow output at `tasks/w8dd8t4q3.output`; verify before citing in the thesis.*
