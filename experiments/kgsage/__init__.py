"""KGSAGE — Knowledge Graph Semantic Anomaly GEnerator.

Phase 1 module: encoder pretraining (RGCN + DistMult on FB15K-237).

The thesis pipeline has four phases:
  Phase 1 (this module)  : encoder pretraining
  Phase 2 (future work)  : KGSAGE Generator + Discriminator
  Phase 3 (future work)  : generation evaluation
  Phase 4 (future work)  : ADKGD integration

See README.md for the file walkthrough and run order.
See ../docs/THESIS_PLAN_pairgan_contradictions.md for the full plan.
"""
from .data import load_fb15k237
from .encoder import KGSAGEEncoder, KGSAGEDistMultDecoder, KGSAGELinkPredictor

__all__ = [
    "load_fb15k237",
    "KGSAGEEncoder",
    "KGSAGEDistMultDecoder",
    "KGSAGELinkPredictor",
]
