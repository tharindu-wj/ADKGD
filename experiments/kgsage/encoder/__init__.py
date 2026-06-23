"""KGSAGE encoder layer — Phase 1 (RGCN + DistMult).

Public API surfaces:
  KGSAGEEncoder            — RGCN backbone producing entity embeddings
  KGSAGEDistMultDecoder    — DistMult head producing relation embeddings as a side effect
  KGSAGELinkPredictor      — combined encoder + decoder for end-to-end training

After Phase 1 training:
  - encoder.ent_emb        → input to Phase 2 KGSAGE Generator + Discriminator
  - decoder.rel_emb        → input to Phase 2 KGSAGE Generator
"""
from kgsage.encoder.models import (
    KGSAGEEncoder,
    KGSAGEDistMultDecoder,
    KGSAGELinkPredictor,
)

__all__ = [
    "KGSAGEEncoder",
    "KGSAGEDistMultDecoder",
    "KGSAGELinkPredictor",
]
