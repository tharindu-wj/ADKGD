"""Phase 1 - Concept Module.

Schema and type information extraction from knowledge graphs.

Sub-modules:
  adapters/      KB-family adapters (Freebase now; WordNet + YAGO later)
  concept_pools  (Phase 1.2) builds headPool[r] and tailPool[r]
  cardinality    (Phase 1.2) classifies relations as 1-1, 1-N, N-1, N-N
  preprocess     (Phase 1.2) CLI orchestrator

This module produces three artefacts per dataset:
  data/<DATASET>/entity_types.tsv                          user-facing types
  data/<DATASET>/entity_types_metadata.json                provenance + stats
  experiments/gan/outputs/concept_pools/<DATASET>.pkl      internal cache

Downstream phases (adversarial/, corruption/) consume the .pkl cache;
the .tsv and .json files exist for reviewers and external consumers.
"""
