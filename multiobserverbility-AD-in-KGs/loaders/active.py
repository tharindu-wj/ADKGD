"""The dataset switch.

Scripts and tools import DATASET from here and never name a dataset directly.
Changing dataset is changing the one import line below.
"""
from loaders import codexs as DATASET

__all__ = ["DATASET"]
