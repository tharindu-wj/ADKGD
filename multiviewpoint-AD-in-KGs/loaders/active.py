"""The dataset switch.

Scripts import DATASET from here and never name a dataset directly.
Changing dataset is changing the one import line below.
"""
from loaders import countries as DATASET

__all__ = ["DATASET"]
