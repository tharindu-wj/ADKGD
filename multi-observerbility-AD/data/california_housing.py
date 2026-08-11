"""The dataset: the frame every tool reads, and the vocabulary that describes it.

WHY THIS IS NOT IN tools/
-------------------------
tools/ holds one file per tool, and nothing else. The dataset is not a tool --
it is what the tools look at. Keeping it here means no tool has to import
another tool just to reach the data (describe_column used to import DATA from
run_lof_per_viewpoint, which quietly made run_lof_per_viewpoint a dependency of a tool that does no scoring).

IMPORT DIRECTION -- one way only
--------------------------------
    orchestrator  ->  tools/  ->  data/

This module is a LEAF: it imports nothing from the project. It must never import
a tool, an agent, or an LLM backend. That is what keeps adding a new
tool, or a second orchestration, a pure addition.

LOADED ONCE, SHARED BY ALL
--------------------------
Three tool files import DATA from here, but the dataset is fetched exactly once:
Python caches modules, so every importer gets the SAME DataFrame object. It
looks like three loads and is not. The check that proves it:

    import tools.describe_column as a, tools.run_lof_per_viewpoint as b
    assert a.DATA is b.DATA

Note the fetch runs at import time, so importing this module (from a tool, a
notebook, or a quick CLI check) triggers the scikit-learn download/cache read.
That has always been true; if it ever becomes a nuisance, wrap it in a lazy
get_data() -- there is no need today.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not score, threshold, filter or rank anything. Producing an anomaly
score remains the exclusive job of tools/run_lof_per_viewpoint.py (PROJECT_SPEC INV-3). Moving
the data out of that file did not move that rule with it.

ADDING A SECOND DATASET
-----------------------
A dataset is ONE file in data/ exporting four names -- this file is the
reference implementation of that contract:

    NAME             short id, recorded in every run file
    ENTITY           what one row IS, as a noun ("census block", "patient", ...)
    DATA             a pandas DataFrame. Numeric columns only, no missing cells,
                     index = entity id -- assumed, not checked: the LOF recipe
                     in the tools quietly measures nonsense otherwise
    COLUMN_MEANINGS  {column: plain-English meaning}. The agent authors its
                     observer points FROM these, so a dataset whose columns have
                     no stateable meaning (V1..V28) cannot support this framework

Then point data/active.py at the new module. Nothing else changes: every tool
imports from data.active and never knows which dataset it serves. Ames (79
features) is the one the roadmap expects next, because several experiments have
no statistical power at 8 columns.
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing

# =============================================================================
# DATA - loaded once, used by every tool that needs it.
# One row = one census block group in California, 1990.
# =============================================================================

raw = fetch_california_housing()

#: Short id, recorded in every run file so a result can never be mistaken for
#: another dataset's.
NAME = "california_housing"

#: What one row IS. The agent's prompt says this instead of hardcoding a noun;
#: the tools themselves just say "entity" and stay dataset-blind.
ENTITY = "census block group"

#: The dataset every tool reads. 20,640 rows x 8 columns, no ID column.
DATA = pd.DataFrame(raw.data, columns=list(raw.feature_names))

#: What each column means, in the agent's language. This is the ONLY place the
#: column vocabulary is written down. It lives beside the frame rather than in
#: list_columns.py because it describes the DATASET, not the tool that prints it
#: -- describe_column can reach for it too without importing a tool.
#:
#: Note AveRooms, AveBedrms and AveOccup are all ratios with *households* in the
#: denominator, which is why blocks with very few households show absurd values
#: (an "average household" of 1,243 people is an institution, not a home).
COLUMN_MEANINGS = {
    "MedInc": "median household income, in tens of thousands of dollars",
    "HouseAge": "median age of the houses, in years",
    "AveRooms": "average rooms per household (rooms / households)",
    "AveBedrms": "average bedrooms per household (bedrooms / households)",
    "Population": "people living in the block group",
    "AveOccup": "average people per household (population / households)",
    "Latitude": "degrees north; higher = further north",
    "Longitude": "degrees east; more negative = further west",
}
