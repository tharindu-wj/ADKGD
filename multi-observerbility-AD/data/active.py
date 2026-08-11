"""The one switch that decides which dataset every tool observes.

Tools import DATA and COLUMN_MEANINGS from HERE, never from a concrete dataset
module, so changing datasets means changing the single import below and nothing
else. The contract a dataset module must honour -- four names: NAME, ENTITY,
DATA, COLUMN_MEANINGS -- is documented in data/california_housing.py, the
reference implementation.

Assumed of every dataset, by agreement rather than by checked code: numeric
columns only, no missing cells, one row per entity. Hand this switch a
well-structured table or the LOF recipe in the tools quietly measures nonsense.

NOTE: the offline regression test (orchestrator_custom.py --dummy) replays a
script written against california_housing's columns. Run it with california
active; with another dataset active its scripted calls come back as ERROR
strings, which exercises error recovery rather than the happy path.
"""

from data.california_housing import COLUMN_MEANINGS, DATA, ENTITY, NAME  # noqa: F401
