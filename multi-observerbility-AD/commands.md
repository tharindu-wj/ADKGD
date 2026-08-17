python agent_custom_single/orchestrator_custom.py "in this California census dataset, find block groups that sit in an abnormal geographic position - physically isolated from where the other blocks are located"

python agent_custom_single/orchestrator_custom.py "in this California census dataset, find block groups with an abnormal housing stock - the age and internal layout of the buildings themselves, not the people living in them"

python agent_custom_single/orchestrator_custom.py "in this California census dataset, find block groups with an abnormal resident population - how many people live there, how crowded their households are, and what they earn, not the buildings"

python agent_custom_single/orchestrator_custom.py "in this California census dataset, find block groups that are abnormal as complete records, taking every recorded attribute into account"

---

## Two observers (agent_adk_multiple) -- cells 2.5 and 3

    adk run agent_adk_multiple

Then type ONE or TWO numbered goals. The count decides the cell.

### Cell 2.5 -- two agents, ONE shared goal (the redundancy control)

Run this FIRST. Both observers get the identical intent, so their overlap is the
most agreement two viewpoints can show -- the ceiling that cell 3 is read
against. If they derive DIFFERENT columns from one intent, the observer point
underdetermines the viewpoint, which is itself the finding.

    Goal 1: find block groups that cannot describe a real place

### Cell 3 -- two agents, one goal each (the contamination test)

    Goal 1: find block groups that cannot describe a real place
    Goal 2: find block groups that do not fit their surrounding region

To measure one-mind contamination, run the SAME pair through the single agent
and compare the derived columns:

    adk run agent_adk_single
    > find block groups that cannot describe a real place, and separately ones that do not fit their region

If one mind's two viewpoints resemble each other more than two minds' do,
contamination is real. If not, cell 2 is sound and the simpler architecture
stands -- either way it is a result.

### Reading the run files

    runs/run_*_adk_gemini.json       one mind   (cell 2)
    runs/run_*_adk_two_gemini.json   two minds  (cells 2.5 and 3, see "cell")

Per-agent goal, spec, status and trace live under `observers`.
