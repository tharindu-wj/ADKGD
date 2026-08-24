"""The tool index: name -> function. This is what makes them tools.

It is the seam an agent binds to. The list is FIXED at four however many
scorers exist -- scorers are the menu run_scorer picks from, never tools in
their own right. Registering them individually would change both agents' tool
lists every time one was added, and the two agents must differ only in the goal
they are given.

Adding a tool:
    1. one new tools/<name>.py with one function returning a string
    2. one line here
    3. one line in the tool list of each agent that should see it
"""
from tools.describe_relation import describe_relation
from tools.list_relations import list_relations
from tools.run_scorer import run_scorer
from tools.sample import sample

TOOLS = {
    "list_relations": list_relations,
    "describe_relation": describe_relation,
    "sample": sample,
    "run_scorer": run_scorer,
}
