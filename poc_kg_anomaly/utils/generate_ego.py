"""Draw the local neighbourhood of one triple, so a human can judge it by eye.

  green  edge = a TRUE tail of the centre for this relation
  red    edge = the INJECTED fake
  grey   edge = neighbor
  blue   edge = where a neighbour lives

  square node = region      round node = country
A red square in a neighbor slot, or a red circle in a locatedin slot, is a
TYPE error and should be obvious without knowing any geography.
"""
import collections
from pathlib import Path

MAX_NEIGHBOURS = 9   # past this the picture stops being readable


def build_context(real_triples):
    """Index the clean graph once: who lives where, who borders whom."""
    located = collections.defaultdict(set)
    neighbours = collections.defaultdict(set)
    for h, r, t in real_triples:
        if r == "locatedin":
            located[h].add(t)
        else:
            neighbours[h].add(t)
            neighbours[t].add(h)          # neighbor is symmetric in this KG
    regions = {t for h, r, t in real_triples if r == "locatedin"}
    return {"located": located, "neighbours": neighbours, "regions": regions}


def hops(start, goal, neighbours, limit=6):
    """Shortest path length in the neighbour graph, or None if further than limit."""
    if start == goal:
        return 0
    seen, frontier = {start}, [start]
    for d in range(1, limit + 1):
        nxt = []
        for node in frontier:
            for m in neighbours.get(node, ()):
                if m == goal:
                    return d
                if m not in seen:
                    seen.add(m)
                    nxt.append(m)
        frontier = nxt
        if not frontier:
            break
    return None


def evidence(h, r, t_fake, ctx):
    """How strongly does the neighbourhood argue against this fake?

    Returns (score, note). score None = no local evidence exists at all.
    Higher score = more contradicted. Scores are NOT comparable across relations.
    """
    located, neighbours = ctx["located"], ctx["neighbours"]
    nbrs = sorted(neighbours.get(h, ()))
    if r == "locatedin":
        if not nbrs:
            return None, "no neighbours in the graph -- no local evidence at all"
        true_regions = located.get(h, set())
        agree_true = sum(1 for n in nbrs if located.get(n, set()) & true_regions)
        agree_fake = sum(1 for n in nbrs if t_fake in located.get(n, set()))
        return agree_true - agree_fake, (
            f"{agree_true}/{len(nbrs)} neighbours share a TRUE region, "
            f"{agree_fake}/{len(nbrs)} are in the FAKE region {t_fake}")
    if t_fake in ctx["regions"]:
        return 99, f"{t_fake} is a REGION, and regions are never neighbours -- type error"
    d = hops(h, t_fake, neighbours)
    if d is None:
        return 99, "claimed neighbour is >6 hops away, or unreachable"
    return d - 1, f"claimed neighbour is {d} hops away in the real neighbour graph"


def draw_ego(h, r, t_fake, kind, ctx, out_path, title_note=""):
    """Render one ego network to out_path. Returns the path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    located, neighbours, regions = ctx["located"], ctx["neighbours"], ctx["regions"]
    g = nx.DiGraph()
    g.add_node(h)

    nbrs = sorted(neighbours.get(h, ()))[:MAX_NEIGHBOURS]

    # Green must mean "a true tail of the relation that was faked", not always
    # locatedin -- otherwise a neighbor fake shows regions in green and lies.
    true_tails = sorted(located.get(h, ())) if r == "locatedin" else nbrs

    for n in nbrs:
        if r == "locatedin":
            g.add_edge(h, n, kind="neighbor")
        for reg in sorted(located.get(n, ())):
            g.add_edge(n, reg, kind="context")

    for tt in true_tails:
        g.add_edge(h, tt, kind="true")

    # Where the centre itself lives -- context that makes a far-away fake obvious.
    if r == "neighbor":
        for reg in sorted(located.get(h, ())):
            g.add_edge(h, reg, kind="context")

    # For a neighbor fake, also draw the claimed neighbour's OWN neighbours --
    # two disconnected clusters is the visual tell that they are nowhere near.
    if r == "neighbor":
        for n in sorted(neighbours.get(t_fake, ()))[:MAX_NEIGHBOURS]:
            g.add_edge(t_fake, n, kind="neighbor")
            for reg in sorted(located.get(n, ())):
                g.add_edge(n, reg, kind="context")
        for reg in sorted(located.get(t_fake, ())):
            g.add_edge(t_fake, reg, kind="context")

    g.add_edge(h, t_fake, kind="fake")

    edge_colour = {"neighbor": "#b0b0b0", "context": "#8ab6d6",
                   "true": "#1e8449", "fake": "#c0392b"}
    ec = [edge_colour[g.edges[e]["kind"]] for e in g.edges]
    ew = [3.2 if g.edges[e]["kind"] in ("fake", "true") else 1.0 for e in g.edges]

    # Square = region, circle = country. Makes a type error visible at a glance.
    node_shape = {n: ("s" if n in regions else "o") for n in g.nodes}
    face = {}
    for n in g.nodes:
        if n == h:
            face[n] = "#fdebd0"
        elif n == t_fake:
            face[n] = "#f5b7b1"
        else:
            face[n] = "#eaeded"

    pos = nx.spring_layout(g, seed=42, k=1.1, iterations=200)
    plt.figure(figsize=(12, 8.5))
    for shape in ("o", "s"):
        sel = [n for n in g.nodes if node_shape[n] == shape]
        if sel:
            nx.draw_networkx_nodes(g, pos, nodelist=sel, node_shape=shape,
                                   node_color=[face[n] for n in sel],
                                   edgecolors="#566573", node_size=2000)
    nx.draw_networkx_edges(g, pos, edge_color=ec, width=ew, arrowsize=13,
                           node_size=2000)
    nx.draw_networkx_labels(g, pos, font_size=8)

    plt.title(f"[{kind.upper()}]  {h}  {r}  {t_fake}\n"
              f"red = injected   green = true {r}   square = region, circle = country"
              + (f"\n{title_note}" if title_note else ""), fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close()
    return out_path


def write_index(entries, out_path):
    """One scrollable page listing every ego image, so 100+ can be reviewed."""
    out_path = Path(out_path)
    rows = []
    for kind, h, r, t, png, note in entries:
        rows.append(
            f'<div class="card"><h3 class="{kind}">[{kind}] {h} &nbsp; {r} &nbsp; {t}</h3>'
            f'<p>{note}</p><img src="{Path(png).name}"></div>')
    html = f"""<meta charset="utf-8"><title>injected anomalies</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#fafafa}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:0 0 22px}}
.card img{{max-width:100%;height:auto;display:block}}
h3{{margin:0 0 4px;font-size:15px}} h3.type_invalid{{color:#b9770e}} h3.type_valid{{color:#922b21}}
p{{margin:0 0 8px;color:#555;font-size:13px}}
</style>
<h1>Injected anomalies ({len(entries)})</h1>
<p>Every picture below is a FAKE triple. Judge each one by eye.</p>
{''.join(rows)}"""
    out_path.write_text(html, encoding="utf-8")
    return out_path
