"""Ego-network view of ONE corruption: the 2-hop neighbourhoods of the head and
the tail, with the changed edge drawn on top.

Feed the ORIGINAL triple and its CORRUPTED counterpart (dataset string IDs --
the same ones that appear in train.txt) and get a figure showing where the
corruption lands relative to the real neighbourhood: whether the replacement
entity sits inside the head/tail's 2-hop context or outside it entirely.

Reading the figure:
    green solid   the original (true) edge
    red dashed    the corrupted edge -- the ONE slot that changed
    blue node     head          green node   true tail
    red node      the replacement entity that was substituted in
    grey nodes    1-hop (darker) and 2-hop (lighter) neighbours

Usage (repo root, PYTHONPATH=experiments):
    python -m kgsage.cli.ego_viz \
        --data data/FB15K-237 \
        --orig /m/027rn /location/country/form_of_government /m/06cx9 \
        --corr /m/027rn /location/country/form_of_government /m/01d_h8 \
        --out reports/ego.pdf

Knobs:
    --hops N            neighbourhood depth per anchor (default 2)
    --max-neighbors N   cap expanded neighbours per node (default 6) -- FB15K-237
                        hubs have hundreds of edges; without a cap the figure is
                        an unreadable hairball
    --label-hops N      label nodes up to this hop (default 1; -1 = label all)
    --edge-labels       also print relation names on every edge (cluttered)
    --seed / --figsize  layout determinism / canvas size
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# --- palette (matches the KGSAGE figures) ---
C_HEAD = "#6c8ebf"      # anchor: head
C_TAIL = "#82b366"      # anchor: true tail
C_CORR = "#b85450"      # the substituted entity
C_HOP1 = "#c9c9c9"
C_HOP2 = "#ebebeb"
C_EDGE = "#d0d0d0"
C_TRUE_EDGE = "#2d7a2d"
C_CORR_EDGE = "#c0392b"


def _read_text_map(path: Path) -> dict:
    """id -> readable name (entity2text.txt / relation2text.txt: 'id<TAB>text')."""
    m = {}
    if not path.exists():
        return m
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                m[parts[0]] = parts[1]
    return m


def _load_triples(data_dir: Path) -> list:
    """All (h, r, t) across train/valid/test -- the full graph the KG describes."""
    triples = []
    for split in ("train", "valid", "test"):
        p = data_dir / f"{split}.txt"
        if not p.exists():
            continue
        with open(p, encoding="utf-8-sig") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    triples.append(tuple(parts))
    return triples


def _build_adjacency(triples) -> dict:
    """node -> list of (neighbour, relation, outgoing?). Undirected for traversal,
    but each entry remembers the true direction so edges are drawn correctly."""
    adj = {}
    for h, r, t in triples:
        adj.setdefault(h, []).append((t, r, True))
        adj.setdefault(t, []).append((h, r, False))
    return adj


def _ego(adj, center, hops, max_neighbors, rng, always_keep=()):
    """BFS out to `hops` from `center`, capping expansion per node.

    Returns (hop_of_node, edges) where edges are (src, dst, relation) in true
    direction. `always_keep` nodes are never dropped by the cap -- that is what
    guarantees the true tail / replacement entity stay visible if they are in
    range at all.
    """
    hop = {center: 0}
    edges = []
    q = deque([center])
    while q:
        node = q.popleft()
        d = hop[node]
        if d >= hops:
            continue
        nbrs = adj.get(node, [])
        if len(nbrs) > max_neighbors:
            keep = [x for x in nbrs if x[0] in always_keep]
            rest = [x for x in nbrs if x[0] not in always_keep]
            rng.shuffle(rest)
            nbrs = keep + rest[:max(0, max_neighbors - len(keep))]
        for nbr, rel, outgoing in nbrs:
            edges.append((node, nbr, rel) if outgoing else (nbr, node, rel))
            if nbr not in hop:
                hop[nbr] = d + 1
                q.append(nbr)
    return hop, edges


def _short(text: str, n: int = 26) -> str:
    text = text.strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _structured_layout(G, head, tail, corr):
    """Deterministic layout built for THIS figure's argument.

    A spring layout is unusable here: the anchors are mutually adjacent so they
    collapse into one blob while degree-1 neighbours get flung to the edges,
    which hides the very edges the figure exists to show. Instead: pin the three
    anchors far apart, assign every other node to its nearest anchor, and fan it
    out on an arc that points AWAY from the middle -- so the true edge and the
    corrupted edge always run through open space.
    """
    import math

    anchors = {head: (-4.0, 0.4), tail: (4.0, 2.2)}
    if corr and corr not in anchors:
        anchors[corr] = (4.0, -2.6)
    # Arc each anchor's neighbours into the open space that anchor faces, so the
    # green/red edges across the middle stay clear. Wide arcs = fewer collisions.
    sector = {head: (100.0, 260.0), tail: (-40.0, 120.0)}
    if corr in anchors:
        sector[corr] = (-120.0, 40.0)

    U = G.to_undirected(as_view=True)
    dist = {a: nx.single_source_shortest_path_length(U, a) for a in anchors}

    buckets = {}
    for n in G.nodes():
        if n in anchors:
            continue
        best_a, best_d = None, 10**9
        for a in anchors:                       # nearest anchor owns the node
            d = dist[a].get(n, 10**9)
            if d < best_d:
                best_a, best_d = a, d
        if best_a is None or best_d >= 10**9:   # unreachable -> park under head
            best_a, best_d = head, 3
        buckets.setdefault(best_a, {}).setdefault(best_d, []).append(n)

    pos = dict(anchors)
    for a, byhop in buckets.items():
        a0, a1 = sector.get(a, (0.0, 360.0))
        for d, nodes in sorted(byhop.items()):
            nodes = sorted(nodes)
            r = 1.5 + 1.35 * (d - 1)
            for i, n in enumerate(nodes):
                frac = (i + 0.5) / len(nodes)
                ang = math.radians(a0 + (a1 - a0) * frac)
                pos[n] = (anchors[a][0] + r * math.cos(ang),
                          anchors[a][1] + r * math.sin(ang))
    return pos


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset dir with train/valid/test.txt")
    ap.add_argument("--orig", nargs=3, required=True, metavar=("H", "R", "T"),
                    help="the ORIGINAL true triple")
    ap.add_argument("--corr", nargs=3, required=True, metavar=("H", "R", "T"),
                    help="the CORRUPTED triple (one slot differs)")
    ap.add_argument("--out", default="ego.pdf", help="output figure path (.pdf/.png)")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--corr-hops", type=int, default=1,
                    help="also draw the REPLACEMENT entity's own ego network to this "
                         "depth (0 = off, just the bare node). Fills the space around "
                         "it and answers the real question: does the replacement's "
                         "context overlap the head's (corroborating the corruption) "
                         "or not (contradicting it)?")
    ap.add_argument("--max-neighbors", type=int, default=6)
    ap.add_argument("--label-hops", type=int, default=1,
                    help="label nodes up to this hop (-1 = all)")
    ap.add_argument("--edge-labels", action="store_true")
    ap.add_argument("--short-relations", action="store_true",
                    help="label relations with the LAST path segment only "
                         "(/people/person/place_of_birth -> place_of_birth). "
                         "Freebase relation text is far too long to print on every "
                         "edge; pair this with --edge-labels to keep them readable")
    ap.add_argument("--layout", choices=["structured", "spring"], default="structured",
                    help="structured (default) pins the anchors and fans neighbours "
                         "into open space; spring is the plain force-directed fallback")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figsize", default="14,9")
    ap.add_argument("--entity2text", default=None)
    ap.add_argument("--relation2text", default=None)
    args = ap.parse_args()

    data = Path(args.data)
    rng = random.Random(args.seed)

    ent_txt = _read_text_map(Path(args.entity2text) if args.entity2text
                             else data / "entity2text.txt")
    rel_txt = _read_text_map(Path(args.relation2text) if args.relation2text
                             else data / "relation2text.txt")
    name = lambda e: _short(ent_txt.get(e, e))

    def rname(r):
        """Readable relation label. Freebase paths are long, so --short-relations
        keeps only the trailing segment, which carries the actual meaning
        (/people/person/place_of_birth -> place_of_birth)."""
        if args.short_relations:
            return _short(r.rstrip("/").split("/")[-1] or r, 24)
        return _short(rel_txt.get(r, r), 34)

    triples = _load_triples(data)
    if not triples:
        print(f"!! no triples under {data}", file=sys.stderr)
        return 2
    adj = _build_adjacency(triples)

    oh, orr, ot = args.orig
    ch, cr, ct = args.corr

    # Which slot moved?
    changed = [s for s, (a, b) in
               zip(("head", "relation", "tail"), zip(args.orig, args.corr)) if a != b]
    if not changed:
        print("!! original and corrupted triple are identical", file=sys.stderr)
        return 2
    slot = changed[0]
    new_ent = ch if slot == "head" else (ct if slot == "tail" else None)

    for e in (oh, ot):
        if e not in adj:
            print(f"!! entity {e} not found in {data}", file=sys.stderr)
            return 2

    # Ego nets of the ORIGINAL head and tail (union), keeping the key nodes.
    keep = {oh, ot} | ({new_ent} if new_ent else set())
    hop_h, edges_h = _ego(adj, oh, args.hops, args.max_neighbors, rng, keep)
    hop_t, edges_t = _ego(adj, ot, args.hops, args.max_neighbors, rng, keep)

    # Is the replacement inside the head/tail context? Decide this BEFORE folding
    # in the replacement's own ego net, otherwise it trivially finds itself.
    anchor_hop = min(hop_h.get(new_ent, 99), hop_t.get(new_ent, 99)) if new_ent else 99
    corr_in_ego = anchor_hop < 99

    hop_c, edges_c = {}, []
    if new_ent and args.corr_hops > 0:
        hop_c, edges_c = _ego(adj, new_ent, args.corr_hops, args.max_neighbors, rng, keep)

    hop = dict(hop_t)
    for src in (hop_h, hop_c):                    # nearest anchor wins
        for n, d in src.items():
            hop[n] = min(d, hop.get(n, 99))

    G = nx.DiGraph()
    for n, d in hop.items():
        G.add_node(n, hop=d)
    for s, d, r in edges_h + edges_t + edges_c:
        if s in hop and d in hop:
            G.add_edge(s, d, rel=r)

    # The replacement entity + the changed edge (added even if far outside the
    # neighbourhood -- that isolation is itself the finding).
    if new_ent and new_ent not in G:
        G.add_node(new_ent, hop=99)
    G.add_edge(ch, ct, rel=cr)
    G.add_edge(oh, ot, rel=orr)

    # ---- layout ----
    if args.layout == "spring":
        init = {oh: (-1.0, 0.0), ot: (1.0, 0.0)}
        if new_ent and new_ent not in init:
            init[new_ent] = (1.0, -1.1)
        pos = nx.spring_layout(G, pos=init, fixed=list(init), seed=args.seed,
                               k=0.55, iterations=200)
    else:
        pos = _structured_layout(G, oh, ot, new_ent)

    w, h = (float(x) for x in args.figsize.split(","))
    fig, ax = plt.subplots(figsize=(w, h))
    ax.axis("off")

    true_edge, corr_edge = (oh, ot), (ch, ct)
    plain = [e for e in G.edges() if e not in (true_edge, corr_edge)]
    nx.draw_networkx_edges(G, pos, edgelist=plain, edge_color=C_EDGE,
                           width=1.0, arrows=False, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=[true_edge], edge_color=C_TRUE_EDGE,
                           width=3.0, arrows=True, arrowsize=20, min_source_margin=18,
                           min_target_margin=18, connectionstyle="arc3,rad=0.06", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=[corr_edge], edge_color=C_CORR_EDGE,
                           width=3.0, style="dashed", arrows=True, arrowsize=20,
                           min_source_margin=18, min_target_margin=18,
                           connectionstyle="arc3,rad=0.16", ax=ax)

    def draw(nodes, color, size, edge="#888888", lw=1.0):
        nodes = [n for n in nodes if n in G]
        if nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color,
                                   node_size=size, edgecolors=edge, linewidths=lw, ax=ax)

    anchors = {oh, ot} | ({new_ent} if new_ent else set())
    draw([n for n, d in hop.items() if d >= 2 and n not in anchors], C_HOP2, 240)
    draw([n for n, d in hop.items() if d == 1 and n not in anchors], C_HOP1, 340)
    draw([oh], C_HEAD, 1100, "#31506f", 2.0)
    draw([ot], C_TAIL, 1100, "#4a7a3a", 2.0)
    if new_ent:
        draw([new_ent], C_CORR, 1100, "#7d2f28", 2.0)

    # Labels sit BELOW their node so they never sit on top of the key edges.
    lab_nodes = (list(G.nodes()) if args.label_hops < 0
                 else [n for n in G.nodes() if hop.get(n, 99) <= args.label_hops])
    for n in set(lab_nodes) | anchors:
        if n not in pos:
            continue
        big = n in anchors
        ax.text(pos[n][0], pos[n][1] - (0.30 if big else 0.20), name(n),
                ha="center", va="top", fontsize=8.0 if big else 6.8,
                fontweight="bold" if big else "normal", zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))

    key = {true_edge: rname(orr)}
    if corr_edge != true_edge:
        key[corr_edge] = rname(cr)
    if args.edge_labels:
        key = {(s, d): rname(a["rel"]) for s, d, a in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=key, font_size=6.5,
                                 bbox=dict(fc="white", ec="none", alpha=0.75), ax=ax)

    where = ("inside the neighbourhood (hop %d)" % hop[new_ent] if corr_in_ego
             else "OUTSIDE the %d-hop neighbourhood" % args.hops) if new_ent else "n/a"
    ax.set_title(
        f"{slot} corrupted:  {name(oh)} —[{rname(orr)}]→ {name(ot)}\n"
        f"replacement: {name(new_ent) if new_ent else '-'}  ·  {where}"
        f"   |   {G.number_of_nodes()} nodes, {args.hops}-hop ego of head+tail"
        f" (≤{args.max_neighbors} nbrs/node)",
        fontsize=10)

    handles = [
        plt.Line2D([], [], color=C_TRUE_EDGE, lw=2.8, label="original (true) edge"),
        plt.Line2D([], [], color=C_CORR_EDGE, lw=2.8, ls="--", label="corrupted edge"),
        plt.Line2D([], [], marker="o", ls="", mfc=C_HEAD, mec="#31506f", ms=10, label="head"),
        plt.Line2D([], [], marker="o", ls="", mfc=C_TAIL, mec="#4a7a3a", ms=10, label="true tail"),
        plt.Line2D([], [], marker="o", ls="", mfc=C_CORR, mec="#7d2f28", ms=10, label="replacement"),
        plt.Line2D([], [], marker="o", ls="", mfc=C_HOP1, mec="#888", ms=8, label="1-hop"),
        plt.Line2D([], [], marker="o", ls="", mfc=C_HOP2, mec="#888", ms=7, label="2-hop"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)

    out = Path(args.out)
    if out.parent and str(out.parent) != "":
        out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=180)

    print(f"slot changed      : {slot}")
    print(f"original          : ({name(oh)}, {rname(orr)}, {name(ot)})")
    print(f"corrupted         : ({name(ch)}, {rname(cr)}, {name(ct)})")
    print(f"replacement entity: {name(new_ent) if new_ent else '-'}  -> {where}")
    print(f"degree(head)={len(adj.get(oh, []))}  degree(true tail)={len(adj.get(ot, []))}"
          f"  degree(replacement)={len(adj.get(new_ent, [])) if new_ent else 0}")
    if new_ent:
        # Context overlap: neighbours shared by the head and each candidate filler.
        # If the replacement shares as many as the true tail does, the corruption is
        # CORROBORATED by the neighbourhood, not contradicted by it.
        nb = lambda e: {x[0] for x in adj.get(e, [])}
        shared_true, shared_corr = nb(oh) & nb(ot), nb(oh) & nb(new_ent)
        print(f"shared neighbours with head: true tail={len(shared_true)}  "
              f"replacement={len(shared_corr)}")
    print(f"figure            : {out}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
