# -*- coding: utf-8 -*-
"""7.4 ego graphs: render one ego-network figure per corruption in a CSV.

Reads a CSV produced by gen_corruptions_csv.py and draws, for each row, the
2-hop neighbourhoods of the head and the tail with the changed edge on top --
so the figure shows whether the replacement sits inside the entity's context or
outside it entirely. The dataset graph is loaded ONCE and reused across rows.

By default renders the clearest exemplars first (tail-slot, zero shared
neighbours, moderate anchor degree so the graph stays legible), capped by
--limit.

Reading a figure:
    green solid   the original (true) edge      blue node   head
    red dashed    the corrupted edge            green node  true tail
    grey nodes    1-hop (darker) / 2-hop        red node    the replacement

Run from repo root (pytorch env):
  PYTHONPATH=experiments python experiments/kgsage/cli/ego_from_csv.py \
      --csv experiments/kgsage/outputs/eval/fb_corruptions.csv \
      --data data/FB15K-237 --out_dir experiments/kgsage/outputs/eval/ego --limit 6
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

# Each eval script writes into its own subfolder under outputs/eval/, resolved
# relative to this file so the location is correct regardless of cwd.
_EVAL_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "eval"

# --- palette (matches the KGSAGE figures) ---
C_HEAD = "#6c8ebf"
C_TAIL = "#82b366"
C_CORR = "#b85450"
C_HOP1 = "#c9c9c9"
C_HOP2 = "#ebebeb"
C_EDGE = "#d0d0d0"
C_TRUE_EDGE = "#2d7a2d"
C_CORR_EDGE = "#c0392b"


# --------------------------------------------------------------------------
# graph helpers (loaded once, reused for every row)
# --------------------------------------------------------------------------

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
    """BFS out to `hops` from `center`, capping expansion per node. Returns
    (hop_of_node, edges); `always_keep` nodes are never dropped by the cap."""
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
    """Pin the three anchors far apart, assign every other node to its nearest
    anchor, and fan it onto an arc pointing away from the middle -- so the true
    edge and the corrupted edge always run through open space."""
    anchors = {head: (-4.0, 0.4), tail: (4.0, 2.2)}
    if corr and corr not in anchors:
        anchors[corr] = (4.0, -2.6)
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
        for a in anchors:
            d = dist[a].get(n, 10**9)
            if d < best_d:
                best_a, best_d = a, d
        if best_a is None or best_d >= 10**9:
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


def render_ego(adj, ent_txt, rel_txt, orig, corr, out, *, hops=2, corr_hops=1,
               max_neighbors=6, label_hops=1, edge_labels=False,
               short_relations=False, layout="structured", seed=0,
               figsize=(14, 9)):
    """Draw ONE corruption's ego figure. `adj`/`ent_txt`/`rel_txt` are the
    once-loaded graph + text maps. `orig`/`corr` are (h, r, t) string triples.
    Returns a stats dict (or None if the triple could not be drawn)."""
    rng = random.Random(seed)
    name = lambda e: _short(ent_txt.get(e, e))

    def rname(r):
        if short_relations:
            return _short(r.rstrip("/").split("/")[-1] or r, 24)
        return _short(rel_txt.get(r, r), 34)

    oh, orr, ot = orig
    ch, cr, ct = corr
    changed = [s for s, (a, b) in
               zip(("head", "relation", "tail"), zip(orig, corr)) if a != b]
    if not changed:
        return None
    slot = changed[0]
    new_ent = ch if slot == "head" else (ct if slot == "tail" else None)
    if oh not in adj or ot not in adj:
        return None

    keep = {oh, ot} | ({new_ent} if new_ent else set())
    hop_h, edges_h = _ego(adj, oh, hops, max_neighbors, rng, keep)
    hop_t, edges_t = _ego(adj, ot, hops, max_neighbors, rng, keep)

    anchor_hop = min(hop_h.get(new_ent, 99), hop_t.get(new_ent, 99)) if new_ent else 99
    corr_in_ego = anchor_hop < 99

    hop_c, edges_c = {}, []
    if new_ent and corr_hops > 0:
        hop_c, edges_c = _ego(adj, new_ent, corr_hops, max_neighbors, rng, keep)

    hop = dict(hop_t)
    for src in (hop_h, hop_c):
        for n, d in src.items():
            hop[n] = min(d, hop.get(n, 99))

    G = nx.DiGraph()
    for n, d in hop.items():
        G.add_node(n, hop=d)
    for s, d, r in edges_h + edges_t + edges_c:
        if s in hop and d in hop:
            G.add_edge(s, d, rel=r)
    if new_ent and new_ent not in G:
        G.add_node(new_ent, hop=99)
    G.add_edge(ch, ct, rel=cr)
    G.add_edge(oh, ot, rel=orr)

    if layout == "spring":
        init = {oh: (-1.0, 0.0), ot: (1.0, 0.0)}
        if new_ent and new_ent not in init:
            init[new_ent] = (1.0, -1.1)
        pos = nx.spring_layout(G, pos=init, fixed=list(init), seed=seed,
                               k=0.55, iterations=200)
    else:
        pos = _structured_layout(G, oh, ot, new_ent)

    w, h = figsize
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

    lab_nodes = (list(G.nodes()) if label_hops < 0
                 else [n for n in G.nodes() if hop.get(n, 99) <= label_hops])
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
    if edge_labels:
        key = {(s, d): rname(a["rel"]) for s, d, a in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=key, font_size=6.5,
                                 bbox=dict(fc="white", ec="none", alpha=0.75), ax=ax)

    where = ("inside the neighbourhood (hop %d)" % hop[new_ent] if corr_in_ego
             else "OUTSIDE the %d-hop neighbourhood" % hops) if new_ent else "n/a"
    ax.set_title(
        f"{slot} corrupted:  {name(oh)} —[{rname(orr)}]→ {name(ot)}\n"
        f"replacement: {name(new_ent) if new_ent else '-'}  ·  {where}"
        f"   |   {G.number_of_nodes()} nodes, {hops}-hop ego of head+tail"
        f" (≤{max_neighbors} nbrs/node)",
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

    out = Path(out)
    if out.parent and str(out.parent) != "":
        out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=180)
    plt.close(fig)

    nb = lambda e: {x[0] for x in adj.get(e, [])}
    shared_true = len(nb(oh) & nb(ot)) if new_ent else 0
    shared_corr = len(nb(oh) & nb(new_ent)) if new_ent else 0
    return {"slot": slot, "corr_in_ego": corr_in_ego,
            "shared_true": shared_true, "shared_corr": shared_corr,
            "nodes": G.number_of_nodes(), "edges": G.number_of_edges()}


# --------------------------------------------------------------------------
# CSV driver
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", default=None,
                    help="output dir for the PNGs; default: outputs/eval/ego_graphs/")
    ap.add_argument("--limit", type=int, default=6,
                    help="max ego graphs to render (0 = all rows)")
    ap.add_argument("--all_rows", action="store_true",
                    help="render every row in file order instead of ranking exemplars")
    ap.add_argument("--deg_min", type=int, default=6)
    ap.add_argument("--deg_max", type=int, default=45)
    ap.add_argument("--ext", default="png", choices=["png", "pdf"])
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--max_neighbors", type=int, default=6)
    ap.add_argument("--entity2text", default=None)
    ap.add_argument("--relation2text", default=None)
    ap.add_argument("--edge_labels", action="store_true", default=True)
    ap.add_argument("--short_relations", action="store_true", default=True)
    args = ap.parse_args()

    data = Path(args.data)
    ent_txt = _read_text_map(Path(args.entity2text) if args.entity2text
                             else data / "entity2text.txt")
    rel_txt = _read_text_map(Path(args.relation2text) if args.relation2text
                             else data / "relation2text.txt")

    triples = _load_triples(data)
    if not triples:
        print(f"!! no triples under {data}", file=sys.stderr)
        return 2
    adj = _build_adjacency(triples)          # built ONCE, reused for every row

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))
    if not args.all_rows:
        def rank_key(r):
            zero = (int(r["shared_neighbours"]) == 0 and r["direct_neighbour"] == "0")
            deg = int(r["anchor_degree"])
            legible = args.deg_min <= deg <= args.deg_max
            return (r["slot"] != "tail", not zero, not legible, deg)
        rows = sorted(rows, key=rank_key)
    if args.limit:
        rows = rows[:args.limit]

    out_dir = Path(args.out_dir) if args.out_dir else (_EVAL_ROOT / "ego_graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for r in rows:
        stem = f"ego_{r['idx']}_{r['relation'].split('/')[-1]}"
        out = out_dir / f"{stem}.{args.ext}"
        stats = render_ego(
            adj, ent_txt, rel_txt,
            (r["orig_h_id"], r["orig_r_id"], r["orig_t_id"]),
            (r["corr_h_id"], r["corr_r_id"], r["corr_t_id"]),
            out, hops=args.hops, max_neighbors=args.max_neighbors,
            edge_labels=args.edge_labels, short_relations=args.short_relations)
        if stats is None:
            print(f"SKIP {stem}: could not draw (identical triple or missing entity)",
                  file=sys.stderr)
            continue
        ok += 1
        print(f"OK  {out}   {r['corr_statement']}   "
              f"[shared true={stats['shared_true']} repl={stats['shared_corr']}]")

    print(f"\nrendered {ok}/{len(rows)} ego graphs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
