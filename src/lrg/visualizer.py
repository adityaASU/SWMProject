"""LRG Visualizer: renders a LRGGraph using Matplotlib + NetworkX layout."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from src.lrg.graph import LRGGraph
from src.lrg.nodes import EdgeType, NodeType

# Node type -> display colour
_NODE_COLORS = {
    NodeType.ENTITY: "#4A90D9",
    NodeType.ALIAS: "#7B4FD9",
    NodeType.ATTRIBUTE: "#5BAD6F",
    NodeType.FILTER: "#D96A4A",
    NodeType.AGGREGATION: "#D9A84A",
    NodeType.GROUPING: "#4AD9C9",
    NodeType.SUBGRAPH: "#D94A8C",
}

_EDGE_COLORS = {
    EdgeType.JOIN: "#333333",
    EdgeType.FILTER_OF: "#D96A4A",
    EdgeType.HAVING: "#D96A4A",
    EdgeType.AGG_OF: "#D9A84A",
    EdgeType.GROUP_BY: "#4AD9C9",
    EdgeType.SUBQUERY_OF: "#D94A8C",
    EdgeType.SELECTS: "#5BAD6F",
}


def render_lrg(
    lrg: LRGGraph,
    title: str = "Logical Reasoning Graph",
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 9),
) -> bytes:
    """Render *lrg* as a PNG and return the raw bytes.

    If *output_path* is provided, the image is also saved to disk.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    g = lrg._g  # access internal DiGraph

    if len(g.nodes) == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Empty LRG", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    # Node labels and colours
    node_labels: dict[str, str] = {}
    node_colors: list[str] = []
    for nid in g.nodes:
        node = lrg.get_node(nid)
        node_labels[nid] = node.label or nid
        node_colors.append(_NODE_COLORS.get(node.node_type, "#AAAAAA"))

    # Edge colours
    edge_colors: list[str] = []
    for src, tgt, data in g.edges(data=True):
        etype = data.get("edge_type", EdgeType.JOIN)
        edge_colors.append(_EDGE_COLORS.get(etype, "#888888"))

    # Layout
    try:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    except Exception:
        pos = nx.spring_layout(g, seed=42)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_axis_off()

    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=1800, alpha=0.92)
    nx.draw_networkx_labels(g, pos, labels=node_labels, ax=ax, font_size=7, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(
        g, pos, ax=ax,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
        width=1.8,
    )

    # Edge labels
    edge_label_map = {}
    for e in lrg.all_edges():
        if e.label:
            edge_label_map[(e.source_id, e.target_id)] = e.label[:30]
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_label_map, ax=ax, font_size=6)

    # Legend
    patches = [
        mpatches.Patch(color=color, label=ntype.value)
        for ntype, color in _NODE_COLORS.items()
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=7, framealpha=0.7)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(buf.getvalue())

    buf.seek(0)
    return buf.read()


def lrg_to_dot(lrg: LRGGraph) -> str:
    """Return a Graphviz DOT string for the LRG."""
    lines = ["digraph LRG {", '  rankdir=TB;', '  node [style=filled fontname="Helvetica"];']
    for nid, node in lrg._nodes.items():
        color = _NODE_COLORS.get(node.node_type, "#AAAAAA")
        label = (node.label or nid).replace('"', '\\"')
        lines.append(f'  "{nid}" [label="{label}" fillcolor="{color}" fontcolor=white];')
    for edge in lrg.all_edges():
        color = _EDGE_COLORS.get(edge.edge_type, "#888888")
        elabel = edge.label.replace('"', '\\"') if edge.label else ""
        lines.append(
            f'  "{edge.source_id}" -> "{edge.target_id}" '
            f'[label="{elabel}" color="{color}"];'
        )
    lines.append("}")
    return "\n".join(lines)
