from __future__ import annotations
from typing import Dict, Any, Set

try:
    from .model import Graph, NodeCategory
    _HAS_GRAPH_TYPES = True
except Exception:
    _HAS_GRAPH_TYPES = False

def ablate_context_graph_dict(graph_dict: Dict[str, Any]) -> Dict[str, Any]:
    nodes = (graph_dict.get("V") or {}).get("nodes") or []
    edges = (graph_dict.get("E") or {}).get("edges") or []
    keep_ids = {n["id"] for n in nodes if n.get("category") == "patch"}
    new_nodes = [n for n in nodes if n["id"] in keep_ids]
    new_edges = [e for e in edges if e.get("src") in keep_ids and e.get("dst") in keep_ids]
    meta = dict(graph_dict.get("meta") or {})
    meta["ablation"] = {"context_removed": True, "kept_categories": ["patch"]}
    return {"meta": meta, "V": {"nodes": new_nodes}, "E": {"edges": new_edges}}

def ablate_context_graph(graph: "Graph") -> "Graph":
    if not _HAS_GRAPH_TYPES:
        raise RuntimeError("Graph types not available")
    keep = [n for n in graph.nodes if getattr(n.category, "value", None) == "patch"]
    keep_ids = {n.id for n in keep}
    from .model import Graph as G
    ng = G(meta=dict(graph.meta), nodes=[], edges=[])
    ng.meta["ablation"] = {"context_removed": True, "kept_categories": ["patch"]}
    for n in keep:
        ng.add_node(n)
    for e in graph.edges:
        if e.src in keep_ids and e.dst in keep_ids:
            ng.add_edge(e)
    return ng
