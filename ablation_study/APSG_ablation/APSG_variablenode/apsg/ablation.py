from __future__ import annotations
from typing import Dict, Any, Set

try:
    from .model import Graph, NodeCategory
    _HAS_GRAPH_TYPES = True
except Exception:
    _HAS_GRAPH_TYPES = False

def ablate_remove_variable_graph_dict(graph_dict: Dict[str, Any]) -> Dict[str, Any]:
    nodes = (graph_dict.get("V") or {}).get("nodes") or []
    edges = (graph_dict.get("E") or {}).get("edges") or []
    keep_nodes = [n for n in nodes if n.get("category") != "variable"]
    keep_ids: Set[str] = {n["id"] for n in keep_nodes}
    keep_edges = [e for e in edges if e.get("src") in keep_ids and e.get("dst") in keep_ids]
    meta = dict(graph_dict.get("meta") or {})
    meta["ablation"] = {"variables_removed": True}
    return {"meta": meta, "V": {"nodes": keep_nodes}, "E": {"edges": keep_edges}}

def ablate_remove_variable_graph(graph: "Graph") -> "Graph":
    if not _HAS_GRAPH_TYPES:
        raise RuntimeError("Graph types not available")
    keep_nodes = [n for n in graph.nodes if getattr(n.category, "value", None) != "variable"]
    keep_ids: Set[str] = {n.id for n in keep_nodes}
    from .model import Graph as G
    ng = G(meta=dict(graph.meta), nodes=[], edges=[])
    ng.meta["ablation"] = {"variables_removed": True}
    for n in keep_nodes:
        ng.add_node(n)
    for e in graph.edges:
        if e.src in keep_ids and e.dst in keep_ids:
            ng.add_edge(e)
    return ng
