from __future__ import annotations
from typing import Dict, Any, Set

try:
    from .model import Graph, NodeCategory, EdgeType
    _HAS_GRAPH_TYPES = True
except Exception:
    _HAS_GRAPH_TYPES = False

def ablate_remove_control_graph_dict(graph_dict: Dict[str, Any]) -> Dict[str, Any]:
    nodes = (graph_dict.get("V") or {}).get("nodes") or []
    edges = (graph_dict.get("E") or {}).get("edges") or []
    keep_nodes = [n for n in nodes if n.get("category") != "control"]
    keep_ids: Set[str] = {n["id"] for n in keep_nodes}
    keep_types: Set[str] = {"data_flow", "subgraph_merge"}
    keep_edges = [e for e in edges if e.get("type") in keep_types and e.get("src") in keep_ids and e.get("dst") in keep_ids]
    meta = dict(graph_dict.get("meta") or {})
    meta["ablation"] = {"control_removed": True, "preserved_edges": ["data_flow", "subgraph_merge"]}
    return {"meta": meta, "V": {"nodes": keep_nodes}, "E": {"edges": keep_edges}}

def ablate_remove_control_graph(graph: "Graph") -> "Graph":
    if not _HAS_GRAPH_TYPES:
        raise RuntimeError("Graph types not available")
    keep_nodes = [n for n in graph.nodes if getattr(n.category, "value", None) != "control"]
    keep_ids: Set[str] = {n.id for n in keep_nodes}
    keep_edge_types = {EdgeType.DATA_FLOW, EdgeType.SUBGRAPH_MERGE}
    from .model import Graph as G
    ng = G(meta=dict(graph.meta), nodes=[], edges=[])
    ng.meta["ablation"] = {"control_removed": True, "preserved_edges": ["data_flow", "subgraph_merge"]}
    for n in keep_nodes:
        ng.add_node(n)
    for e in graph.edges:
        if e.type in keep_edge_types and e.src in keep_ids and e.dst in keep_ids:
            ng.add_edge(e)
    return ng
