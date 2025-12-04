from __future__ import annotations
from typing import Dict, Any

def ablate_structure_only_graph_dict(graph_dict: Dict[str, Any]) -> Dict[str, Any]:
    nodes = (graph_dict.get("V") or {}).get("nodes") or []
    edges = (graph_dict.get("E") or {}).get("edges") or []
    new_nodes = [{"id": n.get("id"), "category": n.get("category"), "label": n.get("label")} for n in nodes]
    new_edges = [{"id": e.get("id"), "type": e.get("type"), "src": e.get("src"), "dst": e.get("dst")} for e in edges]
    meta = dict(graph_dict.get("meta") or {})
    meta["ablation"] = {"structure_only": True}
    return {"meta": meta, "V": {"nodes": new_nodes}, "E": {"edges": new_edges}}

def ablate_structure_only_graph_to_dict(graph_obj: Any) -> Dict[str, Any]:
    d = graph_obj.to_dict()
    return ablate_structure_only_graph_dict(d)
