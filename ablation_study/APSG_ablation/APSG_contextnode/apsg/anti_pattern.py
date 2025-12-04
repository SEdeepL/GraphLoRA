from __future__ import annotations
from typing import Dict, Set
from .model import Graph, NodeCategory

PURE_FUNCS: Set[str] = {"sqrt"}

def detect_dead_store_patch_vars(graph: Graph) -> Dict[str, bool]:
    stmt_nodes = [n for n in graph.nodes if n.category in {NodeCategory.PATCH, NodeCategory.CONTEXT, NodeCategory.CONTROL}]
    id_to_idx = {n.id: i for i, n in enumerate(stmt_nodes)}
    later_concat = {i: " ".join(sn.label for sn in stmt_nodes[i+1:]) for i in range(len(stmt_nodes))}

    flags = {}
    for n in graph.nodes:
        if n.category == NodeCategory.PATCH:
            lhs = n.attributes.get("lhs_var")
            idx = id_to_idx.get(n.id, -1)
            later_blob = later_concat.get(idx, "")
            used_after = (lhs is not None) and (f" {lhs}" in " " + later_blob)
            rhs_kind = n.attributes.get("rhs_kind")
            rhs_func = n.attributes.get("rhs_func")
            pure_rhs = (rhs_kind == "funcall" and rhs_func in PURE_FUNCS)
            flags[n.id] = (not used_after) and pure_rhs
    return flags
