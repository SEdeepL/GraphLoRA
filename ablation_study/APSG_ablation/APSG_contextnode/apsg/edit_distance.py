from __future__ import annotations
from typing import Dict

def assignment_component_edit_distance(old: Dict, new: Dict) -> int:
    dist = 0
    if old.get("decl_type") != new.get("decl_type"):
        dist += 1
    if old.get("lhs") != new.get("lhs"):
        dist += 1
    old_rhs = old.get("rhs", {})
    new_rhs = new.get("rhs", {})
    if old_rhs.get("func") != new_rhs.get("func"):
        dist += 1
    if old_rhs.get("op") != new_rhs.get("op"):
        dist += 1
    old_args = old_rhs.get("args") or old_rhs.get("operands") or []
    new_args = new_rhs.get("args") or new_rhs.get("operands") or []
    n = max(len(old_args), len(new_args))
    for i in range(n):
        o = old_args[i] if i < len(old_args) else None
        v = new_args[i] if i < len(new_args) else None
        if o != v:
            dist += 1
    return dist
