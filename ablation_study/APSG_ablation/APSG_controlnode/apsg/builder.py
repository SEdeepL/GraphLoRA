from __future__ import annotations
from typing import List, Dict, Any, Optional
from .model import Graph, Node, Edge, NodeCategory, EdgeType
from .java_parser import MethodIR, parse_method
from .entropy import EntropyProvider, safe_entropy
from .edit_distance import assignment_component_edit_distance
from .anti_pattern import detect_dead_store_patch_vars

def build_apsg(java_patch_text: str, entropy_provider: Optional[EntropyProvider] = None) -> Graph:
    ir: MethodIR = parse_method(java_patch_text)
    g = Graph(meta={
        "language": "Java",
        "method": ir.name,
        "return_type": ir.ret_type,
        "parameters": ir.parameters,
        "scope": "intra-procedural (method)",
    })

    node_counter = {"S": 0, "V": 0, "E": 0}
    def nid(prefix: str) -> str:
        node_counter[prefix] += 1
        return f"{prefix}{node_counter[prefix]}"

    params_str = ", ".join([f"{p['type']} {p['name']}" for p in ir.parameters])
    entry = Node(id=nid("S"), category=NodeCategory.CONTEXT, label="[entry] parameters: " + params_str, attributes={
        "statement_type": "entry",
        "distance_to_patch": None,
    })
    g.add_node(entry)

    stmt_nodes: List[Node] = []
    removed_buffer: List[Dict[str, Any]] = []
    patch_nodes: List[Node] = []

    for st in ir.stmts:
        p = st.parsed
        if p.get("type") == "if":
            n = Node(id=nid("S"), category=NodeCategory.CONTROL, label=f"if ({p.get('cond')})", attributes={
                "control_type": "if",
                "is_nested": False,
            })
            g.add_node(n)
            stmt_nodes.append(n)
            continue

        if p.get("type") == "return":
            n = Node(id=nid("S"), category=NodeCategory.CONTEXT, label=f"return {p.get('expr')};", attributes={
                "statement_type": "return",
                "operator_type": None,
            })
            g.add_node(n)
            stmt_nodes.append(n)
            continue

        if p.get("type") == "assign":
            label = st.raw if st.raw.endswith(";") else (st.raw + ";")
            cat = NodeCategory.PATCH if st.kind == "added" else NodeCategory.CONTEXT
            n = Node(id=nid("S"), category=cat, label=label, attributes={
                "statement_type": "assignment",
                "operator_type": _operator_type_from_rhs(p.get("rhs")),
                "decl_type": p.get("decl_type"),
                "lhs_var": p.get("lhs"),
                "rhs_kind": p["rhs"].get("kind"),
                "rhs_func": p["rhs"].get("func"),
            })
            g.add_node(n)
            stmt_nodes.append(n)

            if st.kind == "removed":
                removed_buffer.append(p)

            if st.kind == "added":
                edit_dist = None
                edit_expl = None
                repair_action = "Add"
                if removed_buffer:
                    old = removed_buffer.pop(0)
                    edit_dist = assignment_component_edit_distance(old, p)
                    old_lhs = old.get("lhs"); old_type = old.get("decl_type")
                    new_lhs = p.get("lhs"); new_type = p.get("decl_type")
                    rhs_old = old.get("rhs", {}); rhs_new = p.get("rhs", {})
                    edit_expl = f"From '{old_type} {old_lhs} = {_rhs_str(rhs_old)}' -> '{new_type} {new_lhs} = {_rhs_str(rhs_new)}'"
                    repair_action = "Replace"
                entropy_val = safe_entropy(label, entropy_provider)
                n.attributes.update({
                    "edit_distance": edit_dist,
                    "edit_distance_explanation": edit_expl,
                    "patch_entropy": entropy_val,
                    "repair_action": repair_action,
                })
                patch_nodes.append(n)
            continue

        n = Node(id=nid("S"), category=NodeCategory.CONTEXT, label=st.raw, attributes={
            "statement_type": "unknown",
            "operator_type": None,
        })
        g.add_node(n)
        stmt_nodes.append(n)

    var_node_ids: Dict[str, str] = {}
    for sn in stmt_nodes:
        if sn.attributes.get("statement_type") == "assignment":
            decl_type = sn.attributes.get("decl_type") or "unknown"
            lhs = sn.attributes.get("lhs_var")
            v_id = nid("V")
            var_node_ids[(sn.id, lhs)] = v_id
            lhs_node = Node(id=v_id, category=NodeCategory.VARIABLE, label=lhs, attributes={
                "var_type": decl_type,
                "var_role": "AssignLHS",
            })
            g.add_node(lhs_node)
            g.add_edge(Edge(id=nid("E"), type=EdgeType.SUBGRAPH_MERGE, src=v_id, dst=sn.id, explanation="LHS variable anchors to its assignment statement"))

            rhs = _rhs_dict_from_sn(sn)
            for role, vname in rhs.get("vars_with_roles", []):
                rv_id = nid("V")
                rhs_node = Node(id=rv_id, category=NodeCategory.VARIABLE, label=vname, attributes={
                    "var_type": "unknown",
                    "var_role": role,
                })
                g.add_node(rhs_node)
                g.add_edge(Edge(id=nid("E"), type=EdgeType.DATA_FLOW, src=rv_id, dst=v_id, explanation=f"{vname} flows to {lhs}"))

    prev_if = None
    for sn in stmt_nodes:
        if sn.category == NodeCategory.CONTROL:
            prev_if = sn
            continue
        if prev_if is not None and sn.category in {NodeCategory.PATCH, NodeCategory.CONTEXT}:
            g.add_edge(Edge(id=nid("E"), type=EdgeType.CONTROL_FLOW, src=prev_if.id, dst=sn.id, explanation=f"{prev_if.label} controls {sn.label}"))
            prev_if = None

    patch_indices = {i for i, n in enumerate(stmt_nodes) if n.category == NodeCategory.PATCH}
    for i, sn in enumerate(stmt_nodes):
        if sn.category != NodeCategory.PATCH:
            sn.attributes["distance_to_patch"] = min((abs(i - j) for j in patch_indices), default=None)

    flags = detect_dead_store_patch_vars(g)
    for pn in patch_nodes:
        pn.attributes["anti_pattern"] = bool(flags.get(pn.id, False))

    return g

def _rhs_str(rhs: dict) -> str:
    k = rhs.get("kind")
    if k == "funcall":
        return f"{rhs.get('func')}({', '.join(rhs.get('args', []))})"
    if k == "binary":
        ops = rhs.get("operands", ['?','?'])
        return f"{ops[0]} {rhs.get('op')} {ops[1]}"
    if k == "atom":
        return rhs.get("value", "")
    return "?"

def _operator_type_from_rhs(rhs: dict):
    if not rhs:
        return None
    k = rhs.get("kind")
    if k == "binary":
        if rhs.get("op") == "/":
            return "binary-division"
        if rhs.get("op") == "+":
            return "binary-addition"
    return None

def _rhs_dict_from_sn(sn_node: Node):
    rhs_kind = sn_node.attributes.get("rhs_kind")
    out = {"vars_with_roles": []}
    label = sn_node.label
    if rhs_kind == "funcall":
        m = label.split("=", 1)[1].strip().rstrip(";")
        inside = m[m.find("(")+1:m.rfind(")")] if "(" in m and ")" in m else ""
        args = [a.strip() for a in inside.split(",") if a.strip()]
        for a in args:
            out["vars_with_roles"].append(("FuncArg", a))
    elif rhs_kind == "binary":
        rhs = label.split("=", 1)[1].strip().rstrip(";")
        if "/" in rhs:
            left, right = rhs.split("/", 1)
            out["vars_with_roles"].append(("MathOperatorLeft", left.strip()))
            out["vars_with_roles"].append(("MathOperatorRight", right.strip()))
        elif "+" in rhs:
            left, right = rhs.split("+", 1)
            out["vars_with_roles"].append(("MathOperatorLeft", left.strip()))
            out["vars_with_roles"].append(("MathOperatorRight", right.strip()))
    return out
