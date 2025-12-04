from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

class NodeCategory(str, Enum):
    PATCH = "patch"
    CONTROL = "control"
    CONTEXT = "context"
    VARIABLE = "variable"
    ENTRY = "entry"

class EdgeType(str, Enum):
    CONTROL_FLOW = "control_flow"
    DATA_FLOW = "data_flow"
    SUBGRAPH_MERGE = "subgraph_merge"

@dataclass
class Node:
    id: str
    category: NodeCategory
    label: str
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Edge:
    id: str
    type: EdgeType
    src: str
    dst: str
    explanation: Optional[str] = None

@dataclass
class Graph:
    meta: Dict[str, Any] = field(default_factory=dict)
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": self.meta,
            "V": {
                "nodes": [
                    {
                        "id": n.id,
                        "category": n.category.value,
                        "label": n.label,
                        "attributes": n.attributes,
                    }
                    for n in self.nodes
                ]
            },
            "E": {
                "edges": [
                    {
                        "id": e.id,
                        "type": e.type.value,
                        "src": e.src,
                        "dst": e.dst,
                        "explanation": e.explanation,
                    }
                    for e in self.edges
                ]
            },
        }
