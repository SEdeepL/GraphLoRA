from __future__ import annotations
import json
from pathlib import Path
from .model import Graph

def write_json_graph(graph: Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
