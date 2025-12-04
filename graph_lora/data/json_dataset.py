# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    """
    Expects JSONL lines with at least: {"code": str, "graph": {...}, "label": 0/1}
    Optionally: "graph_text": str
    """
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                code = (obj.get("code") or "").strip()
                graph = obj.get("graph") or {}
                label = int(obj.get("label"))
                gtxt = obj.get("graph_text", None)
                self.items.append({"code": code, "graph": graph, "label": label, "graph_text": gtxt})
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

def collate_fn(batch, tokenizer, max_len: int, include_graph_text: bool=False):
    """
    Tokenize code (and optionally graph_text) with tokenizer.
    Returns dict for model forward() plus labels.
    """
    codes = [b["code"] for b in batch]
    if include_graph_text:
        gt = [b.get("graph_text") or "" for b in batch]
        merged = [c + "\n<graph>\n" + t + "\n</graph>" for c, t in zip(codes, gt)]
    else:
        merged = codes
    enc = tokenizer(merged, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    # graphs
    graphs = [b["graph"] for b in batch]
    # pack edges and node features
    node_feats = []
    edge_index = []
    edge_type = []
    n_nodes = []
    for g in graphs:
        nf = g.get("node_feat") or []
        edges = g.get("edges") or []
        n = len(nf)
        n_nodes.append(n)
        node_feats.append(nf)
        # edges as [src, dst, rel]
        edge_index.append([[e[0], e[1]] for e in edges])
        edge_type.append([e[2] for e in edges])
    labels = [int(b["label"]) for b in batch]
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "graphs": {
            "node_feat": node_feats,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "n_nodes": n_nodes,
            "n_rels": graphs[0].get("n_rels", 6) if graphs else 6
        },
        "labels": labels
    }
