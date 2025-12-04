# graph_dora/data/cached_dataset.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset

class CachedGraphTextDataset(Dataset):
    """
    读取 cache_dataset.py 生成的分片 .pt
    meta.json:
      {
        "total": int, "shard_size": int, "model_id": str, "max_len": int,
        "graph_in_dim": int,
        "shards": [{"file":"shard_00000.pt","count":2048}, ...]
      }
    每个分片是 List[Dict]:
      {
        "input_ids": LongTensor[T],
        "attention_mask": LongTensor[T],
        "label": int,
        "graph": {
          "node_feat": FloatTensor[Ni, D],
          "edge_index": LongTensor[2, Ei],
          "edge_type": LongTensor[Ei],
          "n_nodes": int
        }
      }
    """
    def __init__(self, cache_dir: str):
        super().__init__()
        self.cache_dir = cache_dir
        with open(os.path.join(cache_dir, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.shards = self.meta["shards"]
        self._prefix = []
        c = 0
        for s in self.shards:
            self._prefix.append(c)
            c += s["count"]
        self.total = c
        self._loaded_idx = None
        self._loaded_data: List[Dict[str, Any]] | None = None

    def __len__(self):
        return self.total

    def _load_shard(self, si: int):
        if self._loaded_idx == si and self._loaded_data is not None:
            return
        path = os.path.join(self.cache_dir, self.shards[si]["file"])
        self._loaded_data = torch.load(path, map_location="cpu")
        self._loaded_idx = si

    def _locate(self, idx: int) -> Tuple[int, int]:
        # 二分也可；直接线性扫描够快（分片数通常不多）
        for si in range(len(self.shards)-1, -1, -1):
            if idx >= self._prefix[si]:
                return si, idx - self._prefix[si]
        return 0, idx

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        si, off = self._locate(idx)
        self._load_shard(si)
        return self._loaded_data[off]

def collate_cached(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 文本：直接 stack（CPU）
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0).long()
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0).long()
    labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)

    # 图：按样本拼接
    node_feats, edge_indices, edge_types, n_nodes = [], [], [], []
    node_offset = 0
    for b in batch:
        g = b["graph"]
        nf = g["node_feat"].float()
        ei = g["edge_index"].long()
        et = g["edge_type"].long()
        n = int(g["n_nodes"])

        if ei.numel() > 0:
            ei = ei + node_offset  # 偏移
        node_offset += n

        node_feats.append(nf)
        edge_indices.append(ei)
        edge_types.append(et)
        n_nodes.append(n)

    node_feat = torch.cat(node_feats, dim=0) if node_feats else torch.zeros(0, 1)
    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros(2, 0, dtype=torch.long)
    edge_type = torch.cat(edge_types, dim=0) if edge_types else torch.zeros(0, dtype=torch.long)

    graphs = {
        "node_feat": node_feat,     # [sum_N, D] float32
        "edge_index": edge_index,   # [2, sum_E] int64
        "edge_type": edge_type,     # [sum_E]    int64
        "n_nodes": n_nodes,         # List[int]  每条样本节点数
        # "n_rels" 可选：如果你模型里需要固定 n_rels，可在这里补充
    }

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "graphs": graphs,
        "labels": labels,
    }
