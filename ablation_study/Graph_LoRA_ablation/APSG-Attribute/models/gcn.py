# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_rels: int, bias: bool = True):
        super().__init__()
        self.num_rels = num_rels
        self.weight = nn.Parameter(torch.empty(num_rels, in_dim, out_dim))
        self.self_loop = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.self_loop.weight)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        h: [N, Din]; edge_index: [2,E] or [E,2]; edge_type: [E]; return [N, Dout]
        """
        if edge_index.dim() != 2:
            raise ValueError("edge_index must be [2,E] or [E,2]")
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()  # [2,E]
        src, dst = edge_index[0], edge_index[1]
        E = edge_index.shape[1]
        out_dim = self.self_loop.out_features

        # message passing per relation: m_e = h[src] @ W_r
        m = torch.zeros((num_nodes, out_dim), device=h.device, dtype=h.dtype)
        for r in range(self.num_rels):
            mask = (edge_type == r)
            if mask.any():
                src_r = src[mask]
                dst_r = dst[mask]
                h_src = h.index_select(0, src_r)          # [Er, Din]
                W = self.weight[r]                         # [Din, Dout]
                msg = h_src @ W                            # [Er, Dout]
                # aggregate by sum into dst
                m.index_add_(0, dst_r, msg)

        # simple degree normalization on dst
        deg = torch.zeros(num_nodes, device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        deg = deg.clamp_min(1.0).unsqueeze(-1)
        m = m / deg

        return self.self_loop(h) + m


class RGCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_rels: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + [hid_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            RGCNLayer(dims[i], dims[i+1], num_rels=num_rels) for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor,
                n_nodes: List[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        node_feat: [N, Din] (任意精度)，内部统一转 float32 计算，再转回输入精度
        edge_index: [2,E] or [E,2] (long)
        edge_type:  [E] (long)
        n_nodes: list[int] per-sample node counts
        Returns:
            H: [N, Dout] node embeddings
            G: [B, Dout] graph pooled embeddings (mean)
        """
        device = node_feat.device
        in_dtype = node_feat.dtype
        x = node_feat.to(torch.float32)

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type, num_nodes=x.size(0))
            if i != len(self.layers) - 1:
                x = self.act(x)
                x = self.dropout(x)

        H = x.to(in_dtype)

        # per-graph pooling
        slices = [0]
        s = 0
        for n in n_nodes:
            s += n
            slices.append(s)
        pools = []
        for i in range(len(n_nodes)):
            s, e = slices[i], slices[i+1]
            if e > s:
                pools.append(H[s:e].mean(dim=0, keepdim=True))
            else:
                pools.append(torch.zeros(1, H.size(-1), device=device, dtype=H.dtype))
        G = torch.cat(pools, dim=0)  # [B, Dout]
        return H, G
