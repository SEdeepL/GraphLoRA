# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """
    hidden: [B,T,H], mask: [B,T] (1 for valid).
    """
    if mask is None:
        return hidden.mean(dim=1)
    mask = mask.to(hidden.dtype)
    s = (hidden * mask.unsqueeze(-1)).sum(dim=1)
    d = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
    return s / d


class GraphTextFusion(nn.Module):
    """
    Produce a fused vector z (gate input) from:
      - text token embeddings (pre-encoder)
      - graph node embeddings
    """
    def __init__(self, text_dim: int, graph_dim: int, gate_dim: int,
                 dropout: float = 0.1, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.text_proj = nn.Linear(text_dim, gate_dim, bias=True)
        self.graph_proj = nn.Linear(graph_dim, gate_dim, bias=True)
        self.fuse = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_dim, gate_dim, bias=True),
            nn.SiLU()
        )
        self.out_ln = nn.LayerNorm(gate_dim)

    def forward(self, text_tokens: torch.Tensor,  # [B,T,H] (float32 recommended)
                graph_nodes: torch.Tensor,       # [N, G]
                slices: List[int],
                text_mask: torch.Tensor | None = None) -> torch.Tensor:
        device = text_tokens.device
        B, T, H = text_tokens.shape

        # text pooled
        t_pool = masked_mean(text_tokens, text_mask)           # [B,H]
        t_z = self.text_proj(t_pool)                           # [B,gate]

        # graph pooled per sample
        g_pools = []
        for i in range(B):
            s, e = slices[i], slices[i+1]
            if e > s:
                g_pools.append(graph_nodes[s:e].mean(dim=0, keepdim=True))
            else:
                g_pools.append(torch.zeros(1, graph_nodes.size(-1), device=device, dtype=graph_nodes.dtype))
        g_pool = torch.cat(g_pools, dim=0)                     # [B,G]
        g_z = self.graph_proj(g_pool)                          # [B,gate]

        z = self.fuse(t_z + g_z)                               # [B,gate]
        return self.out_ln(z)
