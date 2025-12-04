
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Sequence
import torch
import torch.nn as nn

__all__ = ["GraphTextFusionWeak", "masked_mean"]


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """
    Compute mean over time with optional mask.
    hidden: [B,T,H], mask: [B,T] (1 for valid). If mask is None, it's a simple mean.
    """
    if mask is None:
        return hidden.mean(dim=1)
    mask = mask.to(hidden.dtype)
    s = (hidden * mask.unsqueeze(-1)).sum(dim=1)
    c = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
    return s / c


class GraphTextFusionWeak(nn.Module):
    """
    A lightweight fusion that removes attention-style fusion and uses simple vector concatenation.

    Given:
        - text token embeddings (pre-LLM): [B, T, H_text]
        - per-node graph embeddings (concatenated batch): [N, H_graph]
        - slices: cumulative node counts per sample, length B+1 (e.g., [0, n1, n1+n2, ...])

    It computes:
        t_pool = mean over tokens
        g_pool = mean over nodes per sample (via slices)
        z = LN( MLP( concat([t_pool, g_pool]) ) )  -> [B, gate_dim]

    This z can be written into adapters as the conditional vector.
    """
    def __init__(
        self,
        text_dim: int,
        graph_dim: int,
        gate_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        fused_in = text_dim + graph_dim
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, gate_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_dim, gate_dim),
            nn.SiLU(),
        )
        self.out_ln = nn.LayerNorm(gate_dim)

    def forward(
        self,
        text_tokens: torch.Tensor,     # [B, T, H_text] (float32 recommended)
        graph_nodes: torch.Tensor,     # [N, H_graph]
        slices: Sequence[int],         # len = B+1, cumulative node counts
        text_mask: torch.Tensor | None = None,  # Optional [B, T]
    ) -> torch.Tensor:
        device = text_tokens.device
        B = text_tokens.size(0)

        # 1) Pool text (masked mean if mask provided)
        t_pool = masked_mean(text_tokens, text_mask)   # [B, H_text]

        # 2) Pool graph per sample using slices
        if not torch.is_tensor(slices):
            # Normalize to a 1D Tensor on the same device for easier indexing
            slices = torch.tensor(list(slices), device=device, dtype=torch.long)
        g_pools = []
        H_graph = graph_nodes.size(-1)
        for i in range(B):
            s = int(slices[i].item())
            e = int(slices[i+1].item())
            if e > s:
                g_pools.append(graph_nodes[s:e].mean(dim=0, keepdim=True))
            else:
                g_pools.append(torch.zeros(1, H_graph, device=device, dtype=graph_nodes.dtype))
        g_pool = torch.cat(g_pools, dim=0)  # [B, H_graph]

        # 3) Concatenate & project to gate_dim
        fused_in = torch.cat([t_pool, g_pool], dim=-1)  # [B, H_text + H_graph]
        z = self.fuse(fused_in)                         # [B, gate_dim]
        return self.out_ln(z)                           # [B, gate_dim]
