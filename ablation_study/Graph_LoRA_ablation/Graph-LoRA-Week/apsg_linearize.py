
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import torch
import torch.nn as nn

__all__ = [
    "linearize_apsg_batch",
    "ApsgDirectInjector",
]


def _normalize_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Ensure edge_index shape is [2, E] (long).
    """
    if edge_index.dim() != 2:
        raise ValueError("edge_index must be 2D")
    if edge_index.size(0) == 2:
        return edge_index.long()
    elif edge_index.size(1) == 2:
        return edge_index.t().contiguous().long()
    else:
        raise ValueError(f"Unexpected edge_index shape: {edge_index.shape}")


def _split_edges_by_sample(
    edge_index: torch.Tensor,  # [2, E] global indices
    edge_type: torch.Tensor,   # [E]
    slices: Sequence[int],     # len B+1
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Return per-sample edges: (src, dst, etype), indices rebased to [0, n_i).
    Only keep edges whose src and dst are both within the sample range.
    """
    device = edge_index.device
    if not torch.is_tensor(slices):
        slices = torch.tensor(list(slices), device=device, dtype=torch.long)
    outs = []
    for i in range(len(slices) - 1):
        s = int(slices[i].item())
        e = int(slices[i+1].item())
        if e <= s:
            outs.append((torch.empty(0, dtype=torch.long, device=device),
                         torch.empty(0, dtype=torch.long, device=device),
                         torch.empty(0, dtype=edge_type.dtype, device=device)))
            continue
        # mask edges fully inside [s, e)
        src = edge_index[0]
        dst = edge_index[1]
        m = (src >= s) & (src < e) & (dst >= s) & (dst < e)
        if m.any():
            src_i = (src[m] - s).long()
            dst_i = (dst[m] - s).long()
            et_i = edge_type[m].long()
            outs.append((src_i, dst_i, et_i))
        else:
            outs.append((torch.empty(0, dtype=torch.long, device=device),
                         torch.empty(0, dtype=torch.long, device=device),
                         torch.empty(0, dtype=torch.long, device=device)))
    return outs


def _format_triples_as_text(
    n_nodes: int,
    src: torch.Tensor,  # [E_i]
    dst: torch.Tensor,  # [E_i]
    et: torch.Tensor,   # [E_i]
    max_edges: int | None = None,
) -> str:
    """
    Produce a compact, stable text for one sample:
    Ex: "N=5; E=3; TRIPLES: (0,r1,2); (1,r0,4); (4,r3,3)"
    """
    E = src.numel()
    # Stable ordering: by (src, et, dst)
    if E > 0:
        order = torch.lexsort((dst.cpu(), et.cpu(), src.cpu()))
        src = src[order]
        dst = dst[order]
        et = et[order]
    if (max_edges is not None) and (E > max_edges):
        src = src[:max_edges]
        dst = dst[:max_edges]
        et  = et[:max_edges]
        truncated = True
    else:
        truncated = False

    triples = "; ".join([f"({int(s)},r{int(t)},{int(d)})" for s, t, d in zip(src.tolist(), et.tolist(), dst.tolist())])
    header = f"N={int(n_nodes)}; E={int(E if not truncated else max_edges)}"
    body = f"TRIPLES: {triples}" if len(triples) else "TRIPLES: (none)"
    if truncated:
        body += " ; [TRUNCATED]"
    return f"{header}; {body}"


def linearize_apsg_batch(
    graphs: Dict[str, torch.Tensor | Sequence[int]],
    max_edges_per_graph: int | None = 256,
    prefix_token: str = "[APSG]",
    suffix_token: str = "[/APSG]",
) -> List[str]:
    """
    Turn the batched APSG (edge-only) into per-sample strings.
    Expected keys in `graphs`:
      - 'edge_index': [2,E] or [E,2]
      - 'edge_type': [E]
      - 'n_nodes'   : Sequence[int] of length B
    Returns List[str] length B, each wrapped by prefix/suffix tokens.
    """
    edge_index = graphs.get("edge_index", None)
    edge_type  = graphs.get("edge_type", None)
    n_nodes    = graphs.get("n_nodes", None)
    if edge_index is None or edge_type is None or n_nodes is None:
        raise KeyError("graphs must contain 'edge_index', 'edge_type', and 'n_nodes'")

    edge_index = _normalize_edge_index(edge_index)
    if edge_type.dim() != 1:
        edge_type = edge_type.view(-1)
    if len(n_nodes) == 0:
        return []

    # Build cumulative slices [0, n1, n1+n2, ...]
    device = edge_index.device
    if torch.is_tensor(n_nodes):
        n_nodes_t = n_nodes.to(device=device, dtype=torch.long).view(-1)
    else:
        n_nodes_t = torch.tensor(list(n_nodes), device=device, dtype=torch.long)
    slices = torch.cat([torch.zeros(1, device=device, dtype=torch.long), n_nodes_t.cumsum(dim=0)], dim=0)

    per_sample_edges = _split_edges_by_sample(edge_index, edge_type, slices)
    outs: List[str] = []
    for i, (src, dst, et) in enumerate(per_sample_edges):
        n_i = int(n_nodes_t[i].item())
        txt = _format_triples_as_text(n_i, src, dst, et, max_edges=max_edges_per_graph)
        outs.append(f"{prefix_token} {txt} {suffix_token}")
    return outs


class ApsgDirectInjector(nn.Module):
    """
    Utility to append linearized APSG text right after the original text sequence
    by operating on embeddings to avoid re-tokenizing the original text.

    Workflow:
      1) Get linearized APSG strings via `linearize_apsg_batch(graphs, ...)`.
      2) Call `build_inputs(...)` to create (inputs_embeds, attention_mask) for LLM.forward.

    Notes:
      - Requires a Hugging Face tokenizer compatible with the LLM embedding matrix.
      - Handles variable APSG lengths via padding and masks.
      - Supports max_total_tokens to avoid exceeding context length.
    """
    def __init__(
        self,
        tokenizer,
        max_apsg_tokens: int = 256,
        max_total_tokens: int | None = None,
        add_leading_sep: str | None = "\n\n",
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_apsg_tokens = max_apsg_tokens
        self.max_total_tokens = max_total_tokens
        self.add_leading_sep = add_leading_sep or ""

    @torch.no_grad()
    def build_inputs(
        self,
        llm,  # has get_input_embeddings()
        input_ids: torch.Tensor,        # [B, T_text]
        attention_mask: torch.Tensor,   # [B, T_text]
        apsg_strings: List[str],        # length B
        device: torch.device | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          inputs_embeds: [B, T_text + T_apsg, H]
          new_mask     : [B, T_text + T_apsg]
        """
        if device is None:
            device = input_ids.device
        B = input_ids.size(0)
        assert len(apsg_strings) == B, "Batch size mismatch for apsg_strings"

        # 1) Create the APSG text sequence (optionally with a leading separator)
        apsg_texts = [self.add_leading_sep + s for s in apsg_strings]

        # 2) Tokenize APSG strings
        tok = self.tokenizer(
            apsg_texts,
            padding=True,
            truncation=True,
            max_length=self.max_apsg_tokens,
            add_special_tokens=False,
            return_tensors="pt",
        )
        apsg_ids = tok["input_ids"].to(device)
        apsg_mask = tok["attention_mask"].to(device)

        # 3) Embed original text and APSG segments
        emb = llm.get_input_embeddings()   # nn.Embedding
        text_emb = emb(input_ids.to(device))      # [B, T_text, H]
        apsg_emb = emb(apsg_ids)                 # [B, T_apsg, H]

        # 4) Optionally enforce a max total length
        if self.max_total_tokens is not None:
            T_text = text_emb.size(1)
            T_apsg = apsg_emb.size(1)
            total = T_text + T_apsg
            if total > self.max_total_tokens:
                # First trim APSG tokens
                keep_apsg = max(self.max_total_tokens - T_text, 0)
                if keep_apsg < T_apsg:
                    apsg_emb  = apsg_emb[:, :keep_apsg]
                    apsg_mask = apsg_mask[:, :keep_apsg]
                    T_apsg = keep_apsg
                total = T_text + T_apsg
                # If still too long, trim the *left* side of text (keep recent tokens)
                if total > self.max_total_tokens and T_text > 0:
                    need = total - self.max_total_tokens
                    if need < T_text:
                        text_emb = text_emb[:, need:]
                        attention_mask = attention_mask[:, need:]
                    else:
                        # Extreme case: keep last token
                        text_emb = text_emb[:, -1:]
                        attention_mask = attention_mask[:, -1:]

        # 5) Concatenate
        inputs_embeds = torch.cat([text_emb, apsg_emb], dim=1)
        new_mask = torch.cat([attention_mask.to(device), apsg_mask], dim=1)
        return inputs_embeds, new_mask
