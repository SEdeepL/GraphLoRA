
# -*- coding: utf-8 -*-
"""
Unified Graph-LoRA Model (with Ablations)
-----------------------------------------
This `model.py` provides an all-in-one model that supports three fusion modes:

1) fusion_type="attn"        -> uses GraphTextFusion (from fusion.py)
2) fusion_type="weak"        -> uses GraphTextFusionWeak (from fusion_weak.py)
3) fusion_type="apsg-direct" -> linearizes APSG and appends it to the LLM input (no adapter gating)

Requirements in the same package:
- gcn.py                : RGCNEncoder
- fusion.py             : GraphTextFusion
- fusion_weak.py        : GraphTextFusionWeak (optional; used when fusion_type="weak")
- apsg_linearize.py     : linearize_apsg_batch, ApsgDirectInjector (only for "apsg-direct")
- adapters.py           : wrap_llama_with_adapters

Typical usage:
--------------
from .model import GraphLoRA

model = GraphLoRA(
    model_name_or_path="your-llm",
    tokenizer=tokenizer,             # HF tokenizer that matches your LLM
    fusion_type="attn",              # or "weak" / "apsg-direct"
    gcn_in_dim=graph_in_dim,         # node feature dim
    gcn_hidden_dim=256,
    gcn_out_dim=256,
    gcn_num_relations=num_rel,
    gcn_num_layers=2,
    gate_dim=256,
    cls_hidden=256,                  # classifier hidden size (for apsg-direct path)
    num_labels=2,
    lora_rank=8,
    lora_alpha=16,
    lora_target="attn_ffn",          # ["attn", "ffn", "attn_ffn", "all"]
    use_pissa=True,
    freeze_base=True,
)

out = model(input_ids, attention_mask, graphs, labels)
loss, logits = out["loss"], out["logits"]

Graphs dict must contain:
- node_feat : [N, D_in] (float)
- edge_index: [2, E] or [E, 2] (long)
- edge_type : [E] (long)
- n_nodes   : List[int] length B (node counts per sample, sum equals N)
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Optional, Tuple
import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoConfig
except Exception:
    AutoModel = None
    AutoConfig = None

from .gcn import RGCNEncoder
from .fusion import GraphTextFusion
# optional weak fusion
try:
    from .fusion_weak import GraphTextFusionWeak
except Exception:
    GraphTextFusionWeak = None

# optional APSG-direct
try:
    from .apsg_linearize import linearize_apsg_batch, ApsgDirectInjector
except Exception:
    linearize_apsg_batch = None
    ApsgDirectInjector = None

from .adapters import wrap_llama_with_adapters


# ---- helpers ----

def masked_mean(hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    hidden: [B,T,H], mask: [B,T] (1 for valid). If mask is None, simple mean.
    """
    if mask is None:
        return hidden.mean(dim=1)
    m = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


# ---- model ----

class GraphLoRA(nn.Module):
    """
    A unified Graph-LoRA model supporting three fusion modes.

    - In "attn"/"weak": compute gate vector z from (text tokens, graph nodes)
      and write z into adapters; then run LLM once and classify with (h_text + z).
    - In "apsg-direct": linearize APSG and append to text tokens; do text-only
      pooling/classification (no z).
    """
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        llm: Optional[nn.Module] = None,
        tokenizer=None,
        # Fusion choice
        fusion_type: str = "attn",  # "attn" | "weak" | "apsg-direct"
        # GCN config (for "attn" or "weak")
        gcn_in_dim: int = 256,
        gcn_hidden_dim: int = 256,
        gcn_out_dim: int = 256,
        gcn_num_relations: int = 8,
        gcn_num_layers: int = 2,
        gcn_dropout: float = 0.1,
        # Adapter gating dim (for "attn" or "weak")
        gate_dim: int = 256,
        fusion_dropout: float = 0.1,
        # Classifier
        cls_hidden: int = 256,   # used in apsg-direct path or if you prefer extra MLP
        num_labels: int = 2,
        cls_dropout: float = 0.1,
        # Adapters
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_target: str = "attn_ffn",  # "attn"|"ffn"|"attn_ffn"|"all"
        use_pissa: bool = True,
        # Base
        freeze_base: bool = True,
        gradient_checkpointing: bool = True,
        # APSG-direct limits
        apsg_max_tokens: int = 256,
        max_total_tokens: Optional[int] = None,
        apsg_sep: str = "\n\n",
    ) -> None:
        super().__init__()
        assert fusion_type in {"attn", "weak", "apsg-direct"}
        self.fusion_type = fusion_type
        self.num_labels = num_labels

        # 1) Build/load LLM
        if llm is None:
            if AutoModel is None:
                raise RuntimeError("Transformers is required to AutoModel.from_pretrained(...)")
            if model_name_or_path is None:
                raise ValueError("Provide either llm or model_name_or_path.")
            self.llm = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None)
        else:
            self.llm = llm

        if gradient_checkpointing and hasattr(self.llm, "gradient_checkpointing_enable"):
            try:
                self.llm.gradient_checkpointing_enable()
            except Exception:
                pass

        if freeze_base:
            for p in self.llm.parameters():
                p.requires_grad = False

        self.hidden_size = int(getattr(self.llm.config, "hidden_size", 0) or getattr(self.llm.config, "hidden_dim", 0))
        if self.hidden_size <= 0:
            raise ValueError("Cannot infer LLM hidden size from config.hidden_size/hidden_dim.")

        # 2) Inject adapters (LoRA/DoRA + optional PiSSA). Returns a list of adapter modules.
        self.adapters = wrap_llama_with_adapters(
            self.llm,
            rank=lora_rank,
            alpha=lora_alpha,
            target=lora_target,
            use_pissa=use_pissa,
        )

        # 3) Build graph branch & fusion (only for non-apsg-direct)
        if fusion_type in {"attn", "weak"}:
            self.rgcn = RGCNEncoder(
                in_dim=gcn_in_dim,
                hidden_dim=gcn_hidden_dim,
                out_dim=gcn_out_dim,
                num_relations=gcn_num_relations,
                num_layers=gcn_num_layers,
                dropout=gcn_dropout,
            )
            if fusion_type == "attn":
                self.fusion = GraphTextFusion(
                    text_dim=self.hidden_size,
                    graph_dim=gcn_out_dim,
                    gate_dim=gate_dim,
                    dropout=fusion_dropout,
                )
            else:
                if GraphTextFusionWeak is None:
                    raise RuntimeError("fusion_weak.py not found but fusion_type='weak' was requested.")
                self.fusion = GraphTextFusionWeak(
                    text_dim=self.hidden_size,
                    graph_dim=gcn_out_dim,
                    gate_dim=gate_dim,
                    dropout=fusion_dropout,
                )
            # project pooled text to gate_dim; final cls on fused (h_text + z)
            self.text_to_gate = nn.Linear(self.hidden_size, gate_dim)
            self.cls_head = nn.Sequential(
                nn.Dropout(cls_dropout),
                nn.Linear(gate_dim, num_labels),
            )
        else:
            # APSG-direct path: need tokenizer and injector
            if (linearize_apsg_batch is None) or (ApsgDirectInjector is None):
                raise RuntimeError("apsg_linearize.py not found but fusion_type='apsg-direct' was requested.")
            if tokenizer is None:
                raise ValueError("tokenizer is required for 'apsg-direct'.")
            self.tokenizer = tokenizer
            self.apsg_injector = ApsgDirectInjector(
                tokenizer=self.tokenizer,
                max_apsg_tokens=apsg_max_tokens,
                max_total_tokens=max_total_tokens,
                add_leading_sep=apsg_sep,
            )
            # text-only classifier head
            self.cls_proj = nn.Linear(self.hidden_size, cls_hidden)
            self.cls_head_text = nn.Sequential(
                nn.Dropout(cls_dropout),
                nn.SiLU(),
                nn.Linear(cls_hidden, num_labels),
            )

        self.act = nn.SiLU()

    # ---- utilities ----

    def _set_adapter_context(self, z: Optional[torch.Tensor]) -> None:
        """
        Store a per-batch context vector z inside each adapter for later read
        during the LLM forward pass. When z is None, adapters should behave as
        unconditional (implementation in adapters.py should allow this).
        """
        if z is not None:
            z = z.detach()
        for a in self.adapters:
            # Adapters are expected to read this attribute inside their forward
            a._ctx_z = z

    # ---- forward ----

    def forward(
        self,
        input_ids: torch.Tensor,         # [B, T]
        attention_mask: torch.Tensor,    # [B, T]
        graphs: Optional[Dict[str, torch.Tensor | Sequence[int]]] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        device = input_ids.device

        if self.fusion_type == "apsg-direct":
            # 1) Build APSG strings per sample
            apsg_strings = linearize_apsg_batch(
                graphs,
                max_edges_per_graph=256,
                prefix_token="[APSG]",
                suffix_token="[/APSG]",
            )

            # 2) Embedding-level concatenation
            inputs_embeds, new_mask = self.apsg_injector.build_inputs(
                llm=self.llm,
                input_ids=input_ids,
                attention_mask=attention_mask,
                apsg_strings=apsg_strings,
                device=device,
            )

            # 3) Unconditional adapters (no z)
            self._set_adapter_context(None)

            # 4) LLM forward
            out = self.llm(inputs_embeds=inputs_embeds, attention_mask=new_mask, return_dict=True)
            last = out.last_hidden_state  # [B, T_total, H]

            # 5) Pool (masked mean) and classify
            pooled = masked_mean(last, new_mask)
            h = self.act(self.cls_proj(pooled))
            logits = self.cls_head_text(h)

            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)

            return {"loss": loss, "logits": logits} if return_dict else (loss, logits)

        # ---- "attn" / "weak" paths ----
        if graphs is None:
            raise ValueError("graphs must be provided for fusion_type in {'attn', 'weak'}.")

        node_feat = graphs["node_feat"]           # [N, D_in]
        edge_index = graphs["edge_index"]         # [2,E] or [E,2]
        edge_type = graphs["edge_type"]           # [E]
        n_nodes = graphs["n_nodes"]               # List[int] length B
        if torch.is_tensor(n_nodes):
            n_nodes_list = n_nodes.detach().cpu().tolist()
        else:
            n_nodes_list = list(n_nodes)
        slices = [0]
        for c in n_nodes_list:
            slices.append(slices[-1] + int(c))

        # 1) Build text token embeddings for LLM and for fusion
        emb = self.llm.get_input_embeddings()   # nn.Embedding
        text_emb = emb(input_ids.to(device))    # [B,T,H] (dtype per LLM, e.g., bf16)
        # Use fp32 copy of tokens for stable fusion calc
        text_tokens_fp32 = text_emb.detach().to(torch.float32)

        # 2) Graph encoder -> node embeddings [N, gcn_out_dim]
        H_nodes, _ = self.rgcn(
            node_feat=node_feat,
            edge_index=edge_index,
            edge_type=edge_type,
            n_nodes=n_nodes_list,
        )  # H_nodes:[N, gcn_out_dim]

        # 3) Compute gate vector z:[B, gate_dim]
        z = self.fusion(
            text_tokens=text_tokens_fp32,
            graph_nodes=H_nodes.to(dtype=torch.float32),
            slices=slices,
            text_mask=attention_mask,
        )  # [B, gate_dim]

        # 4) Write z into adapters (detached; dtype/device handled in adapters)
        self._set_adapter_context(z)

        # 5) Single LLM forward
        out = self.llm(inputs_embeds=text_emb, attention_mask=attention_mask, return_dict=True)
        last = out.last_hidden_state  # [B,T,H]

        # 6) Pool text and fuse with z for classification
        h_text = self.text_to_gate(masked_mean(last, attention_mask))  # [B, gate_dim]
        fused = nn.functional.dropout(h_text + z, p=0.0, training=self.training)
        logits = self.cls_head(fused)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits} if return_dict else (loss, logits)
