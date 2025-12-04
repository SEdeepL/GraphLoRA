# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from .fusion_weak import GraphTextFusionWeak as GraphTextFusion

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from .gcn import RGCNEncoder
from .adapters import wrap_llama_with_adapters
from .fusion import GraphTextFusion

# ---- 适配器 forward 打补丁（读取 ctx z）----
from .adapters import DoRAPiSSAAdapter as _Adapter

_old_forward = _Adapter.forward
def _new_forward(self, x, z_fusion=None):
    if z_fusion is None:
        z_fusion = getattr(self, "_ctx_z", None)
        if z_fusion is None:
            B = x.shape[0]
            z_fusion = torch.zeros(B, self.gate[0].in_features, device=x.device, dtype=x.dtype)
    return _old_forward(self, x, z_fusion)

_Adapter.forward = _new_forward


class GraphDoRAModel(nn.Module):
    """
    流程：
      1) token embedding + 图特征 -> 融合得到 z（gate_dim）
      2) 将 z 作为上下文注入到每层 DoRA 适配器（_ctx_z）
      3) 用 inputs_embeds 跑一次 LLM 前向，取 last_hidden_state 做 masked-mean -> h_text
      4) 将 h_text 投影到 gate_dim 与 z 相加，得到 fused_for_clf -> classifier -> logits
      5) 用 labels 计算交叉熵 loss

    方案1要点：
      - 注入适配器的 z 使用 detach()，以兼容 gradient checkpointing 的重算一致性；
      - 冻结底座 LLM，只训练适配器 + 图分支 + 分类头；
      - 不在 forward 末尾清空 _ctx_z（下一次前向会覆盖，且 z 已 detach 不会挂住计算图）。
    """

    def __init__(self, model_id: str,
                 g_in: int, g_hid: int, g_out: int, n_rels: int, g_layers: int,
                 rank: int, alpha: float, pissa: bool, gate_dim: int, target: str = "all"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_id)
        self.llm = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
        )
        self.llm.config.output_hidden_states = False
        self.llm.config.use_cache = False
        try:
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            self.llm.gradient_checkpointing_enable()

        # ---- 冻结底座 LLM（关键）：在注入适配器之前执行，避免把适配器也冻结 ----
        for p in self.llm.parameters():
            p.requires_grad = False

        # 注入 DoRA(+PiSSA) 适配器（这些新参数默认是可训练的）
        self.adapters: List[nn.Module] = wrap_llama_with_adapters(
            self.llm, rank=rank, alpha=alpha, pissa_init=pissa, gate_dim=gate_dim, target=target
        )

        # 图编码器
        self.gcn = RGCNEncoder(in_dim=g_in, hid_dim=g_hid, out_dim=g_out,
                               num_rels=n_rels, num_layers=g_layers, dropout=0.1)

        # 文本-图 融合（内部自带多头，不需要 heads 参数）
        text_dim = self.llm.config.hidden_size
        self.fusion = GraphTextFusion(
            text_dim=llm.config.hidden_size,   # 文本 token 的维度
            graph_dim=rgcn_out_dim,            # 图编码器输出维度
            gate_dim=gate_dim,                 # 适配器门控维度（保持与原实验一致）
            dropout=fusion_dropout,
        )

        # 将文本 pooled 表征投影到 gate_dim，便于与 z 融合
        self.text_proj = nn.Linear(text_dim, gate_dim)
        self.post_drop = nn.Dropout(p=0.1)

        # 分类头（在 fused(z, h_text) 上）
        self.classifier = nn.Sequential(
            nn.Linear(gate_dim, gate_dim),
            nn.ReLU(),
            nn.Linear(gate_dim, 2)
        )

        self.ln = nn.LayerNorm(text_dim)

    # 统一获取 embed_tokens（兼容 LlamaModel 与 LlamaForCausalLM）
    def _get_embed_module(self):
        if hasattr(self.llm, "embed_tokens"):  # LlamaModel
            return self.llm.embed_tokens
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "embed_tokens"):  # LlamaForCausalLM
            return self.llm.model.embed_tokens
        raise AttributeError("LLM has no embed_tokens (unexpected architecture).")

    def _token_embeddings(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 (emb_for_llm, emb_for_fusion)
        - emb_for_llm: 与 LLM 权重 dtype 一致，用于 inputs_embeds 走主干
        - emb_for_fusion: float32，提升融合/分类稳定性
        """
        embed_tokens = self._get_embed_module()
        emb = embed_tokens(input_ids)     # [B,T,H], dtype == self.llm.dtype
        emb_f32 = emb.to(torch.float32)
        return emb, emb_f32

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B,T,H], mask: [B,T] (1 for valid)
        mask = mask.to(dtype=x.dtype)             # 与 x 同 dtype（bf16/fp16）
        num = (x * mask.unsqueeze(-1)).sum(dim=1)
        den = mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        return num / den

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                graphs: Dict[str, Any],
                labels: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:

        # -------- 1) 文本 token embedding --------
        emb_llm, txt_for_fusion = self._token_embeddings(input_ids)   # emb_llm: bf16/fp16; txt_for_fusion: fp32

        # -------- 2) 图编码 --------
        node_feat = graphs["node_feat"]         # (N, Din) float
        edge_index = graphs["edge_index"]       # (2, E) long
        edge_type  = graphs["edge_type"]        # (E,) long
        n_nodes: List[int] = graphs["n_nodes"]  # List[int]
        Hn, _ = self.gcn(node_feat, edge_index, edge_type, n_nodes)  # Hn:(N, g_out), G:(B, g_out) 未使用 G

        # 为每个样本构造 slice
        slices = [0]
        s = 0
        for n in n_nodes:
            s += int(n)
            slices.append(s)

        # -------- 3) 融合 -> z（只用 token embedding 做文本分支，轻量稳定）--------
        txt_feat = self.ln(txt_for_fusion)              # fp32
        z = self.fusion(txt_feat, Hn, slices)           # (B, gate_dim) fp32

        # —— 方案1关键：给适配器的 z 与 LLM dtype 一致，并 detach，兼容 ckpt 重算 —— 
        z_adapt = z.to(self.llm.dtype).detach()
        for adp in self.adapters:
            adp._ctx_z = z_adapt   # 不清空；下一次前向会覆盖

        # -------- 4) LLM 前向（一次）并取 pooled 文本表征 --------
        out = self.llm(inputs_embeds=emb_llm, attention_mask=attention_mask)
        hidden = out.last_hidden_state                  # [B,T,H]
        h_text = self._masked_mean(hidden, attention_mask)   # [B,H] (bf16/fp16)
        h_text = h_text.to(torch.float32)                    # 线性层默认 fp32 更稳定
        h_text = self.text_proj(h_text)                      # [B, gate_dim] fp32

        # -------- 5) 与 z 融合后分类 --------
        fused_for_clf = self.post_drop(z + h_text)           # [B, gate_dim] fp32
        logits = self.classifier(fused_for_clf).float()      # logits fp32

        if labels is not None:
            labels_t = labels.long().to(logits.device)
            loss = F.cross_entropy(logits, labels_t).float()
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
