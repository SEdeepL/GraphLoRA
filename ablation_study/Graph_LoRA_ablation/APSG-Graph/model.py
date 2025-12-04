
# -*- coding: utf-8 -*-
"""
Third ablation: remove APSG entirely and only append the **code patch** text
to the LLM input sequence. No graph encoders, no gating vector z.

Assumptions:
- You have per-sample code patch strings (e.g., unified diff / git patch).
- We'll wrap each with markers like [PATCH] ... [/PATCH] (configurable).
- Adapters (LoRA/DoRA) are injected but run unconditionally (no z).

Usage:
    from .model_patch_direct import GraphLoRA_PatchDirect
    model = GraphLoRA_PatchDirect(llm, tokenizer, cls_hidden=1024, num_labels=2)
    out = model(input_ids, attention_mask, patch_texts=patch_list, labels=labels)
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Dict
import torch
import torch.nn as nn

from .text_inject import TextAddonInjector


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return hidden.mean(dim=1)
    m = mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class GraphLoRA_PatchDirect(nn.Module):
    def __init__(
        self,
        llm,
        tokenizer,
        cls_hidden: int = 1024,
        num_labels: int = 2,
        max_patch_tokens: int = 512,
        max_total_tokens: Optional[int] = None,
        sep: str = "\\n\\n",
        use_code_fence: bool = True,
        header_token: str = "[PATCH]",
        footer_token: str = "[/PATCH]",
        dropout: float = 0.1,
        pool: str = "mean",  # or "last"
    ) -> None:
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.hidden_size = llm.config.hidden_size
        self.pool = pool
        self.header_token = header_token
        self.footer_token = footer_token
        self.use_code_fence = use_code_fence

        self.injector = TextAddonInjector(
            tokenizer=self.tokenizer,
            max_addon_tokens=max_patch_tokens,
            max_total_tokens=max_total_tokens,
            add_leading_sep=sep,
        )

        self.proj = nn.Linear(self.hidden_size, cls_hidden)
        self.classifier = nn.Linear(cls_hidden, num_labels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()

        # NOTE: z-free; adapters should be unconditional.

    def _format_patch(self, s: str) -> str:
        s = s.strip("\n\r")
        if self.use_code_fence:
            return f"{self.header_token}\n```diff\n{s}\n```\n{self.footer_token}"
        return f"{self.header_token} {s} {self.footer_token}"

    def _pool(self, last: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pool == "last":
            idx = mask.sum(dim=1).clamp_min(1).long() - 1
            return last[torch.arange(last.size(0), device=last.device), idx]
        return masked_mean(last, mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        patch_texts: Optional[List[str]] = None,
        graphs: Optional[Dict] = None,  # optional carrier for patch strings
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        device = input_ids.device
        B = input_ids.size(0)

        # 1) Resolve patch strings
        if patch_texts is None:
            if graphs is None:
                raise ValueError("Provide patch_texts=list[str] or graphs containing them.")
            # Try common keys
            for k in ("patch_texts", "code_patch", "patch"):
                if k in graphs:
                    patch_texts = graphs[k]
                    break
        if patch_texts is None:
            raise ValueError("No patch_texts found.")

        if len(patch_texts) != B:
            raise AssertionError(f"patch_texts len {len(patch_texts)} != batch size {B}")

        addon_texts = [self._format_patch(s) for s in patch_texts]

        # 2) Build concatenated inputs
        inputs_embeds, new_mask = self.injector.build_inputs(
            llm=self.llm,
            input_ids=input_ids,
            attention_mask=attention_mask,
            addon_texts=addon_texts,
            device=device,
        )

        # 3) LLM forward (no z to adapters)
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=new_mask, return_dict=True)
        last = out.last_hidden_state

        # 4) Pool + classify
        pooled = self._pool(last, new_mask)
        h = self.act(self.proj(self.drop(pooled)))
        logits = self.classifier(self.drop(h))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        if not return_dict:
            return (loss, logits)
        return {"loss": loss, "logits": logits}
