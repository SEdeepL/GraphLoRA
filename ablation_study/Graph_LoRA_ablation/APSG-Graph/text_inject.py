
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

__all__ = ["TextAddonInjector"]


class TextAddonInjector(nn.Module):
    """
    Generic text appender: tokenize a per-sample list of strings and append
    them (as embeddings) after the original text, updating the attention mask.
    """
    def __init__(
        self,
        tokenizer,
        max_addon_tokens: int = 512,
        max_total_tokens: Optional[int] = None,
        add_leading_sep: str = "\n\n",
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_addon_tokens = max_addon_tokens
        self.max_total_tokens = max_total_tokens
        self.add_leading_sep = add_leading_sep or ""

    @torch.no_grad()
    def build_inputs(
        self,
        llm,
        input_ids: torch.Tensor,         # [B, T_text]
        attention_mask: torch.Tensor,    # [B, T_text]
        addon_texts: List[str],          # [B] strings to append
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = input_ids.device
        B = input_ids.size(0)
        assert len(addon_texts) == B, f"Batch size mismatch: {len(addon_texts)} vs {B}"

        texts = [self.add_leading_sep + s for s in addon_texts]

        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_addon_tokens,
            add_special_tokens=False,
            return_tensors="pt",
        )
        add_ids = tok["input_ids"].to(device)
        add_mask = tok["attention_mask"].to(device)

        emb = llm.get_input_embeddings()
        text_emb = emb(input_ids.to(device))
        addon_emb = emb(add_ids)

        if self.max_total_tokens is not None:
            T_text = text_emb.size(1)
            T_add  = addon_emb.size(1)
            total = T_text + T_add
            if total > self.max_total_tokens:
                keep_add = max(self.max_total_tokens - T_text, 0)
                if keep_add < T_add:
                    addon_emb = addon_emb[:, :keep_add]
                    add_mask  = add_mask[:, :keep_add]
                    T_add = keep_add
                total = T_text + T_add
                if total > self.max_total_tokens and T_text > 0:
                    need = total - self.max_total_tokens
                    if need < T_text:
                        text_emb = text_emb[:, need:]
                        attention_mask = attention_mask[:, need:]
                    else:
                        text_emb = text_emb[:, -1:]
                        attention_mask = attention_mask[:, -1:]

        inputs_embeds = torch.cat([text_emb, addon_emb], dim=1)
        new_mask = torch.cat([attention_mask.to(device), add_mask], dim=1)
        return inputs_embeds, new_mask
