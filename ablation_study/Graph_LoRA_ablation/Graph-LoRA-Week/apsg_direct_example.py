
# -*- coding: utf-8 -*-
"""
Minimal integration sketch for the "APSG-direct" ablation.
- Serialize the APSG graph as text.
- Append it after the original text by concatenating embeddings.
- Skip writing any gating vector z into adapters.
"""

import torch
import torch.nn as nn

from .apsg_linearize import linearize_apsg_batch, ApsgDirectInjector

class GraphLoRA_APSGDirect(nn.Module):
    def __init__(self, llm, tokenizer, gate_dim: int, cls_hidden: int, num_labels: int = 2,
                 max_apsg_tokens: int = 256, max_total_tokens: int | None = None):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer

        # Build injector
        self.apsg_injector = ApsgDirectInjector(
            tokenizer=self.tokenizer,
            max_apsg_tokens=max_apsg_tokens,
            max_total_tokens=max_total_tokens,
            add_leading_sep="\\n\\n"  # makes APSG part visibly separated
        )

        # A small head after pooling last hidden state
        self.text_proj = nn.Linear(llm.config.hidden_size, cls_hidden)
        self.classifier = nn.Linear(cls_hidden, num_labels)
        self.drop = nn.Dropout(0.1)
        self.act = nn.SiLU()

        # NOTE: We assume adapters have been injected elsewhere already.
        # For this ablation we won't provide any z (i.e., unconditional adapters).

    def forward(
        self,
        input_ids: torch.Tensor,         # [B, T_text]
        attention_mask: torch.Tensor,    # [B, T_text]
        graphs: dict,                    # must contain edge_index, edge_type, n_nodes
        labels: torch.Tensor | None = None,
    ):
        device = input_ids.device

        # 1) Linearize APSG per sample
        apsg_strings = linearize_apsg_batch(graphs, max_edges_per_graph=256)

        # 2) Build (inputs_embeds, attention_mask) that appends APSG text AFTER the original text
        inputs_embeds, new_mask = self.apsg_injector.build_inputs(
            llm=self.llm,
            input_ids=input_ids,
            attention_mask=attention_mask,
            apsg_strings=apsg_strings,
            device=device,
        )

        # 3) DO NOT set adapter context (i.e., unconditional)
        # If you previously monkey-patched adapter.forward to expect _ctx_z, make sure it's optional.
        # Example (pseudo):
        # for a in self.adapters:
        #     a._ctx_z = None

        # 4) Run the LLM
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=new_mask, return_dict=True)
        last = out.last_hidden_state  # [B, T_total, H]

        # 5) Masked mean pooling over the full sequence (text + APSG)
        mask = new_mask.float().unsqueeze(-1)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        # 6) Classify
        h = self.act(self.text_proj(self.drop(pooled)))
        logits = self.classifier(self.drop(h))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}
