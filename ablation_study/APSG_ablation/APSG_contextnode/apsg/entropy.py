from __future__ import annotations
from typing import Optional, Protocol
import math

class EntropyProvider(Protocol):
    def compute_entropy(self, text: str) -> float: ...

class StubEntropyProvider:
    def compute_entropy(self, text: str) -> float:
        toks = [t for t in text.replace('(', ' ').replace(')', ' ').replace(';', ' ').replace(',', ' ').split() if t]
        if not toks:
            return 0.0
        unique = len(set(toks))
        return float(unique) / float(len(toks)) * math.log2(len(toks) + 1.0)

class LlamaEntropy:
    def __init__(self, model_path: str, device: str = "auto", dtype: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        kwargs = {}
        if dtype == "float16":
            kwargs["torch_dtype"] = torch.float16
        elif dtype == "bfloat16":
            kwargs["torch_dtype"] = torch.bfloat16
        elif dtype == "float32":
            kwargs["torch_dtype"] = torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, **kwargs)
        self.model.eval()

    def compute_entropy(self, text: str) -> float:
        import torch, math
        with torch.no_grad():
            enc = self.tokenizer(text, return_tensors="pt")
            input_ids = enc["input_ids"]
            attn = enc.get("attention_mask", None)
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            if attn is not None:
                attn = attn.to(device)
            out = self.model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            gathered = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            bits = (-gathered / math.log(2.0)).mean().item()
            return float(bits)
