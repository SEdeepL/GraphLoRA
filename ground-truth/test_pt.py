# -*- coding: utf-8 -*-
from __future__ import annotations
import os, argparse
from contextlib import nullcontext
from typing import Dict, Any, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# ====== datasets ======
from graph_dora.data.cached_dataset import CachedGraphTextDataset, collate_cached
from graph_dora.data.json_dataset import PatchDataset, collate_fn

# ====== model ======
from graph_dora.models.model import GraphDoRAModel

from transformers import AutoTokenizer


# ====== 新的 patch prompt 模板（情况 B：Humanwrite 作为 ground-truth 示例） ======
PATCH_PROMPT_TEMPLATE = (
    "You are a model responsible for assessing patch correctness.\n\n"
    "Below is one ground-truth example. This patch is correct.\n"
    "{humanwrite}\n\n"
    "Assess whether the patch is correct:\n"
    "{patch_text}"
)


def build_patch_prompt(humanwrite: str, patch_text: str) -> str:
    """
    使用 Humanwrite 作为一个 ground-truth 正确补丁的示例说明，
    然后让模型去判断当前 patch_text 是否正确。
    """
    humanwrite = (humanwrite or "").strip()
    patch_text = (patch_text or "").strip()
    return PATCH_PROMPT_TEMPLATE.format(humanwrite=humanwrite, patch_text=patch_text)


def ddp_barrier_safe():
    if dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def setup_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        return True, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return False, 0, 1


def graphs_to_device(graphs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move graph tensors to device. Keep 'n_nodes' as python List[int]."""
    out = {}
    for k, v in graphs.items():
        if k == "n_nodes":
            out[k] = [int(x) for x in v]
        else:
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device, non_blocking=True)
            else:
                if k == "edge_index":
                    t = torch.tensor(v, dtype=torch.long)
                elif k == "edge_type":
                    t = torch.tensor(v, dtype=torch.long)
                else:  # node_feat
                    t = torch.tensor(v, dtype=torch.float32)
                out[k] = t.to(device, non_blocking=True)
    return out


def compute_class_weights_v2(loader) -> torch.Tensor:
    counts = torch.zeros(2, dtype=torch.long)
    for batch in loader:
        y = batch["labels"]
        counts += torch.bincount(y, minlength=2)
    total = counts.sum().item()
    weights = total / (2.0 * counts.clamp(min=1)).float()
    weights = torch.clamp(weights, max=5.0)
    return weights


@torch.no_grad__()
def evaluate(model, data_loader, device, use_amp: bool, amp_dtype: torch.dtype, ce_weight: torch.Tensor):
    model.eval()
    total, correct = 0, 0
    tp = fp = tn = fn = 0
    loss_sum = 0.0

    amp_ctx = (
        torch.autocast("cuda", dtype=amp_dtype) if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        graphs = graphs_to_device(batch["graphs"], device)
        labels_t = batch["labels"].to(device, non_blocking=True).long()

        with amp_ctx:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graphs=graphs,
                labels=labels_t,  # 仍传入，但我们会用 class-weight 重算 loss
            )
            logits = out["logits"].float()
            loss = torch.nn.functional.cross_entropy(logits, labels_t, weight=ce_weight).float()

        loss_sum += loss.item() * labels_t.size(0)
        preds = logits.argmax(dim=-1)

        total += labels_t.size(0)
        correct += (preds == labels_t).sum().item()

        tp += ((preds == 1) & (labels_t == 1)).sum().item()
        fp += ((preds == 1) & (labels_t == 0)).sum().item()
        tn += ((preds == 0) & (labels_t == 0)).sum().item()
        fn += ((preds == 0) & (labels_t == 1)).sum().item()

    # reduce across processes if DDP
    if dist.is_initialized():
        tens = torch.tensor([total, correct, tp, fp, tn, fn, loss_sum], dtype=torch.float64, device=device)
        dist.all_reduce(tens, op=dist.ReduceOp.SUM)
        total, correct, tp, fp, tn, fn, loss_sum = [t.item() for t in tens]

    acc = correct / max(1.0, total)
    precision = tp / max(1.0, (tp + fp))
    recall = tp / max(1.0, (tp + fn))
    f1 = 2 * precision * recall / max(1.0, (precision + recall))
    loss_avg = loss_sum / max(1.0, total)
    return acc, precision, recall, f1, loss_avg


# ---------- 新增：带 Humanwrite+patch 的 collate 包装 ----------
def collate_with_humanwrite_and_patch(
    batch: List[Dict[str, Any]],
    tokenizer,
    max_len: int,
) -> Dict[str, Any]:
    """
    在原有 collate_fn 的基础上，
    用 Humanwrite 和 patch 构造新的 few-shot prompt，然后重新 tokenizer。
    假设每个样本包含：
      - sample["Humanwrite"] : str
      - sample["patch"]      : str
    """
    # 先用原 collate_fn 处理 graphs / labels 等
    base_batch = collate_fn(batch, tokenizer, max_len, include_graph_text=False)

    # 构造文本输入
    texts = []
    for ex in batch:
        hw = ex["Humanwrite"]
        patch_text = ex["patch"]
        prompt = build_patch_prompt(hw, patch_text)
        texts.append(prompt)

    enc = tokenizer(
        texts,
        max_length=max_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    base_batch["input_ids"] = enc["input_ids"]
    base_batch["attention_mask"] = enc["attention_mask"]
    return base_batch


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--test_json", type=str, default=None)
    ap.add_argument("--test_cache_dir", type=str, default=None)
    ap.add_argument("--max_len", type=int, default=1280)

    ap.add_argument("--model_id", required=True)
    ap.add_argument("--graph_in_dim", type=int, default=128)
    ap.add_argument("--graph_hid_dim", type=int, default=256)
    ap.add_argument("--graph_out_dim", type=int, default=256)
    ap.add_argument("--graph_layers", type=int, default=2)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--gate_dim", type=int, default=256)
    ap.add_argument("--pissa", action="store_true")
    ap.add_argument("--target", type=str, default="all")

    ap.add_argument("--micro_batch_size", type=int, default=2)
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true")

    # checkpoint
    ap.add_argument("--ckpt_dir", type=str, required=True, help="训练保存目录，内含 pytorch_model.bin 与 tokenizer")

    args = ap.parse_args()

    # DDP
    use_ddp, rank_id, world_size = setup_dist()
    is_main = (rank_id == 0)

    # AMP
    if args.amp_dtype == "bf16":
        use_amp = True
        amp_dtype = torch.bfloat16
    elif args.amp_dtype == "fp16":
        use_amp = True
        amp_dtype = torch.float16
    else:
        use_amp = False
        amp_dtype = torch.float32

    tokenizer = None
    # 如果没有缓存，就需要 tokenizer（从 ckpt_dir 中加载已保存 tokenizer 更优先）
    use_cache = bool(args.test_cache_dir and os.path.exists(args.test_cache_dir))
    if not use_cache:
        tok_src = args.ckpt_dir if os.path.exists(os.path.join(args.ckpt_dir, "tokenizer_config.json")) else args.model_id
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # dataset
    if use_cache:
        test_ds = CachedGraphTextDataset(args.test_cache_dir)
        test_collate = collate_cached
    else:
        test_ds = PatchDataset(args.test_json)
        test_collate = lambda b: collate_with_humanwrite_and_patch(
            b, tokenizer, args.max_len
        )

    # sampler & loader
    test_sampler = DistributedSampler(test_ds, shuffle=False) if use_ddp else None
    test_loader = DataLoader(
        test_ds,
        batch_size=args.micro_batch_size,
        sampler=test_sampler,
        shuffle=False,
        collate_fn=test_collate,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphDoRAModel(
        model_id=args.model_id,
        g_in=args.graph_in_dim, g_hid=args.graph_hid_dim, g_out=args.graph_out_dim,
        n_rels=6, g_layers=args.graph_layers,
        rank=args.rank, alpha=args.alpha, pissa=args.pissa,
        gate_dim=args.gate_dim, target=args.target
    ).to(device)

    ckpt_path = os.path.join(args.ckpt_dir, "pytorch_model.bin")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    if use_ddp:
        model = DDP(
            model,
            device_ids=[int(os.environ.get("LOCAL_RANK", 0))],
            find_unused_parameters=True,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    ce_weight = compute_class_weights_v2(test_loader).to(device)
    if is_main:
        print(f"[info] test class weights = {ce_weight.tolist()}")

    if is_main:
        print("\n[test] evaluating on test set ...")
    acc, prec, rec, f1, test_loss = evaluate(model, test_loader, device, use_amp, amp_dtype, ce_weight)

    if is_main:
        print(f"[test] acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f} loss={test_loss:.4f}")

    if dist.is_initialized():
        ddp_barrier_safe()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
