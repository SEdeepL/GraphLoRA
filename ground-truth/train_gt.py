# -*- coding: utf-8 -*-
from __future__ import annotations
import os, math, argparse
from contextlib import nullcontext
from typing import Dict, Any, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# ====== datasets ======
from graph_lora.data.cached_dataset import CachedGraphTextDataset, collate_cached
from graph_lora.data.json_dataset import PatchDataset, collate_fn

# ====== model ======
from graph_lora.models.model import GraphDoRAModel

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup


# ====== 新的 patch prompt 模板（Humanwrite + patch，few-shot） ======
# 说明：
# - Humanwrite：某个已知正确补丁的 ground-truth 示例说明（文本）
# - patch_text：当前需要评估的候选补丁
#
# Prompt 结构：
#   You are a model responsible for assessing patch correctness.
#
#   Below is one ground-truth example. This patch is correct.
#   <Humanwrite>
#
#   Assess whether the patch is correct:
#   <patch_text>
#
# 注意：这里的 “This patch is correct.” 只指上面的 ground-truth 示例（Humanwrite 对应的补丁），
# 下面真正要判断的是 “Assess whether the patch is correct:” 这一段的 patch_text。

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


# ---------------- Utils ----------------
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
            out[k] = [int(x) for x in v]  # 保持为 List[int]
        else:
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device, non_blocking=True)
            else:
                # 遇到 list/tuple 时安全张量化
                if k == "edge_index":
                    t = torch.tensor(v, dtype=torch.long)
                elif k == "edge_type":
                    t = torch.tensor(v, dtype=torch.long)
                else:  # node_feat
                    t = torch.tensor(v, dtype=torch.float32)
                out[k] = t.to(device, non_blocking=True)
    return out


def compute_class_weights_v2(loader) -> torch.Tensor:
    """单次遍历训练集统计标签分布，返回 shape=[2] 的权重（小类更大权重），做上限裁剪避免过大。"""
    counts = torch.zeros(2, dtype=torch.long)
    for batch in loader:
        y = batch["labels"]
        counts += torch.bincount(y, minlength=2)
    total = counts.sum().item()
    weights = total / (2.0 * counts.clamp(min=1)).float()   # 反频率
    weights = torch.clamp(weights, max=5.0)                 # 上限防止过大
    return weights


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, val_loader, device, use_amp: bool, amp_dtype: torch.dtype, ce_weight: torch.Tensor):
    model.eval()
    total, correct = 0, 0
    # for precision/recall/f1 (binary)
    tp = fp = tn = fn = 0
    loss_sum = 0.0

    amp_ctx = (
        torch.autocast("cuda", dtype=amp_dtype) if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    for batch in val_loader:
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

        # binary metrics (labels {0,1})
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


def save_ckpt(model, tokenizer, path, is_main):
    if is_main:
        os.makedirs(path, exist_ok=True)
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state, os.path.join(path, "pytorch_model.bin"))
        if tokenizer is not None:
            tokenizer.save_pretrained(path)


# ---------------- 新的 collate 包装函数：插入 Humanwrite + patch 的 few-shot prompt ----------------
def collate_with_humanwrite_and_patch(
    batch: List[Dict[str, Any]],
    tokenizer,
    max_len: int,
) -> Dict[str, Any]:
    """
    在原有 collate_fn 的基础上，
    用 Humanwrite 和 patch 构造新的 few-shot prompt，然后重新 tokenizer。
    假设每个样本里包含：
      - sample["Humanwrite"] : str
      - sample["patch"]      : str
    以及 graph/labels 等字段由原始 collate_fn 负责整理。
    """
    # 先用原来的 collate_fn 做图和标签等的拼接
    base_batch = collate_fn(batch, tokenizer, max_len, include_graph_text=False)

    # 重新构造文本输入
    texts = []
    for ex in batch:
        # 根据你 JSON 的字段名来取，如果 Dataset 里改成其它名字，这里同步一下即可
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


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    # data / cache
    ap.add_argument("--train_json", type=str, default=None)
    ap.add_argument("--val_json", type=str, default=None)
    ap.add_argument("--train_cache_dir", type=str, default=None, help="优先使用缓存分片")
    ap.add_argument("--val_cache_dir", type=str, default=None)
    ap.add_argument("--max_len", type=int, default=1280)

    # model
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

    # train
    ap.add_argument("--micro_batch_size", type=int, default=2)
    ap.add_argument("--global_batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)          # 更稳的默认学习率
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--log_interval", type=int, default=50)

    # amp / perf
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true")

    # misc
    ap.add_argument("--output_dir", type=str, default="result")
    args = ap.parse_args()

    # DDP
    use_ddp, rank_id, world_size = setup_dist()
    is_main = (rank_id == 0)

    # AMP
    if args.amp_dtype == "bf16":
        use_amp = True
        amp_dtype = torch.bfloat16
        scaler = None
    elif args.amp_dtype == "fp16":
        use_amp = True
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:  # fp32
        use_amp = False
        amp_dtype = torch.float32
        scaler = None

    # ---------- tokenizer ----------
    use_train_cache = bool(args.train_cache_dir and os.path.exists(args.train_cache_dir))
    use_val_cache = bool(args.val_cache_dir and os.path.exists(args.val_cache_dir))
    tokenizer = None
    # 只要 train 或 val 有一边是 JSON（非 cache），就需要 tokenizer
    if not (use_train_cache and use_val_cache):
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # ---------- build datasets ----------
    if use_train_cache:
        # 缓存分片下，沿用原来的 cached 逻辑（缓存里已经包含 input_ids 等）
        train_ds = CachedGraphTextDataset(args.train_cache_dir)
        train_collate = collate_cached
    else:
        # 新的 JSON + few-shot prompt 逻辑
        train_ds = PatchDataset(args.train_json)
        train_collate = lambda b: collate_with_humanwrite_and_patch(
            b, tokenizer, args.max_len
        )

    if use_val_cache:
        val_ds = CachedGraphTextDataset(args.val_cache_dir)
        val_collate = collate_cached
    else:
        val_ds = PatchDataset(args.val_json)
        val_collate = lambda b: collate_with_humanwrite_and_patch(
            b, tokenizer, args.max_len
        )

    # samplers
    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None

    # loaders
    num_workers = args.num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=args.micro_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=train_collate,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.micro_batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=val_collate,
        num_workers=max(1, num_workers // 2),
        pin_memory=args.pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=args.prefetch_factor if num_workers > 0 else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphDoRAModel(
        model_id=args.model_id,
        g_in=args.graph_in_dim, g_hid=args.graph_hid_dim, g_out=args.graph_out_dim,
        n_rels=6, g_layers=args.graph_layers,
        rank=args.rank, alpha=args.alpha, pissa=args.pissa,
        gate_dim=args.gate_dim, target=args.target
    ).to(device)

    if use_ddp:
        model = DDP(
            model,
            device_ids=[int(os.environ.get("LOCAL_RANK", 0))],
            find_unused_parameters=True,           # 冻结 LLM 后更稳妥
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    # optimizer / scheduler
    trainable = [p for p in model.parameters() if p.requires_grad]
    if is_main:
        print(f"[info] trainable params: {sum(p.numel() for p in trainable):,}")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # 先算类别权重（在 GPU 前向之外统计，成本低）
    ce_weight = compute_class_weights_v2(train_loader)
    if is_main:
        print(f"[info] class weights = {ce_weight.tolist()}")
    ce_weight = ce_weight.to(device)

    # steps
    accum = max(1, args.global_batch_size // (args.micro_batch_size * world_size))
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    t_total = steps_per_epoch * args.epochs
    warmup_steps = int(t_total * args.warmup_ratio)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    # eval before train
    if is_main:
        print("\n[eval] before training ...")
    acc, prec, rec, f1, val_loss = evaluate(model, val_loader, device, use_amp, amp_dtype, ce_weight)
    if is_main:
        print(f"[val] acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f} loss={val_loss:.4f}\n")

    # training
    global_step = 0
    for epoch in range(args.epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        if is_main:
            print(f"[epoch {epoch}] start ...")

        model.train()
        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            graphs = graphs_to_device(batch["graphs"], device)
            labels_t = batch["labels"].to(device, non_blocking=True).long()

            need_sync = ((step + 1) % accum == 0)
            ddp_sync_ctx = nullcontext() if (not use_ddp or need_sync) else model.no_sync()

            with ddp_sync_ctx:
                amp_ctx = (
                    torch.autocast("cuda", dtype=amp_dtype)
                    if (use_amp and device.type == "cuda")
                    else nullcontext()
                )
                with amp_ctx:
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        graphs=graphs,
                        labels=labels_t,  # 仍传入，但我们用 class-weight 重算 loss
                    )
                    logits = out["logits"].float()
                    loss = torch.nn.functional.cross_entropy(logits, labels_t, weight=ce_weight).float() / accum

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if need_sync:
                # 梯度裁剪（先 unscale 再裁剪）
                if scaler is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

                if is_main and (global_step % args.log_interval == 0):
                    cur_loss = (loss * accum).detach().float().item()  # 打印未除累积的 loss
                    print(f"[epoch {epoch}] step {global_step}/{t_total}  loss={cur_loss:.4f}")

        # ---- epoch end: evaluate + save ----
        if is_main:
            print(f"\n[eval] after epoch {epoch} ...")
        acc, prec, rec, f1, val_loss = evaluate(model, val_loader, device, use_amp, amp_dtype, ce_weight)
        if is_main:
            print(f"[val] epoch={epoch} acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f} loss={val_loss:.4f}\n")
            save_ckpt(model, tokenizer, os.path.join(args.output_dir, f"epoch{epoch}"), True)

    # final save
    if use_ddp:
        ddp_barrier_safe()
    save_ckpt(model, tokenizer, args.output_dir, is_main)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
