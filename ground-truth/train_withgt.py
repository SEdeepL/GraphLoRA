# -*- coding: utf-8 -*-
from __future__ import annotations
import os, math, argparse
from contextlib import nullcontext
from typing import Dict, Any, List, Optional
from pathlib import Path

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


# ====== patch prompt 模板 ======
# 用于在补丁前拼接提示文本：
# "Assess whether the patch is correct ..."
PATCH_PROMPT_TEMPLATE = (
    "Assess whether the patch is correct.\n\n"
    "You are given a code patch. Determine whether it is logically correct and truly fixes the intended issue without introducing new bugs.\n\n"
    "Patch:<PATCH_TEXT_HERE>"
)


def build_patch_prompt(patch_text: str) -> str:
    """
    将补丁文本包装为带有提示的完整输入。
    使用示例：
        full_text = build_patch_prompt(patch_text)
    注意：在本训练脚本中，我们在 token 级别额外在最前面拼接 ground-truth 前缀，
    等价于把一个 'ground-truth 示例说明' 放在本提示之前。
    """
    return (
        "Assess whether the patch is correct.\n\n"
        "You are given a code patch. Determine whether it is logically correct and truly fixes the intended issue without introducing new bugs.\n\n"
        f"Patch:{patch_text}"
    )


def ddp_barrier_safe():
    if dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


# ---------------- Ground-truth patch utils ----------------
def clean_patch_text(raw: str) -> str:
    """
    清洗 patch 文本：
    - 每行去掉右侧多余空格
    - 去掉开头/结尾空行
    - 连续空行压缩为 1 行

    不改变行首缩进、不改动代码内部空格。
    """
    lines = raw.splitlines()

    # 去掉每行右侧空格
    lines = [line.rstrip() for line in lines]

    # 去掉开头空行
    while lines and not lines[0].strip():
        lines.pop(0)

    # 去掉结尾空行
    while lines and not lines[-1].strip():
        lines.pop()

    # 连续空行压成一行
    cleaned_lines: List[str] = []
    prev_blank = False
    for line in lines:
        is_blank = (not line.strip())
        if is_blank and prev_blank:
            continue
        cleaned_lines.append(line)
        prev_blank = is_blank

    return "\n".join(cleaned_lines)


def collect_patch_files(patch_dir: Path) -> List[Path]:
    """
    收集目录下所有 .patch 文件，按文件名中的数字或字典序排序：
    1.patch, 2.patch, ..., 6.patch
    """
    patch_files = list(patch_dir.glob("*.patch"))
    if not patch_files:
        raise FileNotFoundError(f"在目录 {patch_dir} 下没有找到任何 *.patch 文件")

    def sort_key(p: Path):
        stem = p.stem
        try:
            return int(stem)
        except ValueError:
            return stem

    patch_files.sort(key=sort_key)
    return patch_files


def build_ground_truth_prefix_text(patch_dir: str) -> str:
    """
    从 patch_dir 中读取 1.patch..N.patch，清洗补丁，
    构造一个统一的 ground-truth 示例前缀文本。

    这个前缀会在 token 级别被拼接到每个样本的最前面，
    因此在“Assess whether the patch is correct.”之前。
    """
    pdir = Path(patch_dir)
    if not pdir.exists():
        raise FileNotFoundError(f"ground-truth patch 目录不存在: {pdir}")
    if not pdir.is_dir():
        raise NotADirectoryError(f"ground-truth patch 路径不是目录: {pdir}")

    patch_files = collect_patch_files(pdir)

    parts: List[str] = []
    parts.append(
        "The following code patches are known correct (ground-truth) fixes.\n"
        "They correctly resolve their original issues and do not introduce new bugs.\n"
        "Use them as references for what a correct patch looks like.\n\n"
    )

    for i, pf in enumerate(patch_files, 1):
        raw = pf.read_text(encoding="utf-8")
        cleaned = clean_patch_text(raw)
        parts.append(f"[Ground-truth example {i}]\n")
        parts.append("This patch is a known correct fix:\n")
        parts.append(cleaned)
        parts.append("\n\n")

    # 特别说明：后面会出现原本的 "Assess whether the patch is correct." 提示
    parts.append(
        "Now you will be given a new patch together with the instruction "
        "\"Assess whether the patch is correct.\" Decide whether that new patch is correct.\n\n"
    )

    return "".join(parts)


def tokenize_ground_truth_prefix(
    prefix_text: str,
    tokenizer: AutoTokenizer,
) -> torch.Tensor:
    """
    把 ground-truth 文本转成 token id 序列（1D tensor），不添加额外 special tokens。
    """
    enc = tokenizer(
        prefix_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    # shape: [1, L] -> [L]
    return enc["input_ids"][0]


def prepend_gt_prefix_to_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gt_token_ids: Optional[torch.Tensor],
    max_len: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    在 batch 的每个样本前拼接相同的 ground-truth 前缀 token 序列。
    如果 max_len 不为 None，则在拼接后截断到 max_len，优先保留“尾部”token，
    也就是尽可能保留当前样本自身的内容，把最前面的前缀截掉（如果太长）。

    input_ids: [B, L]
    attention_mask: [B, L]
    gt_token_ids: [L_gt] 或 None
    """
    if gt_token_ids is None:
        return input_ids, attention_mask

    B, L = input_ids.size()
    device = input_ids.device

    gt_ids = gt_token_ids.to(device=device, non_blocking=True)
    L_gt = gt_ids.size(0)

    # [L_gt] -> [B, L_gt]
    gt_ids_expanded = gt_ids.unsqueeze(0).expand(B, -1)
    gt_attn = torch.ones(
        (B, L_gt),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    new_input_ids = torch.cat([gt_ids_expanded, input_ids], dim=1)      # [B, L_gt + L]
    new_attention_mask = torch.cat([gt_attn, attention_mask], dim=1)   # [B, L_gt + L]

    if (max_len is not None) and (new_input_ids.size(1) > max_len):
        # 保留最后 max_len 个 token，优先保留 patch 本身
        new_input_ids = new_input_ids[:, -max_len:]
        new_attention_mask = new_attention_mask[:, -max_len:]

    return new_input_ids, new_attention_mask


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
def evaluate(
    model,
    val_loader,
    device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    ce_weight: torch.Tensor,
    gt_token_ids: Optional[torch.Tensor] = None,   # 新增：ground-truth 前缀 token
    max_len: Optional[int] = None,                 # 新增：用于截断
):
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

        # 在这里拼接 ground-truth 前缀，
        # 等价于在文本最前面加上一大段 ground-truth 说明 + 示例，
        # 然后才是原来的 "Assess whether the patch is correct." 提示。
        input_ids, attention_mask = prepend_gt_prefix_to_batch(
            input_ids, attention_mask, gt_token_ids, max_len=max_len
        )

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

    # ground-truth patches 目录
    ap.add_argument(
        "--gt_patch_dir",
        type=str,
        default=None,
        help="包含若干 ground-truth 补丁 (1.patch..N.patch) 的目录，作为统一前缀提示拼接到 'Assess whether the patch is correct.' 前面",
    )

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

    # dataset / tokenizer (only when not using cache)
    tokenizer = None
    if not args.train_cache_dir or not os.path.exists(args.train_cache_dir):
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # 如果使用了 ground-truth 前缀但是前面没加载 tokenizer，这里再补一次
    if args.gt_patch_dir is not None and tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # ground-truth 前缀 token
    gt_token_ids: Optional[torch.Tensor] = None
    if args.gt_patch_dir is not None:
        if tokenizer is None:
            raise RuntimeError("需要 tokenizer 才能构造 ground-truth 前缀，但 tokenizer 为 None")
        if is_main:
            print(f"[info] loading ground-truth patches from: {args.gt_patch_dir}")
        gt_prefix_text = build_ground_truth_prefix_text(args.gt_patch_dir)
        gt_token_ids = tokenize_ground_truth_prefix(gt_prefix_text, tokenizer)
        if is_main:
            print(f"[info] ground-truth prefix length: {gt_token_ids.size(0)} tokens")

    # build datasets
    if args.train_cache_dir and os.path.exists(args.train_cache_dir):
        train_ds = CachedGraphTextDataset(args.train_cache_dir)
        train_collate = collate_cached
    else:
        train_ds = PatchDataset(args.train_json)
        train_collate = lambda b: collate_fn(b, tokenizer, args.max_len, include_graph_text=False)

    if args.val_cache_dir and os.path.exists(args.val_cache_dir):
        val_ds = CachedGraphTextDataset(args.val_cache_dir)
        val_collate = collate_cached
    else:
        val_ds = PatchDataset(args.val_json)
        val_collate = lambda b: collate_fn(b, tokenizer, args.max_len, include_graph_text=False)

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
    acc, prec, rec, f1, val_loss = evaluate(
        model,
        val_loader,
        device,
        use_amp,
        amp_dtype,
        ce_weight,
        gt_token_ids=gt_token_ids,
        max_len=args.max_len,
    )
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

            # 在训练时同样在每个样本前拼 ground-truth 前缀
            input_ids, attention_mask = prepend_gt_prefix_to_batch(
                input_ids, attention_mask, gt_token_ids, max_len=args.max_len
            )

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
        acc, prec, rec, f1, val_loss = evaluate(
            model,
            val_loader,
            device,
            use_amp,
            amp_dtype,
            ce_weight,
            gt_token_ids=gt_token_ids,
            max_len=args.max_len,
        )
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
