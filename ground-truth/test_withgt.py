# -*- coding: utf-8 -*-
from __future__ import annotations
import os, argparse
from contextlib import nullcontext
from typing import Dict, Any, List, Optional
from pathlib import Path

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
    """统计标签分布，返回 shape=[2] 的权重（小类更大权重），做上限裁剪避免过大。"""
    counts = torch.zeros(2, dtype=torch.long)
    for batch in loader:
        y = batch["labels"]
        counts += torch.bincount(y, minlength=2)
    total = counts.sum().item()
    weights = total / (2.0 * counts.clamp(min=1)).float()
    weights = torch.clamp(weights, max=5.0)
    return weights


# ================= Ground-truth patch utils =================

def clean_patch_text(raw: str) -> str:
    """
    清洗 patch 文本：
    - 每行去掉右侧多余空格
    - 去掉开头/结尾空行
    - 连续空行压缩为 1 行
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
    1.patch, 2.patch, ..., N.patch
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
    也就是在数据中原本的 "Assess whether the patch is correct." 提示之前。
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

    parts.append(
        "Now you will be given a new patch and asked to assess whether the patch is correct.\n\n"
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
    尽量保留当前样本自己的 patch 内容。
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
        # 保留最后 max_len 个 token
        new_input_ids = new_input_ids[:, -max_len:]
        new_attention_mask = new_attention_mask[:, -max_len:]

    return new_input_ids, new_attention_mask


# ================= Evaluate =================

@torch.no_grad()
def evaluate(
    model,
    data_loader,
    device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    ce_weight: torch.Tensor,
    gt_token_ids: Optional[torch.Tensor] = None,   # 新增：ground-truth 前缀
    max_len: Optional[int] = None,                 # 新增：长度截断
):
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

        # 在这里把 ground-truth 前缀拼到每个样本的最前面
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

    # ground-truth patches 目录（新增）
    ap.add_argument(
        "--gt_patch_dir",
        type=str,
        default=None,
        help="包含若干 ground-truth 补丁 (1.patch..N.patch) 的目录，作为统一前缀提示拼接到原始提示前面",
    )

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

    # tokenizer：如果没有 cache 或者需要 ground-truth 前缀，则需要 tokenizer
    tokenizer = None
    need_tokenizer = (
        (not args.test_cache_dir or not os.path.exists(args.test_cache_dir))
        or (args.gt_patch_dir is not None)
    )
    if need_tokenizer:
        tok_src = args.ckpt_dir if os.path.exists(os.path.join(args.ckpt_dir, "tokenizer_config.json")) else args.model_id
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
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

    # dataset
    if args.test_cache_dir and os.path.exists(args.test_cache_dir):
        test_ds = CachedGraphTextDataset(args.test_cache_dir)
        test_collate = collate_cached
    else:
        test_ds = PatchDataset(args.test_json)
        test_collate = lambda b: collate_fn(b, tokenizer, args.max_len, include_graph_text=False)

    # sampler & loader
    test_sampler = DistributedSampler(test_ds, shuffle=False) if use_ddp else None
    test_loader = DataLoader(
        test_ds,
        batch_size=args.micro_batch_size,
        sampler=test_sampler,
        shuffle=False if test_sampler is not None else False,
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

    # 把 gt_token_ids 搬到 device（可选）
    if gt_token_ids is not None:
        gt_token_ids = gt_token_ids.to(device=device, non_blocking=True)

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
    acc, prec, rec, f1, test_loss = evaluate(
        model,
        test_loader,
        device,
        use_amp,
        amp_dtype,
        ce_weight,
        gt_token_ids=gt_token_ids,
        max_len=args.max_len,
    )

    if is_main:
        print(f"[test] acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f} loss={test_loss:.4f}")

    if dist.is_initialized():
        ddp_barrier_safe()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
