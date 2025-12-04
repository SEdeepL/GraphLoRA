# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- PiSSA init (SVD-based) --------------------
@torch.no_grad()
def _pissa_init_(up: nn.Linear, down: nn.Linear, parent_weight: torch.Tensor, alpha: float, rank: int):
    """
    用 SVD 对父线性层 W 做低秩近似，初始化 LoRA 的 up/down，使得：
        up.weight @ down.weight  ≈  W_lowrank
    约定：
        W.shape = [out, in]
        down: in -> r   （down.weight.shape = [r, in]）
        up  : r  -> out （up.weight.shape   = [out, r]）
    选择：
        down.weight = sqrt(S) @ V^T
        up.weight   = U @ sqrt(S)
    这样 (U sqrt(S)) @ (sqrt(S) V^T) = U S V^T = W_lowrank
    再按 LoRA 习惯乘上 alpha/rank（可等效吸收到 up.weight 中）。
    """
    W = parent_weight.detach().float()
    out, in_ = W.shape
    r = min(rank, min(out, in_))
    if r <= 0:
        up.weight.zero_()
        down.weight.zero_()
        if up.bias is not None:
            up.bias.zero_()
        return

    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # U:[out,r0], S:[r0], Vh:[r0,in]
    except Exception:
        up.weight.zero_()
        down.weight.zero_()
        if up.bias is not None:
            up.bias.zero_()
        return

    U_r = U[:, :r]                         # [out, r]
    S_r = S[:r]                            # [r]
    Vh_r = Vh[:r, :]                       # [r, in]

    S_r_clamped = torch.clamp(S_r, min=0)
    sqrtS = torch.sqrt(S_r_clamped)        # [r]

    # down.weight: [r, in] = diag(sqrtS) @ Vh_r
    A = (sqrtS.unsqueeze(1) * Vh_r)        # [r, in]
    # up.weight  : [out, r] = U_r @ diag(sqrtS)
    B = (U_r * sqrtS.unsqueeze(0))         # [out, r]

    scaling = alpha / max(1, rank)
    up.weight.zero_()
    up.weight[:, :r].copy_(B.to(dtype=up.weight.dtype) * scaling)
    down.weight.zero_()
    down.weight[:r, :].copy_(A.to(dtype=down.weight.dtype))

    if up.bias is not None:
        up.bias.zero_()


# -------------------- DoRA + (可选) PiSSA 适配器 --------------------
class DoRAPiSSAAdapter(nn.Module):
    """
    DoRA + 可选 PiSSA 的线性适配器：
      y = parent(x) + ( gate(z) * lora_up(lora_down(x)) )
    其中 gate(z) 为条件门控（逐通道缩放），z 通常由图-文融合得到的向量。
    """
    def __init__(
        self,
        parent_linear: nn.Linear,
        rank: int,
        alpha: float,
        gate_dim: int,
        use_pissa: bool = False,
        init_std: float = 1e-4,
    ):
        super().__init__()
        self.parent = parent_linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)
        self.use_pissa = bool(use_pissa)

        in_features = parent_linear.in_features
        out_features = parent_linear.out_features

        # LoRA 分支
        self.lora_down = nn.Linear(in_features, self.rank, bias=False)
        self.lora_up   = nn.Linear(self.rank, out_features, bias=False)

        # 门控分支（逐通道）
        self.gate = nn.Linear(gate_dim, out_features, bias=True)

        # 初始化
        if self.use_pissa:
            _pissa_init_(self.lora_up, self.lora_down, parent_linear.weight.data, alpha=self.alpha, rank=self.rank)
        else:
            nn.init.normal_(self.lora_down.weight, mean=0.0, std=init_std)
            nn.init.zeros_(self.lora_up.weight)
        nn.init.zeros_(self.gate.bias)

        # 冻结父层
        for p in self.parent.parameters():
            p.requires_grad = False

        # 关键修复：LoRA 与 gate 的 dtype/device 与父层对齐，避免 AMP 下 dtype 不匹配
        parent_dtype = self.parent.weight.dtype
        parent_device = self.parent.weight.device
        self.lora_down.to(device=parent_device, dtype=parent_dtype)
        self.lora_up.to(device=parent_device, dtype=parent_dtype)
        self.gate.to(device=parent_device, dtype=parent_dtype)

        # 运行期存放上下文 z（由外部在前向前写入）
        self._ctx_z: Optional[torch.Tensor] = None

    def set_ctx_z(self, z: Optional[torch.Tensor]):
        """供外部写入/清理当前 batch 的条件向量 z。"""
        self._ctx_z = z

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        base = self.parent(x)

        # LoRA 路径（dtype/device 已与父层对齐）
        lora_out = self.lora_up(self.lora_down(x)) * self.scaling  # [*, out_features]

        # 门控：优先使用传进来的 z；否则尝试读取 ctx；都没有则用全 1
        if z is None:
            z = getattr(self, "_ctx_z", None)
        if z is None:
            # 与 base 同形状的全 1（逐通道），确保可广播
            gate = torch.ones_like(base)
        else:
            gate = torch.sigmoid(self.gate(z))  # 期望 [B, out_features]
            # 若 base 是 [B, T, O] 而 gate 是 [B, O]，为跨 T 广播需在 dim=1 增一维
            if base.dim() == 3 and gate.dim() == 2 and base.size(0) == gate.size(0) and base.size(-1) == gate.size(-1):
                gate = gate.unsqueeze(1)

        # 统一 dtype（极端情况下 z/gate 可能在 fp32，base 在 bf16）
        if gate.dtype != base.dtype:
            gate = gate.to(base.dtype)
        if lora_out.dtype != base.dtype:
            lora_out = lora_out.to(base.dtype)

        return base + gate * lora_out


# -------------------- 包装 Linear：用适配器替换 --------------------
class _Wrap(nn.Module):
    """
    将原始 Linear 与 Adapter 打包，保持参数注册与前向路径：
      forward(x):
         z = adapter._ctx_z （由外部写入）
         return adapter(x, z)
    """
    def __init__(self, base: nn.Linear, adapter: DoRAPiSSAAdapter):
        super().__init__()
        self.base = base
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = getattr(self.adapter, "_ctx_z", None)
        return self.adapter(x, z)


# -------------------- 选择哪些 Linear 需要注入 --------------------
_ATT_KEYS = {"q_proj", "k_proj", "v_proj", "o_proj"}          # 自注意力
_FFN_KEYS = {"up_proj", "down_proj", "gate_proj"}             # MLP

def _should_adapt(attr_name: str, target: str) -> bool:
    """
    target: "attn" | "ffn" | "all"
    """
    if target == "attn":
        return attr_name in _ATT_KEYS
    if target == "ffn":
        return attr_name in _FFN_KEYS
    return attr_name in _ATT_KEYS or attr_name in _FFN_KEYS


def _iter_named_linears(module: nn.Module) -> Iterable[Tuple[str, nn.Module, str, nn.Linear]]:
    """
    遍历 module 下的所有 Linear，返回 (全名, 父模块, 属性名, 线性层)
    """
    for name, sub in module.named_modules():
        for child_name, child in sub.named_children():
            if isinstance(child, nn.Linear):
                full_name = f"{name}.{child_name}" if name else child_name
                yield full_name, sub, child_name, child


# -------------------- 注入入口（兼容 pissa_init 老参数名） --------------------
def wrap_llama_with_adapters(
    llm: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    gate_dim: int = 256,
    use_pissa: bool = False,
    target: str = "all",
    **kwargs,
) -> List[DoRAPiSSAAdapter]:
    """
    在 LLaMA（HF Transformers 架构）中，为选择的 Linear 注入 DoRA(+PiSSA) 适配器。
    兼容旧调用：如果传入 pissa_init=...，将与 use_pissa 做逻辑或。
    返回注入的 adapter 列表，便于后续统一写入 ctx z 或保存/加载。
    """
    # 兼容老参数名
    if "pissa_init" in kwargs:
        use_pissa = bool(use_pissa or kwargs["pissa_init"])

    adapters: List[DoRAPiSSAAdapter] = []

    for full, parent_module, attr, linear in _iter_named_linears(llm):
        if not _should_adapt(attr, target):
            continue

        # 构造 adapter（会自动把 dtype/device 与父层对齐，并冻结父层）
        adp = DoRAPiSSAAdapter(
            parent_linear=linear,
            rank=rank,
            alpha=alpha,
            gate_dim=gate_dim,
            use_pissa=use_pissa,
        )
        adapters.append(adp)

        # 冻结原始 Linear（双保险）
        for p in linear.parameters():
            p.requires_grad = False

        # 用包装器替换
        setattr(parent_module, attr, _Wrap(linear, adp))

    return adapters
