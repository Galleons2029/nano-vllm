"""
nanovllm/layers/activation.py - 激活函数模块

本模块实现了 Transformer 模型中使用的激活函数。
主要包含 SwiGLU（SiLU-Gated Linear Unit）激活函数的实现。

SwiGLU 是 GLU 变体，在 LLaMA、Qwen 等模型中广泛使用：
- 原始 GLU: GLU(x, y) = x ⊗ σ(y)，其中 σ 是 sigmoid
- SwiGLU: SwiGLU(x, y) = x ⊗ SiLU(y)，使用 SiLU 替代 sigmoid

SiLU（Swish）函数：
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

性能优化：
- 使用 @torch.compile 装饰器编译，减少 Python 开销
- 在单个 kernel 中完成 chunk + silu + mul，减少内存访问
"""

import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """
    SiLU + 元素乘法激活函数
    
    计算: SiLU(x) * y，其中 [x, y] = input.chunk(2, -1)
    
    这是 SwiGLU MLP 的核心操作：
    输入: [gate, up] 拼接的张量，形状 [..., 2 * intermediate_size]
    输出: SiLU(gate) * up，形状 [..., intermediate_size]
    
    使用场景：
    在 Qwen3MLP 中，gate_up_proj 输出 [gate, up] 拼接的张量，
    通过本模块计算激活值后传给 down_proj。
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，最后一维是偶数（会被对半分割）
        
        Returns:
            torch.Tensor: SiLU(gate) * up
        
        计算步骤:
        1. x.chunk(2, -1) -> 将最后一维对半分成 [gate, up]
        2. F.silu(gate) -> 应用 SiLU 激活函数
        3. silu_gate * up -> 元素乘法
        """
        # 沿最后一维分成两半
        x, y = x.chunk(2, -1)
        # SiLU 激活后与另一半相乘
        return F.silu(x) * y
