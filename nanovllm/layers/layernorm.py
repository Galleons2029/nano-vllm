"""
nanovllm/layers/layernorm.py - 归一化层实现

本模块实现了 RMSNorm（Root Mean Square Layer Normalization）。
RMSNorm 是 LayerNorm 的简化版本，在 LLaMA、Qwen 等模型中广泛使用。

RMSNorm vs LayerNorm:
- LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
- RMSNorm:   y = x / sqrt(mean(x^2) + eps) * weight

优势：
- 计算更简单：不需要计算均值和偏置
- 效果相当：在实践中性能与 LayerNorm 相近
- 更快：减少了计算量和内存访问

本实现的优化：
- 使用 @torch.compile 编译优化
- 支持 fused add + norm（减少内存访问）
- 使用 float32 进行中间计算（数值稳定性）
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    对输入进行均方根归一化：
    y = x / RMS(x) * weight
    RMS(x) = sqrt(mean(x^2) + eps)
    
    特点：
    - 没有偏置项（不像 LayerNorm）
    - 不减去均值（只除以 RMS）
    - 支持与残差连接融合
    
    Attributes:
        eps (float): 防止除零的小常数
        weight (nn.Parameter): 可学习的缩放参数
    """

    def __init__(
        self,
        hidden_size: int,    # 归一化维度
        eps: float = 1e-6,   # 数值稳定性常数
    ) -> None:
        super().__init__()
        self.eps = eps
        # 初始化为全 1（不改变分布）
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        基本的 RMSNorm 前向传播
        
        Args:
            x: 输入张量 [..., hidden_size]
        
        Returns:
            torch.Tensor: 归一化后的张量
        
        计算步骤:
        1. 转换为 float32（数值稳定性）
        2. 计算 x^2 的均值
        3. 计算 rsqrt(var + eps)
        4. 归一化并乘以 weight
        5. 转回原始 dtype
        """
        orig_dtype = x.dtype
        # 使用 float32 计算以保证精度
        x = x.float()
        # 计算方差（x^2 的均值）
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # 原地归一化（节省内存）
        x.mul_(torch.rsqrt(var + self.eps))
        # 转回原始类型并乘以 weight
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        融合残差加法的 RMSNorm
        
        计算: norm(x + residual)，同时返回更新后的 residual
        
        这是 Pre-Norm Transformer 中常用的模式：
        residual = x + residual  (先加残差)
        output = norm(residual)  (再归一化)
        
        Args:
            x: 输入张量（上一层的输出）
            residual: 累积的残差
        
        Returns:
            (normalized, new_residual): 
                - normalized: 归一化后的结果
                - new_residual: 更新后的残差（供下一层使用）
        
        优势：
        - 减少内存访问：一次遍历完成 add 和 norm
        - 减少中间变量：不需要额外存储 x + residual
        """
        orig_dtype = x.dtype
        # 融合加法
        x = x.float().add_(residual.float())
        # 保存新的残差（用于下一层）
        residual = x.to(orig_dtype)
        # RMSNorm 计算
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        统一的前向传播接口
        
        Args:
            x: 输入张量
            residual: 可选的残差张量
        
        Returns:
            - 无 residual: 返回归一化后的张量
            - 有 residual: 返回 (归一化结果, 更新后的残差)
        
        使用场景：
        - 第一层/独立使用：forward(x)
        - Pre-Norm 中间层：forward(x, residual)
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
