"""
nanovllm/layers/rotary_embedding.py - 旋转位置编码（RoPE）

本模块实现了 Rotary Position Embedding（RoPE），
这是现代 LLM（如 LLaMA、Qwen、GPT-NeoX）使用的位置编码方式。

RoPE 原理：
- 将位置信息通过旋转矩阵编码到 Q 和 K 中
- 使用复数表示简化计算：将向量解释为复数，乘以旋转因子
- 天然支持相对位置：Q 和 K 的内积只依赖于相对位置

数学公式：
对于位置 m 的 token，head 维度上的第 i 对元素 (x_2i, x_2i+1)：
    x'_2i = x_2i * cos(m * θ_i) - x_2i+1 * sin(m * θ_i)
    x'_2i+1 = x_2i+1 * cos(m * θ_i) + x_2i * sin(m * θ_i)

其中 θ_i = base^(-2i/d)，d 是 head 维度

优势：
- 位置信息平滑融入注意力计算
- 支持无限外推（但效果会逐渐衰减）
- 计算高效：只需要预计算 cos 和 sin

实现优化：
- 预计算所有位置的 cos 和 sin（减少运行时开销）
- 使用 @torch.compile 编译
- 向量化计算（无循环）
"""

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    应用旋转位置编码
    
    Args:
        x: 输入张量 [..., head_dim]
        cos: 余弦值 [..., head_dim/2]
        sin: 正弦值 [..., head_dim/2]
    
    Returns:
        torch.Tensor: 应用 RoPE 后的张量
    
    计算步骤：
    1. 将 x 分成两半：x1, x2
    2. 旋转变换：
       y1 = x1 * cos - x2 * sin
       y2 = x2 * cos + x1 * sin
    3. 拼接回原形状
    """
    # 将最后一维分成两半
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # 应用旋转变换
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 拼接并转回原始类型
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码模块
    
    预计算所有可能位置的 cos 和 sin 值，
    推理时只需要查表。
    
    Attributes:
        head_size (int): head 维度
        cos_sin_cache (torch.Tensor): 预计算的 [cos, sin] 缓存
            形状: [max_position, 1, head_size]（1 是 head 维度的占位）
    """

    def __init__(
        self,
        head_size: int,                # head 维度
        rotary_dim: int,               # 旋转编码的维度（通常等于 head_size）
        max_position_embeddings: int,  # 最大位置（预计算范围）
        base: float,                   # 基础频率（10000 或更高）
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size  # 当前实现要求全维度旋转
        
        # 计算频率：inv_freq[i] = base^(-2i/d)
        # 这决定了不同维度的旋转速度
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        
        # 计算所有位置的相位角
        # t: [max_position]，位置索引
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # freqs: [max_position, rotary_dim/2]
        # freqs[m, i] = m * inv_freq[i]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        
        # 计算 cos 和 sin
        cos = freqs.cos()
        sin = freqs.sin()
        
        # 缓存格式: [max_position, 1, head_size]
        # 拼接 cos 和 sin，后续一次取出
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # 注册为 buffer（不参与梯度计算，但会随模型保存/加载）
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,  # 位置索引 [num_tokens]
        query: torch.Tensor,      # Q [num_tokens, num_heads, head_dim]
        key: torch.Tensor,        # K [num_tokens, num_kv_heads, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        应用旋转位置编码到 Q 和 K
        
        Args:
            positions: 每个 token 的位置索引
            query: Query 张量
            key: Key 张量
        
        Returns:
            (rotated_query, rotated_key): 应用 RoPE 后的 Q 和 K
        """
        # 从缓存中查找 cos 和 sin
        cos_sin = self.cos_sin_cache[positions]  # [num_tokens, 1, head_size]
        # 分割 cos 和 sin
        cos, sin = cos_sin.chunk(2, dim=-1)
        # 应用到 Q 和 K
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """
    获取或创建 RoPE 模块（带缓存）
    
    使用 lru_cache 确保相同参数只创建一个实例，
    节省内存并支持权重共享。
    
    Args:
        head_size: head 维度
        rotary_dim: 旋转维度
        max_position: 最大位置
        base: 基础频率
        rope_scaling: RoPE 缩放配置（当前未实现）
    
    Returns:
        RotaryEmbedding: RoPE 模块实例
    
    注意：
    - 当前不支持 rope_scaling（用于超长上下文）
    - 如果需要支持，需要实现 LinearScaling、DynamicNTK 等变体
    """
    assert rope_scaling is None  # 暂不支持缩放
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
