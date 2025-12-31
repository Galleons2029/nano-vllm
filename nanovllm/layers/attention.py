"""
nanovllm/layers/attention.py - 注意力计算模块

本模块实现了高效的注意力计算，核心功能包括：
1. KV 缓存写入（使用 Triton kernel）
2. Prefill 阶段的变长注意力（使用 Flash Attention）
3. Decode 阶段的 KV 缓存注意力（使用 Flash Attention with KV cache）
4. 前缀缓存支持

关键技术：
- Flash Attention: O(N) 内存复杂度的注意力计算
- Triton Kernel: 高效的 KV 缓存写入
- 分页 KV 缓存: 支持块化的 KV 存储和前缀共享

两种注意力模式：
1. Prefill（变长注意力）:
   - 处理完整的 prompt
   - Q, K, V 长度可以不同（前缀缓存时）
   - 使用 flash_attn_varlen_func

2. Decode（KV 缓存注意力）:
   - 每次只处理一个新 token
   - Q 长度为 1，K/V 从缓存读取
   - 使用 flash_attn_with_kvcache
"""

import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,          # Key 张量指针
    key_stride,       # Key 的行跨度
    value_ptr,        # Value 张量指针
    value_stride,     # Value 的行跨度
    k_cache_ptr,      # K 缓存指针
    v_cache_ptr,      # V 缓存指针
    slot_mapping_ptr, # 槽位映射指针
    D: tl.constexpr,  # 每行的元素数（num_heads * head_dim）
):
    """
    Triton kernel：将 K, V 写入缓存
    
    每个 program 处理一个 token 的 K 和 V。
    
    Args:
        key_ptr, value_ptr: 源 K, V 张量
        key_stride, value_stride: 行跨度
        k_cache_ptr, v_cache_ptr: 目标缓存
        slot_mapping_ptr: 每个 token 对应的缓存槽位
        D: 每个 token 的总维度（num_heads * head_dim）
    
    内存布局：
    - Key/Value: [num_tokens, num_heads, head_dim]
    - Cache: [num_blocks, block_size, num_heads, head_dim]
    
    工作原理：
    1. 获取当前 program 的 token 索引
    2. 从 slot_mapping 获取目标槽位
    3. 加载 K, V 数据
    4. 写入到缓存的对应位置
    """
    # 当前 token 索引
    idx = tl.program_id(0)
    # 获取目标槽位（-1 表示跳过，CUDA Graph 预留空间）
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return  # 无效槽位，跳过
    
    # 计算源数据偏移
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    
    # 加载 K, V 数据
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    # 计算目标缓存偏移并写入
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    将 K, V 写入缓存（Python 封装）
    
    Args:
        key: Key 张量 [N, num_heads, head_dim]
        value: Value 张量 [N, num_heads, head_dim]
        k_cache: K 缓存 [num_blocks * block_size, num_heads, head_dim]
        v_cache: V 缓存 [num_blocks * block_size, num_heads, head_dim]
        slot_mapping: 槽位映射 [N]
    
    注意：
    - 输入张量必须是 contiguous（stride(-1) == 1）
    - 使用 Triton kernel 并行处理所有 token
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    
    # 验证张量布局
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    
    # 启动 Triton kernel，每个 token 一个 program
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """
    注意力计算模块
    
    封装了 prefill 和 decode 两种模式的注意力计算，
    自动根据上下文选择合适的实现。
    
    Attributes:
        num_heads (int): Query 头数
        head_dim (int): 每个头的维度
        scale (float): 注意力缩放因子（1/sqrt(head_dim)）
        num_kv_heads (int): Key/Value 头数（GQA 支持）
        k_cache, v_cache: KV 缓存张量，由 ModelRunner 设置
    """

    def __init__(
        self,
        num_heads,     # Q 头数
        head_dim,      # 头维度
        scale,         # 缩放因子
        num_kv_heads,  # KV 头数
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # KV 缓存，初始为空，由 ModelRunner.allocate_kv_cache 设置
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        前向传播
        
        Args:
            q: Query 张量 [N, num_heads, head_dim]
            k: Key 张量 [N, num_kv_heads, head_dim]
            v: Value 张量 [N, num_kv_heads, head_dim]
        
        Returns:
            torch.Tensor: 注意力输出 [N, num_heads, head_dim]
        
        计算流程：
        1. 将新的 K, V 写入缓存
        2. 根据模式（prefill/decode）调用对应的 Flash Attention 函数
        """
        # 获取当前上下文（包含 slot_mapping, cu_seqlens 等）
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # 将 K, V 写入缓存（如果缓存已分配）
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:
            # ==================== Prefill 模式 ====================
            # 使用变长注意力处理完整 prompt
            
            if context.block_tables is not None:
                # 有前缀缓存：K, V 从缓存读取
                k, v = k_cache, v_cache
            
            # flash_attn_varlen_func 支持变长序列的批处理
            # cu_seqlens: 累积序列长度，用于定位每个序列
            # max_seqlen: 最大序列长度，用于内存分配
            # block_table: 用于前缀缓存时的块索引
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:
            # ==================== Decode 模式 ====================
            # 每个序列只有一个新 token，使用 KV 缓存注意力
            
            # flash_attn_with_kvcache 专门为 decode 优化
            # q.unsqueeze(1): [N, num_heads, head_dim] -> [N, 1, num_heads, head_dim]
            # cache_seqlens: 每个序列的上下文长度
            # block_table: 块索引，用于定位 KV 缓存
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
