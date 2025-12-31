"""
nanovllm/utils/context.py - 推理上下文管理

本模块定义了 Attention 计算所需的全局上下文，
使用全局变量模式在模型执行过程中传递状态。

为什么需要 Context：
- Attention 层需要知道当前是 prefill 还是 decode
- 需要访问 KV cache 的 slot 映射和 block 表
- 需要序列长度信息用于 Flash Attention
- 使用全局变量避免修改模型 forward 签名

设计思路：
1. Context 是一个数据类，存储所有 Attention 需要的元信息
2. get_context / set_context 提供全局访问接口
3. 在每次模型 forward 前通过 set_context 设置状态
4. Attention 层通过 get_context 获取状态

使用流程：
1. model_runner 执行前调用 set_context
2. 模型 forward 过程中 Attention 层调用 get_context
3. 执行完成后调用 reset_context 清理状态
"""

from dataclasses import dataclass
import torch


@dataclass
class Context:
    """
    推理上下文数据类
    
    存储当前 forward pass 的所有元信息，
    供 Attention 层使用。
    
    Attributes:
        is_prefill (bool): 当前是否为 prefill 阶段
            - True: prefill（处理完整 prompt）
            - False: decode（自回归生成）
        
        cu_seqlens_q (torch.Tensor): Query 的累积序列长度
            - 形状: [batch_size + 1]
            - 用于 Flash Attention 的变长序列处理
            - 例如 [0, 128, 256] 表示两个长度为 128 的序列
        
        cu_seqlens_k (torch.Tensor): Key 的累积序列长度
            - 形状: [batch_size + 1]
            - prefill 时等于 cu_seqlens_q
            - decode 时是完整上下文长度
        
        max_seqlen_q (int): 最大 Query 序列长度
            - prefill: prompt 最大长度
            - decode: 1
        
        max_seqlen_k (int): 最大 Key 序列长度
            - 完整上下文的最大长度（包括 KV cache）
        
        slot_mapping (torch.Tensor): token 到 KV cache slot 的映射
            - 形状: [num_tokens]
            - 用于 prefill 阶段写入 KV cache
            - slot_id = block_id * block_size + offset
        
        context_lens (torch.Tensor): 每个序列的上下文长度
            - 形状: [batch_size]
            - decode 阶段 PagedAttention 需要
        
        block_tables (torch.Tensor): 每个序列的 block 表
            - 形状: [batch_size, max_num_blocks]
            - decode 阶段用于查找 KV cache 位置
    """
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


# 全局上下文实例
# 使用模块级变量实现单例模式
_CONTEXT = Context()


def get_context():
    """
    获取当前推理上下文
    
    Returns:
        Context: 当前的上下文对象
    
    调用位置：
    - Attention.forward() 中获取 KV cache 信息
    - 判断 prefill/decode 决定使用哪种 Attention 实现
    """
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """
    设置推理上下文
    
    在模型 forward 之前调用，设置当前批次的元信息。
    
    Args:
        is_prefill: 是否为 prefill 阶段
        cu_seqlens_q: Query 累积序列长度
        cu_seqlens_k: Key 累积序列长度
        max_seqlen_q: 最大 Query 长度
        max_seqlen_k: 最大 Key 长度
        slot_mapping: KV cache slot 映射
        context_lens: 上下文长度（decode 用）
        block_tables: Block 表（decode 用）
    
    调用时机：
    - model_runner.forward() 开始时
    - 每个 batch 执行前都需要重新设置
    """
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)


def reset_context():
    """
    重置上下文到默认状态
    
    在执行完成后调用，清理状态避免内存泄漏。
    """
    global _CONTEXT
    _CONTEXT = Context()

