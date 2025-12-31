"""
nanovllm/layers/embed_head.py - 词嵌入和语言模型头

本模块实现了支持张量并行的词嵌入层和语言模型头。
这两个层都涉及词表维度，在张量并行时需要特殊处理。

词表并行策略：
- 将词表按张量并行度切分，每个 GPU 只存储部分词表
- 输入时需要检查 token id 是否在当前分片范围内
- 输出时需要通过 all_reduce 或 gather 汇聚结果

例如（4路并行，词表大小 100000）：
- GPU 0: 词表 [0, 25000)
- GPU 1: 词表 [25000, 50000)
- GPU 2: 词表 [50000, 75000)
- GPU 3: 词表 [75000, 100000)

关键技术：
- 掩码机制：处理不在当前分片的 token
- all_reduce：汇聚 embedding 结果
- gather：汇聚 logits 结果
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    词表并行词嵌入层
    
    将词表按张量并行度切分，每个进程只存储和计算部分词嵌入。
    
    Attributes:
        tp_rank (int): 当前进程的 rank
        tp_size (int): 张量并行度
        num_embeddings (int): 总词表大小
        num_embeddings_per_partition (int): 每个分片的词表大小
        vocab_start_idx (int): 当前分片的起始词表索引
        vocab_end_idx (int): 当前分片的结束词表索引
        weight (nn.Parameter): 词嵌入权重 [num_embeddings_per_partition, embedding_dim]
    """

    def __init__(
        self,
        num_embeddings: int,   # 总词表大小
        embedding_dim: int,    # 嵌入维度
    ):
        super().__init__()
        # 获取张量并行信息
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # 确保词表大小可以均匀切分
        assert num_embeddings % self.tp_size == 0
        
        self.num_embeddings = num_embeddings
        # 每个分片的大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 当前分片负责的词表范围
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        
        # 权重参数（只存储当前分片）
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 设置权重加载器（用于从 checkpoint 加载）
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        加载权重（从完整权重中提取当前分片）
        
        Args:
            param: 目标参数
            loaded_weight: 完整的词嵌入权重 [num_embeddings, embedding_dim]
        """
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        # 提取当前分片
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        前向传播
        
        Args:
            x: 输入 token ids [...]
        
        Returns:
            torch.Tensor: 词嵌入 [..., embedding_dim]
        
        计算流程（多 GPU）：
        1. 创建掩码，标记当前分片负责的 token
        2. 将超出范围的 token id 映射到 0（避免越界）
        3. 查表获取 embedding
        4. 将非本分片的 embedding 置零
        5. all_reduce 汇聚所有分片的结果
        """
        if self.tp_size > 1:
            # 创建掩码：标记在当前分片范围内的 token
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 映射到本地索引（不在范围内的映射到 0）
            x = mask * (x - self.vocab_start_idx)
        
        # 词嵌入查表
        y = F.embedding(x, self.weight)
        
        if self.tp_size > 1:
            # 将非本分片的 embedding 置零
            y = mask.unsqueeze(1) * y
            # 汇聚所有分片的结果
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    词表并行语言模型头
    
    继承自 VocabParallelEmbedding，将隐藏状态映射到词表上的 logits。
    支持词嵌入权重共享（tie_word_embeddings）。
    
    与 Embedding 的区别：
    - 使用 linear 而非 embedding（矩阵乘法 vs 查表）
    - 在 prefill 时只取每个序列最后一个 token
    - 输出需要 gather 到 rank 0（只在 rank 0 采样）
    """

    def __init__(
        self,
        num_embeddings: int,   # 词表大小
        embedding_dim: int,    # 隐藏层维度
        bias: bool = False,    # 是否有偏置（LM Head 通常没有）
    ):
        assert not bias  # LM Head 不使用偏置
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        前向传播
        
        Args:
            x: 隐藏状态 [num_tokens, hidden_size]
        
        Returns:
            torch.Tensor: logits [num_seqs, vocab_size]（只在 rank 0 返回完整）
        
        计算流程：
        1. Prefill 时：只取每个序列最后一个 token
        2. 线性变换得到 logits
        3. 多 GPU 时 gather 到 rank 0
        """
        context = get_context()
        
        if context.is_prefill:
            # Prefill 模式：只取每个序列的最后一个 token 的隐藏状态
            # cu_seqlens_q[1:] - 1 得到每个序列最后一个 token 的索引
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        
        # 线性变换: [num_seqs, hidden_size] @ [hidden_size, vocab_per_partition] -> [num_seqs, vocab_per_partition]
        logits = F.linear(x, self.weight)
        
        if self.tp_size > 1:
            # 多 GPU：gather 所有分片的 logits 到 rank 0
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            # 拼接得到完整 logits
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        
        return logits
