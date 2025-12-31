"""
nanovllm/engine/scheduler.py - 请求调度器

本模块实现了 Nano-vLLM 的核心调度逻辑，负责决定每一步执行哪些序列。
调度器是连接请求管理和模型执行的桥梁。

调度策略概述：
1. Prefill 优先：如果有等待中的新请求，优先进行 prefill（处理完整 prompt）
2. Decode 批处理：没有新 prefill 时，对所有运行中的序列进行 decode（生成下一个 token）
3. 资源约束：受 max_num_seqs 和 max_num_batched_tokens 限制
4. 抢占机制：当 KV 缓存不足时，抢占最新的序列释放资源

状态转换图：
    WAITING ──(allocate)──> RUNNING ──(finish)──> FINISHED
       ↑                        │
       └────(preempt)───────────┘

两阶段推理：
- Prefill 阶段：处理完整的 prompt，计算所有 token 的 KV cache
- Decode 阶段：基于已有 KV cache，每次生成一个新 token
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    请求调度器
    
    管理请求队列，决定每步执行哪些序列，协调 KV 缓存分配。
    
    Attributes:
        max_num_seqs (int): 单批次最大序列数
        max_num_batched_tokens (int): 单批次最大 token 数
        eos (int): 结束符 token id
        block_manager (BlockManager): KV 缓存块管理器
        waiting (deque): 等待队列，存放待 prefill 的序列
        running (deque): 运行队列，存放正在 decode 的序列
    """

    def __init__(self, config: Config):
        """
        初始化调度器
        
        Args:
            config: 全局配置对象
        """
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        # 创建块管理器，管理 KV 缓存的分配和释放
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # 等待队列：新添加的请求在这里等待 prefill
        self.waiting: deque[Sequence] = deque()
        # 运行队列：已完成 prefill，正在 decode 的序列
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """检查是否所有请求都已完成"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """
        添加新请求到等待队列
        
        Args:
            seq: 要添加的序列
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        执行调度，决定本轮执行哪些序列
        
        Returns:
            (scheduled_seqs, is_prefill): 
                - scheduled_seqs: 本轮要执行的序列列表
                - is_prefill: True 表示 prefill 阶段，False 表示 decode 阶段
        
        调度逻辑：
        1. 首先尝试调度等待队列中的序列进行 prefill
        2. 如果没有可 prefill 的序列，则调度运行队列中的序列进行 decode
        3. 如果 KV 缓存不足，会触发抢占机制
        """
        # ==================== Prefill 调度 ====================
        # 优先处理新请求的 prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            
            # 检查资源约束：
            # 1. 总 token 数不超过 max_num_batched_tokens
            # 2. BlockManager 能分配足够的 KV 块
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break  # 资源不足，停止添加更多序列
            
            num_seqs += 1
            # 分配 KV 缓存块（包括前缀缓存查找）
            self.block_manager.allocate(seq)
            # 实际需要计算的 token 数 = 总长度 - 缓存命中的 token 数
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            # 更新状态为 RUNNING
            seq.status = SequenceStatus.RUNNING
            # 从等待队列移到运行队列
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        
        # 如果有序列被调度进行 prefill，返回
        if scheduled_seqs:
            return scheduled_seqs, True

        # ==================== Decode 调度 ====================
        # 没有新的 prefill 任务，对运行中的序列进行 decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            
            # 检查是否可以追加新的 KV（可能需要新的块）
            while not self.block_manager.can_append(seq):
                # KV 缓存不足，需要抢占序列释放资源
                if self.running:
                    # 抢占运行队列尾部的序列（最新添加的）
                    self.preempt(self.running.pop())
                else:
                    # 没有其他序列可抢占，只能抢占当前序列
                    self.preempt(seq)
                    break
            else:
                # 成功分配资源，将序列加入本轮调度
                num_seqs += 1
                # 可能需要追加新的 KV 块
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        
        # 确保至少有一个序列被调度
        assert scheduled_seqs
        # 将调度的序列重新放回运行队列头部
        # 使用 extendleft + reversed 保持原有顺序
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占序列，释放其 KV 缓存
        
        当 KV 缓存不足以继续 decode 时，需要抢占一些序列。
        被抢占的序列会回到等待队列，等待重新 prefill。
        
        Args:
            seq: 要抢占的序列
        
        注意：
        - 抢占是"懒惰"的，只释放 KV 缓存，不保存中间状态
        - 被抢占的序列需要完全重新 prefill
        - 抢占策略：LIFO（后进先出），优先抢占最新的序列
        """
        seq.status = SequenceStatus.WAITING
        # 释放 KV 缓存块
        self.block_manager.deallocate(seq)
        # 放到等待队列头部，优先重新调度
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        后处理：更新序列状态，检查是否完成
        
        Args:
            seqs: 本轮执行的序列列表
            token_ids: 对应的采样结果（每个序列一个 token id）
        
        处理逻辑：
        1. 将新生成的 token 追加到序列
        2. 检查是否遇到 EOS 或达到最大长度
        3. 如果完成，释放 KV 缓存并从运行队列移除
        """
        for seq, token_id in zip(seqs, token_ids):
            # 追加新 token
            seq.append_token(token_id)
            
            # 检查是否完成
            # 条件1：遇到 EOS（如果未设置 ignore_eos）
            # 条件2：达到最大生成长度
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                # 释放 KV 缓存
                self.block_manager.deallocate(seq)
                # 从运行队列移除
                self.running.remove(seq)
