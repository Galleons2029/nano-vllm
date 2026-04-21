"""
nanovllm/engine/scheduler.py - 请求调度器

本模块实现了 Nano-vLLM 的核心调度逻辑，负责决定每一步执行哪些序列。
调度器是连接请求管理和模型执行的桥梁。

调度策略概述：
1. Prefill 优先：如果有等待中的新请求，优先进行 prefill
2. Chunked Prefill：当批量 token 预算不足时，只允许首个序列分块 prefill
3. Decode 批处理：没有新 prefill 时，对运行中的序列统一 decode
4. 抢占机制：当 KV 缓存不足时，抢占最新的序列释放资源

状态转换图：
    WAITING ──(allocate)──> RUNNING ──(finish)──> FINISHED
       ↑                        │
       └────(preempt)───────────┘

两阶段推理：
- Prefill 阶段：处理 prompt，并支持将超长 prompt 分多轮完成
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
        # 优先处理新请求或被抢占后重新进入等待队列的请求
        scheduled_seqs = []
        num_batched_tokens = 0

        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]

            # 剩余还需要计算的 token 数；decode 前至少保留 1 个 token 进入模型
            num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0 or (not seq.block_table and not self.block_manager.can_allocate(seq)):
                break
            # 只允许首个序列做 chunked prefill，避免打散整个批次
            if remaining < num_tokens and scheduled_seqs:
                break

            if not seq.block_table:
                self.block_manager.allocate(seq)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            if seq.num_scheduled_tokens == num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
            num_batched_tokens += seq.num_scheduled_tokens
        
        # 如果有序列被调度进行 prefill，返回
        if scheduled_seqs:
            return scheduled_seqs, True

        # ==================== Decode 调度 ====================
        # 没有新的 prefill 任务，对运行中的序列进行 decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
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
                seq.num_scheduled_tokens = 1
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

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """
        后处理：更新序列状态，检查是否完成
        
        Args:
            seqs: 本轮执行的序列列表
            token_ids: 对应的采样结果（每个序列一个 token id）
            is_prefill: 本轮是否为 prefill
        
        处理逻辑：
        1. 将新生成的 token 追加到序列
        2. 检查是否遇到 EOS 或达到最大长度
        3. 如果完成，释放 KV 缓存并从运行队列移除
        """
        for seq, token_id in zip(seqs, token_ids):
            if is_prefill:
                # prefill 只是在补齐已有 prompt 的 KV cache，不一定会立刻生成新 token
                seq.num_cached_tokens = min(seq.num_cached_tokens + seq.num_scheduled_tokens, seq.num_tokens)
                if seq.num_cached_tokens < seq.num_tokens or seq.num_completion_tokens > 0:
                    seq.num_scheduled_tokens = 0
                    continue

            # prefill 完成或 decode 阶段，都会真正追加一个新 token
            seq.append_token(token_id)
            seq.num_cached_tokens += 1
            seq.num_scheduled_tokens = 0
            
            # 检查是否完成
            # 条件1：遇到 EOS（如果未设置 ignore_eos）
            # 条件2：达到最大生成长度
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                # 释放 KV 缓存
                self.block_manager.deallocate(seq)
                # 从运行队列移除
                self.running.remove(seq)
