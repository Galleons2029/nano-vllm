"""
nanovllm/engine/block_manager.py - KV 缓存块管理器

本模块实现了 Nano-vLLM 的 KV 缓存管理系统，核心功能包括：
1. KV 缓存的块化管理（分块分配，避免内存碎片）
2. 前缀缓存（Prefix Caching）：相同前缀共享 KV 缓存
3. 引用计数：支持多个序列共享同一个块
4. 哈希索引：快速查找已缓存的前缀

设计思想：
- 将 KV 缓存划分为固定大小的块（默认 256 tokens/块）
- 使用哈希表索引已完成的块，实现前缀缓存
- 引用计数管理块的生命周期
- 未完成的块（最后一个块）不参与缓存共享

块状态：
- 空闲块：ref_count = 0，在 free_block_ids 中
- 已分配块：ref_count >= 1，在 used_block_ids 中
- 已完成块：hash != -1，可被后续序列复用

前缀缓存示意：
    序列A: [block0] [block1] [block2]
    序列B: [block0] [block1] [block3]
                ↑       ↑
           共享的前缀块（ref_count=2）
"""

from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    KV 缓存块
    
    每个块存储固定数量 token 的 Key 和 Value 向量。
    
    Attributes:
        block_id (int): 块的唯一标识符
        ref_count (int): 引用计数，表示有多少序列在使用此块
        hash (int): 块内容的哈希值，-1 表示块未完成（内容可能变化）
        token_ids (list): 块中存储的 token id 列表，用于验证缓存命中
    """

    def __init__(self, block_id):
        """
        初始化块
        
        Args:
            block_id: 块的唯一 ID
        """
        self.block_id = block_id
        self.ref_count = 0    # 初始无引用
        self.hash = -1        # 未完成的块
        self.token_ids = []   # 空内容

    def update(self, hash: int, token_ids: list[int]):
        """
        更新块的哈希和内容
        
        当块被填满后调用此方法，标记块为已完成。
        
        Args:
            hash: 块内容的哈希值
            token_ids: 块中的 token id 列表
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """
        重置块状态
        
        当块被重新分配给新序列时调用。
        """
        self.ref_count = 1    # 新分配给一个序列
        self.hash = -1        # 标记为未完成
        self.token_ids = []   # 清空内容


class BlockManager:
    """
    KV 缓存块管理器
    
    负责管理 KV 缓存块的分配、释放和复用。
    实现前缀缓存以提高相似请求的处理效率。
    
    Attributes:
        block_size (int): 每个块的 token 容量
        blocks (list[Block]): 所有块的列表
        hash_to_block_id (dict): 哈希值到块 ID 的映射，用于前缀缓存查找
        free_block_ids (deque): 空闲块 ID 队列
        used_block_ids (set): 已使用块 ID 集合
    """

    def __init__(self, num_blocks: int, block_size: int):
        """
        初始化块管理器
        
        Args:
            num_blocks: 总块数
            block_size: 每块的 token 容量
        """
        self.block_size = block_size
        # 创建所有块对象
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 哈希索引：用于快速查找已缓存的块
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲块队列（初始所有块都是空闲的）
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 已使用块集合
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算块内容的哈希值
        
        使用 xxhash 算法，支持链式哈希以处理前缀依赖。
        
        Args:
            token_ids: 块中的 token id 列表
            prefix: 前一个块的哈希值，-1 表示这是第一个块
        
        Returns:
            int: 64 位哈希值
        
        哈希链示意：
            block0_hash = hash(tokens0)
            block1_hash = hash(block0_hash + tokens1)
            block2_hash = hash(block1_hash + tokens2)
        """
        h = xxhash.xxh64()
        if prefix != -1:
            # 将前缀哈希作为输入的一部分
            h.update(prefix.to_bytes(8, "little"))
        # 将 token ids 转换为字节并计算哈希
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        分配一个空闲块
        
        Args:
            block_id: 要分配的块 ID
        
        Returns:
            Block: 分配的块对象
        
        Raises:
            AssertionError: 如果块的引用计数不为 0
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0  # 确保块是空闲的
        block.reset()
        # 从空闲队列移除
        self.free_block_ids.remove(block_id)
        # 加入已使用集合
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """
        释放一个块
        
        Args:
            block_id: 要释放的块 ID
        
        Raises:
            AssertionError: 如果块的引用计数不为 0
        """
        assert self.blocks[block_id].ref_count == 0
        # 从已使用集合移除
        self.used_block_ids.remove(block_id)
        # 加入空闲队列
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """
        检查是否可以为序列分配 KV 缓存
        
        Args:
            seq: 要检查的序列
        
        Returns:
            bool: 如果有足够的空闲块返回 True
        
        注意：
        - 这里使用最坏情况估计（假设没有缓存命中）
        - 实际分配时可能因缓存命中而使用更少的块
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为序列分配 KV 缓存块
        
        这是前缀缓存的核心逻辑：
        1. 遍历序列的每个块位置
        2. 计算块内容的哈希值
        3. 检查是否有缓存命中
        4. 命中则复用已有块，未命中则分配新块
        
        Args:
            seq: 要分配缓存的序列
        
        前缀缓存逻辑：
        - 连续的块通过链式哈希关联
        - 一旦某个块未命中，后续所有块都需要重新计算
        - cache_miss 标志用于跳过后续的缓存查找
        """
        assert not seq.block_table  # 确保序列还没有分配块
        
        h = -1              # 上一个块的哈希值
        cache_miss = False  # 是否已经发生缓存未命中
        
        for i in range(seq.num_blocks):
            # 获取当前块的 token ids
            token_ids = seq.block(i)
            
            # 只有完整的块（填满 block_size）才计算哈希
            # 最后一个不完整的块 hash=-1，不参与缓存
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            
            # 查找缓存
            block_id = self.hash_to_block_id.get(h, -1)
            
            # 验证缓存是否真正命中
            # 需要检查 token_ids 是否完全匹配（防止哈希冲突）
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True  # 标记未命中，后续块都需要重新计算
            
            if cache_miss:
                # 缓存未命中：分配新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中：复用已有块
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # 块已被其他序列使用，增加引用计数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 块在空闲队列中（之前被释放但保留了内容）
                    block = self._allocate_block(block_id)
            
            # 如果是完整块，更新哈希索引
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            
            # 将块 ID 添加到序列的块表
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """
        释放序列的 KV 缓存块
        
        按逆序释放块，减少引用计数，
        当引用计数归零时释放块到空闲队列。
        
        Args:
            seq: 要释放缓存的序列
        
        注意：
        - 释放的块可能保留在哈希索引中
        - 后续序列仍可能通过缓存查找复用这些块
        """
        # 逆序释放，因为后面的块依赖前面的块
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        
        # 重置序列的缓存状态
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        检查是否可以追加新的 KV（decode 阶段）
        
        Args:
            seq: 要检查的序列
        
        Returns:
            bool: 如果可以追加返回 True
        
        逻辑说明：
        - 只有当序列长度 % block_size == 1 时（刚跨入新块）
        - 才需要分配新块
        - 其他情况使用现有块的剩余空间
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        可能追加新块（decode 阶段）
        
        根据序列长度决定是否需要分配新块或更新现有块的哈希。
        
        Args:
            seq: 要追加的序列
        
        三种情况：
        1. len(seq) % block_size == 1: 刚跨入新块，需要分配
        2. len(seq) % block_size == 0: 刚填满一个块，更新哈希
        3. 其他: 块未填满，无需操作
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        
        if len(seq) % self.block_size == 1:
            # 情况1：序列长度刚跨入新块（当前块只有1个token）
            # 说明上一个块已经完成，需要分配新块
            assert last_block.hash != -1  # 上一个块应该已完成
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
            
        elif len(seq) % self.block_size == 0:
            # 情况2：当前块刚好填满
            # 计算哈希并更新索引
            assert last_block.hash == -1  # 块应该还未完成
            token_ids = seq.block(seq.num_blocks-1)
            # 获取前一个块的哈希作为前缀
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
            
        else:
            # 情况3：块未填满，继续使用
            assert last_block.hash == -1  # 确保块还未完成
