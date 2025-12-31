"""
nanovllm/engine/sequence.py - 序列状态管理

本模块定义了请求序列的数据结构和状态管理。
每个用户请求会被封装为一个 Sequence 对象，贯穿整个推理过程。

序列包含：
- 唯一标识（seq_id）
- 状态信息（WAITING/RUNNING/FINISHED）
- Token 数据（prompt tokens + completion tokens）
- KV 缓存信息（block_table）
- 采样参数（temperature, max_tokens 等）

序列生命周期：
1. 创建：用户提交请求，创建 Sequence，状态为 WAITING
2. Prefill：调度器分配资源，状态变为 RUNNING
3. Decode：循环生成 token，状态保持 RUNNING
4. 完成：遇到 EOS 或达到 max_tokens，状态变为 FINISHED
5. 可能的抢占：资源不足时回到 WAITING

序列化支持：
- __getstate__ 和 __setstate__ 方法支持跨进程传输
- 只传输必要的数据以减少通信开销
"""

from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    序列状态枚举
    
    WAITING: 等待 prefill，在调度器的等待队列中
    RUNNING: 正在执行（prefill 或 decode），在运行队列中
    FINISHED: 已完成生成，等待返回结果
    """
    WAITING = auto()   # 等待处理
    RUNNING = auto()   # 正在运行
    FINISHED = auto()  # 已完成


class Sequence:
    """
    推理序列类
    
    封装单个推理请求的所有状态信息。
    
    类属性:
        block_size (int): KV 缓存块大小，默认 256
        counter (itertools.count): 全局序列 ID 计数器，确保唯一性
    
    实例属性:
        seq_id (int): 序列唯一标识
        status (SequenceStatus): 当前状态
        token_ids (list[int]): 所有 token ids（prompt + completion）
        last_token (int): 最后一个 token，用于 decode 阶段的输入
        num_tokens (int): 当前总 token 数
        num_prompt_tokens (int): prompt 的 token 数（固定不变）
        num_cached_tokens (int): 已缓存的 token 数（前缀缓存命中）
        block_table (list[int]): KV 缓存块 ID 列表
        temperature (float): 采样温度
        max_tokens (int): 最大生成 token 数
        ignore_eos (bool): 是否忽略 EOS
    """
    
    block_size = 256        # 块大小，与 BlockManager 保持一致
    counter = count()       # 全局 ID 计数器

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """
        初始化序列
        
        Args:
            token_ids: prompt 的 token id 列表
            sampling_params: 采样参数
        """
        # 分配唯一 ID
        self.seq_id = next(Sequence.counter)
        # 初始状态为等待
        self.status = SequenceStatus.WAITING
        # 复制 token_ids 避免外部修改影响
        self.token_ids = copy(token_ids)
        # 记录最后一个 token（用于 decode 阶段）
        self.last_token = token_ids[-1]
        # 当前总 token 数
        self.num_tokens = len(self.token_ids)
        # prompt token 数（固定不变）
        self.num_prompt_tokens = len(token_ids)
        # 缓存命中的 token 数（prefill 时计算）
        self.num_cached_tokens = 0
        # KV 缓存块表
        self.block_table = []
        # 从采样参数复制配置
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """返回当前总 token 数"""
        return self.num_tokens

    def __getitem__(self, key):
        """支持下标访问 token_ids"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """是否已完成生成"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的 completion token 数"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """获取 prompt 部分的 token ids"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """获取 completion 部分的 token ids（生成的内容）"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """缓存命中的完整块数"""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """
        总块数（向上取整）
        
        公式: ceil(num_tokens / block_size)
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """
        最后一个块中的 token 数
        
        范围: 1 到 block_size
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        获取第 i 个块的 token ids
        
        Args:
            i: 块索引，范围 [0, num_blocks)
        
        Returns:
            list[int]: 该块的 token id 列表
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """
        追加新生成的 token
        
        Args:
            token_id: 新生成的 token id
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        序列化状态（用于跨进程传输）
        
        优化：只传输必要的数据
        - 如果还没开始生成（prefill 阶段），传输完整 token_ids
        - 如果已开始生成（decode 阶段），只传输 last_token
        
        Returns:
            tuple: 序列化的状态数据
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """
        反序列化状态
        
        Args:
            state: 序列化的状态数据
        
        注意：
        - prefill 阶段需要完整 token_ids 用于计算
        - decode 阶段只需要 last_token 作为输入
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            # prefill 阶段：恢复完整 token_ids
            self.token_ids = state[-1]
        else:
            # decode 阶段：只需要 last_token
            self.last_token = state[-1]
