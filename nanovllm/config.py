"""
nanovllm/config.py - 全局配置类

本模块定义了 Nano-vLLM 引擎的全局配置，包括：
- 模型相关配置（路径、上下文长度等）
- 批处理配置（最大 batch 大小、最大 token 数等）
- 内存管理配置（GPU 显存利用率、KV 缓存块大小等）
- 并行配置（张量并行度）
- 性能优化配置（是否启用 CUDA Graph）

这些配置项对推理性能和显存占用有重要影响，应根据具体硬件和使用场景调整。
"""

import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """
    Nano-vLLM 全局配置类
    
    Attributes:
        model (str): 模型路径，必须是有效的本地目录，包含 HuggingFace 格式的模型文件
        
        max_num_batched_tokens (int): 单次前向传播的最大 token 数
            - 包括 prompt tokens 和生成的 tokens
            - 较大的值可以提高 GPU 利用率，但需要更多显存
            - 必须 >= max_model_len（单个请求可能占满整个批次）
            
        max_num_seqs (int): 单批次最大序列数
            - 控制并发处理的请求数量
            - 较大的值可以提高吞吐，但可能增加延迟
            
        max_model_len (int): 模型支持的最大上下文长度
            - 会与 HuggingFace 配置中的 max_position_embeddings 取最小值
            - 影响 RoPE 位置编码的预计算范围
            
        gpu_memory_utilization (float): GPU 显存利用率，范围 (0, 1]
            - 用于计算可分配的 KV 缓存块数量
            - 较高的值可以缓存更多 KV，但可能导致 OOM
            - 建议设置为 0.9 左右，预留一些显存给其他操作
            
        tensor_parallel_size (int): 张量并行分片数，范围 [1, 8]
            - 将模型权重分片到多个 GPU 上
            - 需要 GPU 数量 >= tensor_parallel_size
            - 对于 Qwen3-0.6B 这样的小模型，通常设为 1
            
        enforce_eager (bool): 是否强制使用即时执行模式
            - True: 禁用 CUDA Graph，每次都重新启动 CUDA kernel
            - False: 启用 CUDA Graph 优化，减少 kernel 启动开销
            - 调试时建议设为 True，正式运行设为 False
            
        hf_config (AutoConfig): HuggingFace 模型配置
            - 在 __post_init__ 中自动从模型目录加载
            - 包含模型架构信息（层数、头数、隐藏层大小等）
            
        eos (int): 结束符 token id
            - 在 LLMEngine 初始化时从 tokenizer 获取
            - 用于判断生成是否完成
            
        kvcache_block_size (int): KV 缓存块大小（token 数）
            - 必须是 256 的倍数（Flash Attention 的对齐要求）
            - 较大的块可以减少块管理开销，但可能浪费显存
            
        num_kvcache_blocks (int): KV 缓存总块数
            - -1 表示自动计算（根据可用显存）
            - 在 ModelRunner.allocate_kv_cache 中设置
    """
    model: str                                  # 模型路径（必需参数）
    max_num_batched_tokens: int = 16384        # 单批最大 token 数
    max_num_seqs: int = 512                     # 单批最大序列数
    max_model_len: int = 4096                   # 模型最大上下文长度
    gpu_memory_utilization: float = 0.9         # GPU 显存利用率
    tensor_parallel_size: int = 1               # 张量并行分片数
    enforce_eager: bool = False                 # 是否禁用 CUDA Graph
    hf_config: AutoConfig | None = None         # HuggingFace 模型配置
    eos: int = -1                               # 结束符 token id
    kvcache_block_size: int = 256               # KV 缓存块大小
    num_kvcache_blocks: int = -1                # KV 缓存总块数

    def __post_init__(self):
        """
        配置后处理与验证
        
        执行以下操作：
        1. 验证模型路径是否为有效目录
        2. 验证 KV 缓存块大小是否为 256 的倍数
        3. 验证张量并行度是否在有效范围内
        4. 加载 HuggingFace 模型配置
        5. 调整 max_model_len 不超过模型支持的最大位置
        6. 验证 max_num_batched_tokens 足够大
        """
        # 验证模型路径
        assert os.path.isdir(self.model)
        # KV 缓存块大小必须是 256 的倍数（Flash Attention 的对齐要求）
        assert self.kvcache_block_size % 256 == 0
        # 张量并行度范围检查
        assert 1 <= self.tensor_parallel_size <= 8
        # 加载 HuggingFace 配置
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # max_model_len 不能超过模型的位置编码支持范围
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        # max_num_batched_tokens 必须能容纳单个最长序列
        assert self.max_num_batched_tokens >= self.max_model_len
