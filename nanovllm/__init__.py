"""
nanovllm/__init__.py - Nano-vLLM 包的入口模块

Nano-vLLM 是一个轻量级的 vLLM 实现，用约 1200 行 Python 代码复刻了 vLLM 的核心功能。
提供高效的离线 LLM 推理能力，支持：
- 前缀缓存 (Prefix Caching)
- 张量并行 (Tensor Parallelism)
- CUDA Graph 优化
- torch.compile 编译加速

导出的主要类：
- LLM: 大语言模型推理引擎，用于执行文本生成
- SamplingParams: 采样参数配置，控制生成行为（温度、最大长度等）

使用示例：
    from nanovllm import LLM, SamplingParams
    llm = LLM("/path/to/model")
    outputs = llm.generate(["Hello"], SamplingParams())
"""

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
