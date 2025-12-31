"""
nanovllm/llm.py - LLM 类定义

本模块定义了 Nano-vLLM 的顶层用户接口类 LLM。
LLM 类直接继承自 LLMEngine，提供与 vLLM 兼容的 API。

设计说明：
- 这里使用简单的继承而非组合，是因为 LLM 就是 LLMEngine 的用户友好包装
- 未来可以在此添加更多高级功能，如流式输出、异步接口等

使用方式：
    from nanovllm import LLM, SamplingParams
    llm = LLM(model_path, **config_kwargs)
    outputs = llm.generate(prompts, sampling_params)
"""

from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """
    LLM 推理引擎的用户接口类
    
    继承自 LLMEngine，提供完整的文本生成功能：
    - 模型加载与初始化
    - 批量请求处理
    - 调度与内存管理
    - 采样与解码
    
    参数:
        model (str): 模型路径，指向 HuggingFace 格式的模型目录
        **kwargs: 其他配置参数，参见 Config 类
            - max_num_batched_tokens: 单批最大 token 数
            - max_num_seqs: 单批最大序列数
            - max_model_len: 模型上下文长度上限
            - tensor_parallel_size: 张量并行分片数
            - enforce_eager: 是否禁用 CUDA Graph
            - gpu_memory_utilization: GPU 显存利用率
    """
    pass
