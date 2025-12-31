"""
bench.py - Nano-vLLM 吞吐量基准测试

本文件用于测试 Nano-vLLM 推理引擎的吞吐性能，通过生成大量随机序列来模拟实际工作负载。
测试指标：
- 总生成 token 数
- 总耗时
- 吞吐量（tokens/s）

测试配置（默认）:
- 256 个请求序列
- 输入长度: 100-1024 tokens 随机
- 输出长度: 100-1024 tokens 随机
"""

import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams  # 使用 Nano-vLLM
# from vllm import LLM, SamplingParams    # 取消注释可切换到官方 vLLM 进行对比


def main():
    # ==================== 1. 测试参数配置 ====================
    seed(0)                    # 固定随机种子，保证可复现性
    num_seqs = 256             # 测试序列数量
    max_input_len = 1024       # 最大输入长度
    max_ouput_len = 1024       # 最大输出长度

    # ==================== 2. 初始化 LLM 引擎 ====================
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # enforce_eager=False: 启用 CUDA Graph 优化以获得最佳性能
    # max_model_len=4096: 设置模型支持的最大上下文长度
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # ==================== 3. 生成随机测试数据 ====================
    # 生成随机 token id 序列作为输入
    # 每个序列的长度在 100 到 max_input_len 之间随机
    # token id 在 0-10000 范围内随机（覆盖词表的常用部分）
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    
    # 为每个序列生成独立的采样参数
    # - temperature=0.6: 中等随机性
    # - ignore_eos=True: 忽略结束符，强制生成到 max_tokens
    # - max_tokens: 每个序列的输出长度随机
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    
    # 如果使用官方 vLLM，需要将 prompt_token_ids 包装成字典格式
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # ==================== 4. 预热运行 ====================
    # 先执行一次小规模推理，触发模型编译、CUDA Graph 捕获等初始化操作
    # 避免这些开销影响正式测试的计时
    llm.generate(["Benchmark: "], SamplingParams())
    
    # ==================== 5. 正式基准测试 ====================
    t = time.time()  # 记录开始时间
    # 批量生成，use_tqdm=False 关闭进度条以避免额外开销
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)  # 计算总耗时
    
    # ==================== 6. 统计并输出结果 ====================
    # 计算总输出 token 数（由于 ignore_eos=True，实际输出等于 max_tokens 的总和）
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    # 计算吞吐量
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
