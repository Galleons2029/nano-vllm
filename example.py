"""
example.py - Nano-vLLM 使用示例

本文件演示了如何使用 Nano-vLLM 进行大语言模型推理。
主要展示了:
1. 如何初始化 LLM 引擎
2. 如何配置采样参数
3. 如何格式化对话模板
4. 如何执行批量推理并获取结果
"""

import os
from nanovllm import LLM, SamplingParams  # 导入 Nano-vLLM 的核心类
from transformers import AutoTokenizer     # 用于加载 HuggingFace tokenizer


def main():
    # ==================== 1. 初始化配置 ====================
    # 模型路径，使用 expanduser 展开 ~ 为用户主目录
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    # 加载 tokenizer，用于将文本转换为 token 序列
    # use_fast=True（默认）使用 Rust 实现的快速分词器
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    # 初始化 LLM 推理引擎
    # - enforce_eager=True: 禁用 CUDA Graph 优化，使用即时执行模式（更稳定，便于调试）
    # - tensor_parallel_size=1: 单 GPU 运行，不启用张量并行
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # ==================== 2. 配置采样参数 ====================
    # SamplingParams 控制文本生成行为:
    # - temperature=0.6: 温度系数，较低的值使输出更确定性，较高的值增加随机性
    # - max_tokens=256: 最大生成 token 数量
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    
    # ==================== 3. 准备输入提示词 ====================
    # 原始用户输入
    prompts = [
        "introduce yourself",              # 自我介绍
        "list all prime numbers within 100",  # 列出100以内的质数
    ]
    
    # 使用 chat template 将原始提示词格式化为模型期望的对话格式
    # apply_chat_template 会将用户消息包装成特定格式，例如:
    # <|im_start|>user\nintroduce yourself<|im_end|>\n<|im_start|>assistant\n
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],  # 构建单轮对话消息
            tokenize=False,                          # 返回字符串而非 token ids
            add_generation_prompt=True,              # 添加助手回复的起始标记
        )
        for prompt in prompts
    ]
    
    # ==================== 4. 执行推理 ====================
    # generate 方法支持批量处理多个提示词
    # 返回值是一个列表，每个元素包含 "text" 和 "token_ids" 字段
    outputs = llm.generate(prompts, sampling_params)

    # ==================== 5. 输出结果 ====================
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")           # 打印格式化后的提示词
        print(f"Completion: {output['text']!r}")  # 打印生成的文本


if __name__ == "__main__":
    main()
