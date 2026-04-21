"""
nanovllm/engine/llm_engine.py - LLM 推理引擎核心实现

本模块是 Nano-vLLM 的核心，负责协调整个推理流程：
1. 初始化模型运行器（ModelRunner）和调度器（Scheduler）
2. 管理多进程张量并行
3. 处理请求的添加、调度和执行
4. 组织生成循环并返回结果

架构概述：
┌─────────────────────────────────────────────────────────────┐
│                        LLMEngine                             │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐│
│  │  Tokenizer  │  │  Scheduler  │  │    ModelRunner(s)    ││
│  │  (分词器)   │  │  (调度器)   │  │  (模型执行器)        ││
│  └─────────────┘  └─────────────┘  └──────────────────────┘│
└─────────────────────────────────────────────────────────────┘

推理流程：
1. add_request(): 将用户输入转换为 Sequence 并加入等待队列
2. schedule(): 调度器决定本轮执行哪些序列（prefill 或 decode）
3. run(): ModelRunner 执行前向传播和采样
4. postprocess(): 更新序列状态，判断是否完成
5. 循环 2-4 直到所有序列完成
"""

import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    LLM 推理引擎类
    
    负责管理整个推理系统的生命周期，包括：
    - 多进程初始化与通信
    - 请求管理
    - 生成循环控制
    
    Attributes:
        ps (list): 子进程列表（用于张量并行）
        events (list): 进程间同步事件
        model_runner (ModelRunner): 主进程的模型执行器
        tokenizer: HuggingFace 分词器
        scheduler (Scheduler): 请求调度器
    """

    def __init__(self, model, **kwargs):
        """
        初始化 LLM 推理引擎
        
        Args:
            model (str): 模型路径
            **kwargs: 配置参数，会被过滤并传递给 Config
        
        初始化流程：
        1. 解析配置参数，创建 Config 对象
        2. 启动子进程（如果 tensor_parallel_size > 1）
        3. 创建主进程的 ModelRunner
        4. 加载 tokenizer
        5. 创建 Scheduler
        6. 注册退出清理函数
        """
        # ==================== 1. 配置解析 ====================
        # 从 Config 数据类中获取所有有效字段名
        config_fields = {field.name for field in fields(Config)}
        # 过滤 kwargs，只保留 Config 支持的参数
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        
        # ==================== 2. 多进程初始化（张量并行）====================
        self.ps = []      # 子进程列表
        self.events = []  # 同步事件列表
        # 使用 spawn 方式创建子进程（CUDA 要求）
        ctx = mp.get_context("spawn")
        
        # 为每个非主进程创建子进程
        # rank 0 是主进程，rank 1~n-1 是子进程
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # 创建进程间同步事件
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        # ==================== 3. 主进程 ModelRunner ====================
        # rank 0 的 ModelRunner 在主进程中创建
        # 它持有所有子进程的 events，用于广播命令
        self.model_runner = ModelRunner(config, 0, self.events)
        
        # ==================== 4. 加载 Tokenizer ====================
        # use_fast=True 使用 Rust 实现的快速分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # 从 tokenizer 获取 EOS token id
        config.eos = self.tokenizer.eos_token_id
        
        # ==================== 5. 创建调度器 ====================
        self.scheduler = Scheduler(config)
        
        # ==================== 6. 注册退出清理 ====================
        # 确保程序退出时正确清理资源（销毁进程组、释放显存等）
        atexit.register(self.exit)

    def exit(self):
        """
        清理资源并退出
        
        执行顺序：
        1. 向所有进程发送退出命令
        2. 销毁主进程的 ModelRunner
        3. 等待所有子进程结束
        """
        self.model_runner.call("exit")  # 广播退出命令
        del self.model_runner            # 触发析构
        for p in self.ps:
            p.join()                     # 等待子进程结束

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加推理请求
        
        Args:
            prompt: 输入提示，可以是字符串或 token id 列表
            sampling_params: 采样参数
        
        处理流程：
        1. 如果是字符串，使用 tokenizer 编码为 token ids
        2. 创建 Sequence 对象封装请求
        3. 将 Sequence 加入调度器的等待队列
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        执行一步推理
        
        Returns:
            outputs: 本步完成的序列列表，每项为 (seq_id, completion_token_ids)
            num_tokens: 本步处理的 token 数
                - 正数：prefill 阶段处理的 token 总数
                - 负数：decode 阶段处理的序列数的负值
        
        执行流程：
        1. 调度：决定执行哪些序列，是 prefill 还是 decode
        2. 执行：调用 ModelRunner 进行前向传播和采样
        3. 后处理：更新序列状态，检查是否完成
        """
        # 调度：获取本轮要执行的序列和执行模式
        seqs, is_prefill = self.scheduler.schedule()
        # 执行：前向传播 + 采样
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 后处理：更新缓存状态、追加 token、检查完成条件
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 收集已完成序列的输出
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # 计算吞吐量统计用的 token 数
        # prefill: 实际调度执行的 token 数（正数）
        # decode: 序列数的负值（负数表示 decode 模式）
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        """检查是否所有请求都已完成"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量生成文本
        
        Args:
            prompts: 输入提示列表，可以是字符串列表或 token id 列表的列表
            sampling_params: 采样参数，可以是单个（应用于所有请求）或列表（每个请求独立）
            use_tqdm: 是否显示进度条
        
        Returns:
            输出列表，每项包含:
                - "text": 生成的文本
                - "token_ids": 生成的 token id 列表
            输出顺序与输入顺序一致
        
        执行流程：
        1. 初始化进度条（可选）
        2. 添加所有请求到调度器
        3. 循环执行 step() 直到所有请求完成
        4. 按原始顺序整理输出
        5. 解码 token ids 为文本
        """
        # ==================== 1. 初始化 ====================
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        # 如果 sampling_params 是单个对象，复制为列表
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # ==================== 2. 添加请求 ====================
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        # ==================== 3. 生成循环 ====================
        outputs = {}  # seq_id -> token_ids
        prefill_throughput = decode_throughput = 0.
        
        while not self.is_finished():
            t = perf_counter()  # 计时开始
            output, num_tokens = self.step()
            
            # 更新吞吐量统计（用于进度条显示）
            if use_tqdm:
                if num_tokens > 0:  # prefill 阶段
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:  # decode 阶段
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # 收集完成的输出
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        
        # ==================== 4. 整理输出 ====================
        # 按 seq_id 排序，保证输出顺序与输入一致
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 解码 token ids 为文本
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        
        if use_tqdm:
            pbar.close()
        return outputs
