"""
nanovllm/engine/model_runner.py - 模型执行器

本模块负责实际的模型推理执行，是 Nano-vLLM 的计算核心。
主要功能包括：
1. 模型加载与初始化
2. KV 缓存分配
3. 输入数据准备（prefill 和 decode）
4. 模型前向传播
5. Token 采样
6. CUDA Graph 优化
7. 多进程通信（张量并行）

架构说明：
- 每个 GPU 运行一个 ModelRunner 实例
- rank 0（主进程）负责调度和采样
- rank 1~n-1（子进程）只负责前向传播
- 通过共享内存进行跨进程通信

优化技术：
- Flash Attention：高效的注意力计算
- CUDA Graph：减少 kernel 启动开销
- torch.compile：编译优化小算子
- 前缀缓存：复用相同前缀的 KV
"""

import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    模型执行器类
    
    负责加载模型、管理 KV 缓存、执行推理。
    支持单 GPU 和多 GPU 张量并行。
    
    Attributes:
        config (Config): 全局配置
        block_size (int): KV 缓存块大小
        enforce_eager (bool): 是否禁用 CUDA Graph
        world_size (int): 张量并行总进程数
        rank (int): 当前进程的 rank
        event: 进程同步事件
        model: 加载的模型
        sampler: Token 采样器
        kv_cache: KV 缓存张量
        graphs: CUDA Graph 字典（batch size -> graph）
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化模型执行器
        
        Args:
            config: 全局配置
            rank: 当前进程的 rank（0 为主进程）
            event: 同步事件
                - rank 0: 事件列表（用于通知子进程）
                - rank > 0: 单个事件（用于接收通知）
        
        初始化流程：
        1. 设置基本配置
        2. 初始化分布式进程组
        3. 加载模型并设置权重
        4. 预热模型
        5. 分配 KV 缓存
        6. 捕获 CUDA Graph（如果启用）
        7. 设置共享内存（多进程时）
        """
        # ==================== 1. 基本配置 ====================
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # ==================== 2. 分布式初始化 ====================
        # 使用 NCCL 后端进行 GPU 间通信
        # tcp://localhost:2333 用于进程间初始化握手
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)  # 绑定当前进程到对应 GPU
        
        # ==================== 3. 加载模型 ====================
        # 保存默认 dtype，加载完成后恢复
        default_dtype = torch.get_default_dtype()
        # 设置模型使用的数据类型（通常是 bfloat16 或 float16）
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        # 创建模型实例
        self.model = Qwen3ForCausalLM(hf_config)
        # 从 safetensors 文件加载权重
        load_model(self.model, config.model)
        # 创建采样器
        self.sampler = Sampler()
        
        # ==================== 4. 预热模型 ====================
        # 预热触发内核编译和内存分配，避免首次推理延迟
        self.warmup_model()
        
        # ==================== 5. 分配 KV 缓存 ====================
        # 根据可用显存计算并分配 KV 缓存
        self.allocate_kv_cache()
        
        # ==================== 6. CUDA Graph 捕获 ====================
        # CUDA Graph 可以减少 kernel 启动开销
        if not self.enforce_eager:
            self.capture_cudagraph()
        
        # 恢复默认设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # ==================== 7. 共享内存设置（多进程）====================
        if self.world_size > 1:
            if rank == 0:
                # 主进程创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)  # 1MB
                dist.barrier()  # 等待所有进程就绪
            else:
                # 子进程等待共享内存创建完成后连接
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                # 子进程进入事件循环，等待主进程的命令
                self.loop()

    def exit(self):
        """
        清理资源并退出
        
        执行顺序：
        1. 关闭共享内存
        2. 清理 CUDA Graph
        3. 同步 CUDA
        4. 销毁分布式进程组
        """
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()  # 等待所有进程完成清理
            if self.rank == 0:
                self.shm.unlink()  # 只有主进程删除共享内存
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """
        子进程事件循环
        
        子进程（rank > 0）在此循环中等待主进程的命令，
        执行相应的方法，直到收到 exit 命令。
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """
        从共享内存读取命令
        
        共享内存格式：
        - 前 4 字节：数据长度（小端整数）
        - 后续字节：pickle 序列化的数据
        
        Returns:
            (method_name, args): 方法名和参数
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()  # 等待主进程发送信号
        # 读取数据长度
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 反序列化数据
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()  # 重置事件
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        向共享内存写入命令
        
        Args:
            method_name: 要调用的方法名
            *args: 方法参数
        """
        assert self.world_size > 1 and self.rank == 0
        # 序列化数据
        data = pickle.dumps([method_name, *args])
        n = len(data)
        # 写入长度和数据
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        # 通知所有子进程
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """
        调用方法（支持跨进程广播）
        
        如果是主进程且有多个进程，先通过共享内存广播命令，
        然后执行本地方法。
        
        Args:
            method_name: 方法名
            *args: 参数
        
        Returns:
            方法返回值
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
        预热模型
        
        执行一次最大规模的前向传播，触发：
        - CUDA kernel 编译
        - 内存分配和优化
        - torch.compile 编译
        
        这样后续推理不会因为首次编译而产生延迟。
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        # 计算最大序列数：不超过配置限制
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        # 创建虚拟序列进行预热
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)  # 执行 prefill
        
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        分配 KV 缓存
        
        根据可用 GPU 显存计算可以分配的 KV 块数量，
        然后创建统一的 KV 缓存张量并绑定到各层。
        
        KV 缓存布局：
        [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        - 2: K 和 V 两个缓存
        - num_layers: Transformer 层数
        - num_blocks: 总块数
        - block_size: 每块的 token 数
        - num_kv_heads: KV 头数（可能与 Q 头数不同，GQA）
        - head_dim: 每个头的维度
        """
        config = self.config
        hf_config = config.hf_config
        
        # ==================== 计算可用显存 ====================
        free, total = torch.cuda.mem_get_info()
        used = total - free
        # peak: 峰值显存（预热时的最大值）
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        # current: 当前显存（模型权重等）
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        # ==================== 计算单个块的字节数 ====================
        # 张量并行时，每个进程只存储部分 KV 头
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 单个块的字节数 = 2(K+V) * 层数 * 块大小 * KV头数 * 头维度 * dtype大小
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        
        # ==================== 计算可分配的块数 ====================
        # 可用显存 = 总显存 * 利用率 - 已使用 - (峰值 - 当前)
        # (峰值 - 当前) 是预留给推理过程中的临时内存
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        
        # ==================== 分配 KV 缓存张量 ====================
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        
        # ==================== 绑定到各层注意力模块 ====================
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]  # K 缓存
                module.v_cache = self.kv_cache[1, layer_id]  # V 缓存
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        准备块表张量
        
        将多个序列的 block_table 合并为一个批量张量，
        用于 Flash Attention 的块化 KV 访问。
        
        Args:
            seqs: 序列列表
        
        Returns:
            torch.Tensor: 形状 [num_seqs, max_num_blocks] 的块表
        """
        # 找到最大块数，用于 padding
        max_len = max(len(seq.block_table) for seq in seqs)
        # 将所有 block_table 对齐到相同长度（用 -1 填充）
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # 使用 pin_memory 和 non_blocking 加速 CPU->GPU 传输
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        准备 prefill 阶段的输入
        
        Prefill 阶段处理完整的 prompt，需要准备：
        1. input_ids: 需要计算的 token ids（排除已缓存的）
        2. positions: 每个 token 的位置编码索引
        3. cu_seqlens: 累积序列长度（用于变长注意力）
        4. slot_mapping: 每个 token 对应的 KV 缓存槽位
        5. block_tables: 用于前缀缓存时的块表
        
        Args:
            seqs: 序列列表
        
        Returns:
            (input_ids, positions): GPU 上的输入张量
        """
        input_ids = []      # 输入 token ids
        positions = []      # 位置索引
        cu_seqlens_q = [0]  # Q 的累积序列长度
        cu_seqlens_k = [0]  # K 的累积序列长度（包含缓存）
        max_seqlen_q = 0    # Q 的最大序列长度
        max_seqlen_k = 0    # K 的最大序列长度
        slot_mapping = []   # KV 槽位映射
        block_tables = None
        
        for seq in seqs:
            seqlen = len(seq)
            start = min(seq.num_cached_tokens, seqlen - 1)
            seqlen_q = seq.num_scheduled_tokens
            seqlen_k = seqlen
            end = start + seqlen_q
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))

            # 更新累积长度
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # warmup 时没有 block_table
            if not seq.block_table:
                continue

            # 计算 slot_mapping：每个需要计算的 token 对应的 KV 缓存位置
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        
        # 如果有前缀缓存（K 的总长度 > Q 的总长度），需要 block_tables
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        
        # 转换为 GPU 张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # 设置全局上下文供注意力层使用
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        准备 decode 阶段的输入
        
        Decode 阶段每次只生成一个 token，输入更简单：
        1. input_ids: 每个序列的最后一个 token
        2. positions: 每个 token 的位置索引
        3. slot_mapping: 新 token 要写入的 KV 槽位
        4. context_lens: 每个序列的上下文长度（用于 attention）
        5. block_tables: KV 块表
        
        Args:
            seqs: 序列列表
        
        Returns:
            (input_ids, positions): GPU 上的输入张量
        """
        input_ids = []     # 每个序列的最后一个 token
        positions = []     # 位置索引
        slot_mapping = []  # 新 token 的 KV 槽位
        context_lens = []  # 上下文长度
        
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # 新 token 写入最后一个块的下一个槽位
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        
        # 转换为 GPU 张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        
        # 设置全局上下文
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """
        准备采样参数
        
        Args:
            seqs: 序列列表
        
        Returns:
            torch.Tensor: 温度参数张量
        """
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        执行模型前向传播
        
        Args:
            input_ids: 输入 token ids
            positions: 位置索引
            is_prefill: 是否为 prefill 阶段
        
        Returns:
            torch.Tensor: logits 输出
        
        逻辑说明：
        - Prefill 阶段或禁用 CUDA Graph 时：直接执行前向传播
        - Decode 阶段且启用 CUDA Graph：使用预捕获的图
        - Batch size > 512 时：不使用 CUDA Graph（显存考虑）
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 直接执行前向传播
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 使用 CUDA Graph
            bs = input_ids.size(0)
            context = get_context()
            
            # 找到大于等于当前 batch size 的最小预捕获图
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            
            # 将输入复制到预分配的缓冲区
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)  # 先清空
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            
            # 重放图
            graph.replay()
            
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        执行一步推理
        
        Args:
            seqs: 序列列表
            is_prefill: 是否为 prefill 阶段
        
        Returns:
            list[int]: 采样得到的 token id 列表（只有 rank 0 返回）
        """
        # 准备输入
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 只有主进程准备采样参数（只有主进程执行采样）
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        
        # 模型前向传播
        logits = self.run_model(input_ids, positions, is_prefill)
        
        # 采样（只有主进程）
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        
        # 重置上下文
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        捕获 CUDA Graph
        
        为常见的 batch size 预捕获 CUDA Graph，
        在 decode 阶段可以通过重放图来减少 kernel 启动开销。
        
        捕获的 batch size: [1, 2, 4, 8, 16, 32, ..., max_bs]
        
        CUDA Graph 工作原理：
        1. 预先运行一遍计算，记录所有 CUDA 操作
        2. 后续通过 replay() 重放，跳过 Python 和 CUDA 启动开销
        
        限制：
        - 图捕获后输入大小固定
        - 不支持动态控制流
        - 需要使用固定的输入输出缓冲区
        """
        config = self.config
        hf_config = config.hf_config
        
        # 最大 batch size（受配置和内存限制）
        max_bs = min(self.config.max_num_seqs, 512)
        # 最大块数
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        # ==================== 分配固定缓冲区 ====================
        # 这些缓冲区在图捕获和重放中共享
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        
        # 要捕获的 batch size 列表
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # ==================== 逐个捕获图 ====================
        # 从大到小捕获，共享内存池
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            
            # 设置上下文
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # 预热运行（确保所有内存都已分配）
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 捕获图
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 第一个图创建内存池，后续图共享
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存缓冲区引用供 run_model 使用
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
